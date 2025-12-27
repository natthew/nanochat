from collections import deque
import json
import gzip

import torch

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    Stream pretraining text from jsonl files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    BATCH_SIZE = 1024  # number of documents per batch (similar to parquet row_groups)

    def document_batches():
        jsonl_paths = list_parquet_files()
        assert len(jsonl_paths) != 0, "No dataset jsonl files found, did you run dataset.py?"
        jsonl_paths = jsonl_paths[:-1] if split == "train" else jsonl_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        first_pass = True
        pq_idx = resume_pq_idx # we kick off jsonl files at the resume index (or by default just 0)
        while True: # iterate infinitely (multi-epoch)
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(jsonl_paths): # iterate over all jsonl files
                filepath = jsonl_paths[pq_idx]

                # Determine if file is gzipped
                is_gzipped = filepath.endswith('.gz')
                open_fn = gzip.open if is_gzipped else open
                mode = 'rt' if is_gzipped else 'r'

                # Read and batch the jsonl file
                with open_fn(filepath, mode, encoding='utf-8') as f:
                    # Count total batches for this file
                    # Start from resume point if resuming on same file, otherwise from DDP rank
                    if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                        base_idx = resume_rg_idx // ddp_world_size # in units of ddp_world_size
                        base_idx += 1 # advance by 1 so that we definitely don't repeat data after resuming
                        rg_idx = base_idx * ddp_world_size + ddp_rank
                        resume_rg_idx = None # set to None as we only want to do this a single time
                    else:
                        rg_idx = ddp_rank

                    batch = []
                    batch_idx = 0
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            doc = json.loads(line)
                            batch.append(doc['text'])

                            # When we've accumulated BATCH_SIZE documents, process this batch
                            if len(batch) >= BATCH_SIZE:
                                # Check if this is a batch we should process (DDP)
                                if batch_idx >= rg_idx and (batch_idx - rg_idx) % ddp_world_size == 0:
                                    # the tokenizer encode might want to go in even smaller batches, e.g. 128 rows
                                    for i in range(0, len(batch), tokenizer_batch_size):
                                        yield batch[i:i+tokenizer_batch_size], (pq_idx, batch_idx)
                                batch = []
                                batch_idx += 1
                        except (json.JSONDecodeError, KeyError):
                            # Skip malformed lines
                            continue

                    # Process any remaining documents in the last batch
                    if batch and batch_idx >= rg_idx and (batch_idx - rg_idx) % ddp_world_size == 0:
                        for i in range(0, len(batch), tokenizer_batch_size):
                            yield batch[i:i+tokenizer_batch_size], (pq_idx, batch_idx)

                pq_idx += 1 # advance to the next jsonl file
            first_pass = False
    batches = document_batches()

    # Now emit batches of tokens.
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) # in PyTorch, long=int64
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # we need this in case we wish to approximately resume training
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
