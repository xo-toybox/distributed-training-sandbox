"""
Implements GPipe Pipeline Parallelism

GPipe uses an all-forward-then-all-backward schedule for pipeline parallelism.
All microbatches complete forward passes through all stages before any backward pass begins.

Trade-off vs 1F1B: same pipeline bubble
+ Simpler scheduling logic
- Higher peak memory (~n_microbatches vs ~n_stages activations)

See 1f1b.py for a more memory-efficient alternative.
"""

from accelerate.utils import set_seed

set_seed(42)

import torch
import torch.nn as nn
from collections import deque

# Example small model for us to work with
model = nn.Sequential(
    nn.Linear(50, 500),
    nn.ReLU(),
    nn.Linear(500, 500),
    nn.ReLU(),
    nn.Linear(500, 500),
    nn.ReLU(),
    nn.Linear(500, 500),
    nn.ReLU(),
    nn.Linear(500, 500),
    nn.ReLU(),
    nn.Linear(500, 50),
)

# Split the model into 2 stages, each ~1/2 the model size
n_stages = 2
layers_per_stage = (len(model) + n_stages - 1) // n_stages
stages = []
start = 0
for i in range(n_stages):
    end = min(start + layers_per_stage, len(model))
    sub = nn.Sequential(*model[start:end]).to(f"cuda:{i}")
    sub.train()
    stages.append(sub)
    start = end


batch_size = 32
micro_batch_size = 8
n_micro = batch_size // micro_batch_size

loss_func = nn.MSELoss()

# We create an optimizer for every stage
opts = [torch.optim.Adam(stage.parameters(), lr=1e-6) for stage in stages]

# Inter-stage queues for us to go through and monitor throughout training
# fwd[i] is the activations produced by stage i for stage i+1
# bwd[i+1] is the grads produced by stage i+1 for stage i

fwd_queues = [deque() for _ in range(n_stages)]
bwd_queues = [deque() for _ in range(n_stages)]

# We need to save activations for the backward pass
# microbatch id -> inp/out for each stage
saved_activations = [{} for _ in range(n_stages)]


# Create a scheduler
def run_pipe(inputs, targets):
    # Zero grads everywhere
    for opt in opts:
        opt.zero_grad()

    # Split into micro-batches
    micro_ins  = inputs.chunk(n_micro)
    micro_tgts = targets.chunk(n_micro)

    # Store losses for backward
    losses = {}  
    loss_total = 0.0
    loss_count = 0

    # Forward phase, all microbatches through all stages, starting with stage0 queue
    
    for microbatch_id, x in enumerate(micro_ins):
        fwd_queues[0].append((microbatch_id, x.to("cuda:0", non_blocking=True)))
    
    # Process all forward passes
    for stage_id in range(n_stages):
        while fwd_queues[stage_id]:
            device = f"cuda:{stage_id}"
            microbatch_id, x = fwd_queues[stage_id].popleft()
            
            # For first stage, ensure input requires grad
            x = x.requires_grad_(True)
            
            # Forward through stage
            out = stages[stage_id](x)
            
            # Save activations for backward
            saved_activations[stage_id][microbatch_id] = (x, out)
            
            if stage_id < n_stages - 1:  # send to next stage
                # Move output to next GPU without breaking the graph
                out_next = out.to(f"cuda:{stage_id + 1}", non_blocking=True)
                fwd_queues[stage_id + 1].append((microbatch_id, out_next))
            else:  # last stage -> compute loss
                y = micro_tgts[microbatch_id].to(device)
                loss = loss_func(out, y) / n_micro
                loss_total += loss.item()
                loss_count += 1
                losses[microbatch_id] = loss
    
    # Backward phase: all microbatches go in reverse order
    # Start backward from all losses at last stage
    for microbatch_id in reversed(range(n_micro)):
        bwd_queues[n_stages - 1].append((microbatch_id, None))
    
    # Process all backward passes
    for stage_id in reversed(range(n_stages)):
        while bwd_queues[stage_id]:
            microbatch_id, grad_output = bwd_queues[stage_id].popleft()
            grad_output = grad_output.to(out.device) if grad_output is not None else None
            
            if stage_id == n_stages - 1 and grad_output is None:
                # Last stage: backward from loss
                retain = (microbatch_id != 0)
                loss = losses.pop(microbatch_id)
                loss.backward(retain_graph=retain)
                
                # Get gradient w.r.t. stage output to send to previous stage
                x, out = saved_activations[stage_id].pop(microbatch_id)
                if stage_id > 0 and x.grad is not None:
                    bwd_queues[stage_id - 1].append((microbatch_id, x.grad))
            else:
                # Intermediate stages: backward from next stage's gradient
                x, out = saved_activations[stage_id].pop(microbatch_id)
                
                # Set output gradient and run backward
                out.backward(gradient=grad_output)
                
                # Send input gradient to previous stage
                if stage_id > 0 and x.grad is not None:
                    bwd_queues[stage_id - 1].append((microbatch_id, x.grad))
    
    # Optimizer step on all stages
    for opt in opts:
        opt.step()
    
    return loss_total / loss_count if loss_count else None

if __name__ == "__main__":
    import argparse
    import time
    import json

    parser = argparse.ArgumentParser(description="GPipe Pipeline Parallelism Training")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    losses = []
    epoch_times = []

    # Track peak memory per GPU
    torch.cuda.reset_peak_memory_stats()

    print(f"[GPipe] Starting training for {args.num_epochs} epochs")
    print(f"[GPipe] Config: {n_stages} stages, {n_micro} microbatches, batch_size={batch_size}")

    start_time = time.time()
    for epoch in range(args.num_epochs):
        x = torch.randn(batch_size, 50)
        y = torch.randn(batch_size, 50)

        for i in range(n_stages):
            torch.cuda.synchronize(i)
        epoch_start = time.time()

        loss = run_pipe(x, y)

        for i in range(n_stages):
            torch.cuda.synchronize(i)
        epoch_end = time.time()

        epoch_time = epoch_end - epoch_start
        losses.append(loss)
        epoch_times.append(epoch_time)

        if epoch % max(1, args.num_epochs // 10) == 0 or epoch == args.num_epochs - 1:
            print(f"[GPipe] epoch {epoch}/{args.num_epochs}: loss={loss:.6f}, time={epoch_time:.3f}s")

    total_time = time.time() - start_time

    # Get peak memory for each GPU
    peak_memory_mb = {i: torch.cuda.max_memory_allocated(i) / 1024**2 for i in range(n_stages)}

    # Calculate metrics
    avg_loss = sum(losses) / len(losses)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    # Output results
    results = {
        "strategy": "GPipe",
        "num_epochs": args.num_epochs,
        "n_stages": n_stages,
        "n_microbatches": n_micro,
        "batch_size": batch_size,
        "final_loss": losses[-1],
        "avg_loss": avg_loss,
        "total_time_sec": total_time,
        "avg_epoch_time_sec": avg_epoch_time,
        "throughput_epochs_per_sec": 1 / avg_epoch_time,
        "peak_memory_mb": peak_memory_mb,
        "total_peak_memory_mb": sum(peak_memory_mb.values()),
    }

    print("\n" + "="*60)
    print("GPIPE RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))
    print("="*60)
