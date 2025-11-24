"""
Implements One-Forward-One-Backward (1F1B) Pipeline Parallelism

1F1B interleaves forward and backward passes to reduce memory usage and pipeline bubbles.
Unlike GPipe (all-forward then all-backward), 1F1B starts backward passes as soon as
the pipeline fills, allowing activations to be freed earlier.

Key characteristics:
- Clock-based scheduler: runs for (n_microbatches + n_stages - 1) ticks
- Each tick: all stages attempt both forward (if input ready) and backward (if grad ready)
- Memory efficient: activations freed sooner vs GPipe (peak ~n_stages microbatches vs ~n_microbatches)
- Same pipeline bubble as GPipe: 2*(n_stages-1), but better memory-time trade-off

Pipeline Parallelism Variants:
1. GPipe (All forward -> all backward) - see gpipe.py
2. 1F1B (One forward, one backward) - this implementation
3. Interleaved 1F1B (multiple stages per device)
4. Fill-Drain (similar to GPipe but with microbatches)
5. Async (Don't synchronize forward/backward passes)
"""

from accelerate.utils import set_seed

set_seed(42)

import torch
import torch.nn as nn
from collections import deque

# ---------- 1. Toy model ----------
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

# ---------- 2. Split into stages ----------
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

# ---------- 3. Micro-batch bookkeeping ----------
batch_size = 32
micro_batch_size = 8
n_micro = batch_size // micro_batch_size

# ---------- 4. Loss on last stage ----------
criterion = nn.MSELoss()

# ---------- 5. Optimizers (one per stage) ----------
opts = [torch.optim.Adam(s.parameters(), lr=1e-6) for s in stages]

# ---------- 6. Inter-stage queues ----------
# fwd[i]  == activations produced by stage i for stage i+1
# bwd[i+1]== gradients produced by stage i+1 for stage i
fwd_queues = [deque() for _ in range(n_stages)]
bwd_queues = [deque() for _ in range(n_stages)]

# ---------- 7. Saved tensors for backward ----------
# Map micro-batch id -> (input, output) tensors on each stage
saved_activations = [{} for _ in range(n_stages)]

def show_gradnorm(stage_id, stage):
    total = 0.0
    n = 0
    for p in stage.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item()
            n += 1
    # print(f"stage {stage_id}: {n} params have grad, total L2 = {total:.6f}")

# ---------- 8. Single-process scheduler ----------
def run_pipe(inputs, targets):
    # Zero grads everywhere
    for opt in opts:
        opt.zero_grad()

    # Split into micro-batches
    micro_ins  = inputs.chunk(n_micro)
    micro_tgts = targets.chunk(n_micro)

    # Push first micro-batches into stage-0 queue
    for mb_id, x in enumerate(micro_ins):
        fwd_queues[0].append((mb_id, x.to("cuda:0")))

    # Clock-cycle counter
    ticks = n_micro + n_stages - 1
    loss_total = 0.0
    loss_count = 0
    losses = {}  # Store losses for backward

    for clk in range(ticks):
        for stage_id in range(n_stages):
            dev = f"cuda:{stage_id}"
            
            # ---------- Forward ----------
            if fwd_queues[stage_id]:
                mb_id, x = fwd_queues[stage_id].popleft()
                x = x.detach().to(dev).requires_grad_(True)
                
                # Forward through stage
                out = stages[stage_id](x)
                
                # Save activations for backward
                saved_activations[stage_id][mb_id] = (x, out)
                
                if stage_id < n_stages - 1:  # send to next stage
                    fwd_queues[stage_id + 1].append((mb_id, out.detach().to(f"cuda:{stage_id + 1}")))
                else:  # last stage -> compute loss
                    y = micro_tgts[mb_id].to(dev)
                    loss = criterion(out, y) / n_micro
                    loss_total += loss.item()
                    loss_count += 1
                    losses[mb_id] = loss
                    # Start backward immediately
                    bwd_queues[stage_id].append((mb_id, None))

            # ---------- Backward ----------
            if bwd_queues[stage_id]:
                mb_id, grad_output = bwd_queues[stage_id].popleft()
                
                if stage_id == n_stages - 1 and grad_output is None:
                    # Last stage: backward from loss
                    retain = (mb_id != n_micro - 1)
                    loss = losses.pop(mb_id)
                    loss.backward(retain_graph=retain)
                    
                    # Get gradient w.r.t. stage output to send to previous stage
                    x, out = saved_activations[stage_id].pop(mb_id)
                    if stage_id > 0 and x.grad is not None:
                        bwd_queues[stage_id - 1].append((mb_id, x.grad.detach().to(f"cuda:{stage_id - 1}")))
                else:
                    # Intermediate stages: backward from next stage's gradient
                    x, out = saved_activations[stage_id].pop(mb_id)
                    
                    # Set output gradient and run backward
                    out.backward(gradient=grad_output)
                    
                    # Send input gradient to previous stage
                    if stage_id > 0 and x.grad is not None:
                        bwd_queues[stage_id - 1].append((mb_id, x.grad.detach().to(f"cuda:{stage_id - 1}")))
                
                show_gradnorm(stage_id, stages[stage_id])

    # Optimizer step on all stages
    for opt in opts:
        opt.step()

    return loss_total / loss_count if loss_count else None

# ---------- 9. Dummy data and run ----------
if __name__ == "__main__":
    import argparse
    import time
    import json

    parser = argparse.ArgumentParser(description="1F1B Pipeline Parallelism Training")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    losses = []
    epoch_times = []

    # Track peak memory per GPU
    torch.cuda.reset_peak_memory_stats()

    print(f"[1F1B] Starting training for {args.num_epochs} epochs")
    print(f"[1F1B] Config: {n_stages} stages, {n_micro} microbatches, batch_size={batch_size}")

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
            print(f"[1F1B] epoch {epoch}/{args.num_epochs}: loss={loss:.6f}, time={epoch_time:.3f}s")

    total_time = time.time() - start_time

    # Get peak memory for each GPU
    peak_memory_mb = {i: torch.cuda.max_memory_allocated(i) / 1024**2 for i in range(n_stages)}

    # Calculate metrics
    avg_loss = sum(losses) / len(losses)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    # Output results
    results = {
        "strategy": "1F1B",
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
    print("1F1B RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))
    print("="*60)
