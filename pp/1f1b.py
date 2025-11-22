"""
Implements One-Forward-One-Backward PP

Kinds include:
1. GPipe (All forward -> all backward)
2. 1F1B (One forward, one backward)
3. Interleaved 1F1B 
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
    for epoch in range(10):
        x = torch.randn(batch_size, 50)
        y = torch.randn(batch_size, 50)
        loss = run_pipe(x, y)
        print(f"epoch {epoch}: loss = {loss:.4f}")
