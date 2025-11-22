"""
Implements basic ZeRO-1 (optimizer sharding) with torch profiling

2 separate runs: baseline (standard Adam) and sharded (standard Adam wrapped in ShardedOptimizer).

The Setup
1. Initializes distribution training: `torch.distributed.init_process_group("nccl")` and sets the rank and world size.
    - Setup profiler scheduler on rank 0: start at 8 (skip_first + wait + warmup), active for 5, repeat 1 (8,9,10,11,12).
2. Builds a synthetic MLP (6 layer stack of 10k x 10k Linear + ReLU).
3. Runs dummy data generation and warmup training step to avoid first-step overhead. 
    - `record_function` blocks for "data_generation".
    - Prints initial state memory stats (before training loop, after warmup step).
4. Runs a short synthetic MSE training loop (20 steps for better profiling). 
    - Profiles memory and step structure with `torch.profiler`. 
      `record_function` blocks for per-step: forward, backward, optimizer_step_total, and zero_grad.
    - Prints peak memory per step and final.

Communications: within each training step
- dist.barrier() at the end of each step to ensure profiling captures each step independently
- ShardedOptimizer: within optimizer.step()
    - all_reduce_gradients
    - broadcast_parameters
"""

import os
import sys
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam, Optimizer

# Add training_utils to path
sys.path.append(str(Path(__file__).parent.parent))
from training_utils.memory import print_memory_stats
from training_utils.utils import get, set_seed
from torch.profiler import profile, record_function, ProfilerActivity, schedule

set_seed(42)

class ShardedOptimizer:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.original_param_groups = optimizer.param_groups
        self.params = [
            param for group in self.original_param_groups for param in group["params"]
        ]
    
        world_size = get('ws')
        rank = get('rank')
        
        # Shard parameters across GPUs
        params_per_rank = len(self.params) // world_size
        remainder = len(self.params) % world_size
        
        start_idx = rank * params_per_rank + min(rank, remainder)
        end_idx = start_idx + params_per_rank + (1 if rank < remainder else 0)
        
        self.local_param_indices = list(range(start_idx, end_idx))
        self.local_params = set(self.params[i] for i in self.local_param_indices)
        
        # Remove non-local parameters from optimizer
        self._shard_optimizer_params()

        self.broadcast_count = 0
        self.communication_time = 0.0
        self.step_time = 0.0

    def _shard_optimizer_params(self):
        """Remove non-local parameters from optimizer param groups"""
        for group in self.optimizer.param_groups:
            group['params'] = [p for p in group['params'] if p in self.local_params]

    def step(self, closure=None):
        """All-reduce gradients, then step optimizer"""
        step_start = time.perf_counter()

        with record_function("all_reduce_gradients"):
            for p in self.params:
                if p.grad is not None:
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    p.grad.data /= get("ws")

        with record_function("optimizer_step"):
            # Only step on local parameters
            self.optimizer.step(closure)

        # Broadcast updated params
        with record_function("broadcast_parameters"):
            params_per_rank = len(self.params) // get('ws')
            remainder = len(self.params) % get('ws')

            for i, p in enumerate(self.params):
                # Recompute owner rank for this param index
                if i < (params_per_rank + 1) * remainder:
                    owner_rank = i // (params_per_rank + 1)
                else:
                    owner_rank = (i - remainder) // params_per_rank

                dist.broadcast(p.data, src=owner_rank)
        
        torch.cuda.synchronize()
        self.step_time += time.perf_counter() - step_start

    def zero_grad(self):
        self.optimizer.zero_grad()


def train(model, optimizer, device, is_sharded=False, profiler_context=None):
    rank = get("rank")
    batch_size = 16
    
    with record_function("data_generation"):
        x = torch.randn(batch_size, 10000, device=device)
        y = torch.randn(batch_size, 10000, device=device)

    # Warmup step to avoid first-step overhead
    optimizer.zero_grad()
    output = model(x)
    loss = nn.functional.mse_loss(output, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    # Reset timers and counters after warmup
    if is_sharded:
        optimizer.communication_time = 0.0
        optimizer.step_time = 0.0
        optimizer.broadcast_count = 0

    if rank == 0:
        print_memory_stats("Initial state", model, optimizer, rank, device)
    dist.barrier()

    peak_memories = []
    num_steps = 20  # More steps for better profiling
    
    for i in range(num_steps):
        if i > num_steps:
            break
            
        torch.cuda.reset_peak_memory_stats(device)
        
        with record_function("zero_grad"):
            optimizer.zero_grad()
        
        with record_function("forward"):
            output = model(x)
            loss = nn.functional.mse_loss(output, y)

        # Print memory before backward
        if rank == 0 and i == 0:
            print(f"\nStep {i} memory:")
            print(
                f"Before backward: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB"
            )

        with record_function("backward"):
            loss.backward()
            torch.cuda.synchronize()

        # Print gradient memory after backward
        if rank == 0 and i == 0:
            grad_memory = sum(
                p.grad.numel() * p.grad.element_size() / 1024**2
                for p in model.parameters()
                if p.grad is not None
            )
            print(f"Gradient memory after backward: {grad_memory:.2f} MB")

        with record_function("optimizer_step_total"):
            optimizer.step()
        
        if profiler_context:
            profiler_context.step()
            
        current_peak = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_memories.append(current_peak)

        if rank == 0 and i == 0:
            print(f"Peak memory this step: {current_peak:.2f} MB")
        dist.barrier()

    if rank == 0:
        print(f"\nFinal peak memory: {max(peak_memories):.2f} MB")

    # Add timing stats at the end
    if is_sharded and rank == 0:
        avg_step_time = optimizer.step_time / num_steps
        avg_comm_time = optimizer.communication_time / num_steps
        print("\nTiming and Communication Stats:")
        print("-" * 40)
        print(f"Average step time: {avg_step_time:.3f}s")
        print(f"Average communication time: {avg_comm_time:.3f}s")
        print(f"Average compute time: {avg_step_time - avg_comm_time:.3f}s")
        print(f"Communication overhead: {(avg_comm_time / avg_step_time) * 100:.1f}%")

    return model, optimizer, max(peak_memories)


def test_zero1():
    dist.init_process_group("nccl")
    rank = get("rank")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Get trace directory from environment variable
    trace_dir = os.environ.get("TRACE_DIR", "./profiler_traces")
    trace_path = Path(trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)

    # Setup profiler for regular Adam (only on rank 0)
    profiler_context = None
    if rank == 0:
        profiler_schedule = schedule(
            skip_first=5,
            wait=1,
            warmup=2,
            active=5,
            repeat=1
        )
        profiler_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=profiler_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_path / "regular_adam")),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        )

    # Test with regular Adam
    print(f"\nGPU {rank} - Testing with regular Adam:")
    torch.cuda.reset_peak_memory_stats()
    model = nn.Sequential(
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
    ).to(device)
    regular_optimizer = Adam(model.parameters(), lr=0.001)
    
    if profiler_context:
        profiler_context.__enter__()
    
    model, regular_optimizer, peak_memory_adam = train(
        model, regular_optimizer, device, is_sharded=False, profiler_context=profiler_context
    )
    
    if profiler_context:
        profiler_context.__exit__(None, None, None)

    # Clear memory before testing sharded optimizer
    del model, regular_optimizer
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()

    # Setup profiler for ZeRO (only on rank 0)
    profiler_context_zero = None
    if rank == 0:
        profiler_schedule = schedule(
            skip_first=5,
            wait=1,
            warmup=2,
            active=5,
            repeat=1
        )
        profiler_context_zero = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=profiler_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_path / "zero1_adam")),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        )

    # Test with Sharded Adam
    print(f"\nGPU {rank} - Testing with Sharded Adam:")
    model = nn.Sequential(
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
    ).to(device)
    base_optimizer = Adam(model.parameters(), lr=0.001)
    sharded_optimizer = ShardedOptimizer(base_optimizer)
    
    if profiler_context_zero:
        profiler_context_zero.__enter__()
    
    model, sharded_optimizer, peak_memory_z1 = train(
        model, sharded_optimizer, device, is_sharded=True, profiler_context=profiler_context_zero
    )
    
    if profiler_context_zero:
        profiler_context_zero.__exit__(None, None, None)

    # Print final memory comparison
    if rank == 0:
        print("\nMemory Usage Summary:")
        print("-" * 40)
        print(f"Peak memory with regular Adam: {peak_memory_adam:.2f} MB")
        print(f"Peak memory with ZeRO-1 (sharded optimizer states): {peak_memory_z1:.2f} MB")
        print(
            f"Memory reduction: {(peak_memory_adam - peak_memory_z1):.2f} MB ({((peak_memory_adam - peak_memory_z1) / peak_memory_adam * 100):.2f}%)"
        )
        print("\nProfiler traces saved to:")
        print(f"  - {trace_path / 'regular_adam'}")
        print(f"  - {trace_path / 'zero1_adam'}")
        print(f"\nView with: tensorboard --logdir={trace_path}")


if __name__ == "__main__":
    test_zero1()
    dist.destroy_process_group()
