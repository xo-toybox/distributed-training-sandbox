"""
Implements basic ZeRO-3 (optimizer, grad, and param sharding with hooks)
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


# ZeRO3Hook
class Zero3ParamManager:
    """
    Tracks a parameter shard and gathers/releases full weight as needed.
    """
    def __init__(self, param, shard_idx, world_size, shard_dim=0):
        self.param = param
        self.shard_idx = shard_idx
        self.world_size = world_size
        self.shard_dim = shard_dim
        self.full_data = None

    def materialize(self):
        local_shard: torch.Tensor = self.param.data.contiguous()
        global_shards = [torch.empty_like(local_shard) for _ in range(get('ws'))]
        dist.all_gather(global_shards, local_shard)
        self.full_data = torch.cat(global_shards, dim=self.shard_dim)
        self.param.data = self.full_data

    def release(self):
        shards = self.param.data.chunk(get('ws'), dim=self.shard_dim)
        local_shard = shards[get('rank')].contiguous()
        self.param.data = local_shard
        if self.param.grad is not None and self.param.grad.shape != local_shard.shape:
            # Shrink/cut the grads to align with the local shard
            grad_shards = self.param.grad.data.chunk(get('ws'), dim=self.shard_dim)
            local_grad = grad_shards[get('rank')].contiguous()
            self.param.grad.data = local_grad
        self.full_data = None



def register_zero3_hooks(model, param_managers):
    """
    Attach hooks to modules so params are all-gathered before forward,
    and released after forward.
    """
    def pre_hook(module, inputs):
        for _, param in module.named_parameters(recurse=False):
            manager = param_managers.get(param)
            if manager is not None:
                manager.materialize()

    def post_hook(module, inputs, outputs):
        for _, param in module.named_parameters(recurse=False):
            manager = param_managers.get(param)
            if manager is not None:
                manager.release()

    for m in model.modules():
        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(post_hook)
        m.register_full_backward_pre_hook(pre_hook)
        m.register_full_backward_hook(post_hook)



class ShardedOptimizer:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.original_param_groups = optimizer.param_groups
        self.params = [
            param for group in self.original_param_groups for param in group["params"]
        ]

        world_size = get("ws")
        rank = get("rank")

        # Shard parameters across GPUs
        params_per_rank = len(self.params) // world_size
        remainder = len(self.params) % world_size

        start_idx = rank * params_per_rank + min(rank, remainder)
        end_idx = start_idx + params_per_rank + (1 if rank < remainder else 0)

        self.local_param_indices = list(range(start_idx, end_idx))
        self.local_params = set(self.params[i] for i in self.local_param_indices)

        # Replace param.data with shard AND replace in optimizer param_groups
        # Holds all the hooks
        self.param_managers = {}
        for param in self.params:
            shard_dim = 0
            chunks = param.data.chunk(get('ws'), dim=shard_dim)
            local_shard = chunks[get('rank')].contiguous()
            param.data = local_shard
            self.param_managers[param] = Zero3ParamManager(param, get('rank'), get('ws'), shard_dim)


        # Now fix optimizer param_groups: keep only local_params (shards)
        for group in self.optimizer.param_groups:
            group["params"] = [p for p in group["params"] if p in self.local_params]

        # Gradient hooks
        self.grad_hooks = {}

        self.communication_time = 0.0
        self.step_time = 0.0

    def step(self, closure=None):
        step_start = time.perf_counter()
        comm_start = step_start

        # Gradient reduce-scatter
        world_size = get("ws")
        rank = get("rank")

        for i, param in enumerate(self.params):
            grad = param.grad
            if grad is None:
                continue

            # Split grad into shards
            manager = self.param_managers[param]
            shard_dim = manager.shard_dim

            # If grad is full sized, shard!
            if grad.shape != param.data.shape:
                chunks = grad.data.chunk(get('ws'), dim=shard_dim)
                grad = chunks[get('rank')].contiguous()

            # All-reduce across ranks to average the shard
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad /= world_size

            # Delete grads ~after~ averaging
            for i in self.local_param_indices:
                param.grad = grad
            else:
                param.grad = None



        torch.cuda.synchronize()
        self.communication_time += time.perf_counter() - comm_start

        # Local optimizer step (now states match shard size)
        self.optimizer.step(closure)

        # No broadcast needed in ZeRO-3
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

    # Register ZeRO-3 hooks if sharded
    if is_sharded:
        register_zero3_hooks(model, optimizer.param_managers)

    # Warmup
    optimizer.zero_grad()
    output = model(x)
    loss = nn.functional.mse_loss(output, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    # Reset timers
    if is_sharded:
        optimizer.communication_time = 0.0
        optimizer.step_time = 0.0

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

        if rank == 0 and i == 0:
            print(f"\nStep {i} memory:")
            print(f"Before backward: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

        with record_function("backward"):
            loss.backward()
            torch.cuda.synchronize()

        if rank == 0 and i == 0:
            grad_memory = sum(
                p.grad.numel() * p.grad.element_size() / 1024**2
                for p in model.parameters() if p.grad is not None
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


def test_zero3():
    dist.init_process_group("nccl")
    rank = get("rank")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    set_seed(42)

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

    # Setup profiler for ZeRO-3 (only on rank 0)
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
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_path / "zero3_adam")),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        )

    print(f"\nGPU {rank} - Testing with ZeRO-3 Sharded Adam:")
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

    model, sharded_optimizer, peak_memory_z3 = train(
        model, sharded_optimizer, device, is_sharded=True, profiler_context=profiler_context_zero
    )

    if profiler_context_zero:
        profiler_context_zero.__exit__(None, None, None)

    # Print final memory comparison
    if rank == 0:
        print("\nMemory Usage Summary:")
        print("-" * 40)
        print(f"Peak memory with regular Adam: {peak_memory_adam:.2f} MB")
        print(f"Peak memory with ZeRO-3 (full sharding): {peak_memory_z3:.2f} MB")
        print(
            f"Memory reduction: {(peak_memory_adam - peak_memory_z3):.2f} MB ({((peak_memory_adam - peak_memory_z3) / peak_memory_adam * 100):.2f}%)"
        )
        print("\nProfiler traces saved to:")
        print(f"  - {trace_path / 'regular_adam'}")
        print(f"  - {trace_path / 'zero3_adam'}")
        print(f"\nView with: tensorboard --logdir={trace_path}")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    test_zero3()
