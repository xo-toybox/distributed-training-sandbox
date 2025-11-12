"""
Implements basic ZeRO-2 (optimizer & grad) sharding
"""

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

set_seed(42)

class Zero2Hook:
    """
    A hook which will discard gradients of parameters not on
    the current device (after sharding)
    """
    def __init__(self, param: torch.nn.Parameter, is_local_param: bool = False):
        self.param = param
        self.is_local_param = is_local_param

    def __call__(self, grad):
        if not self.is_local_param:
            return None
        return grad
    

class ShardedOptimizer:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.original_param_groups = optimizer.param_groups
        self.params = [
            param for group in self.original_param_groups for param in group["params"]
        ]
        
        # For ZeRO-2, we need to shard both parameters and optimizer states
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
        
        self.grad_hooks = {}
        self.register_gradient_hooks()

        self.broadcast_count = 0
        self.communication_time = 0.0
        self.step_time = 0.0

    def _shard_optimizer_params(self):
        """Remove non-local parameters from optimizer param groups"""
        for group in self.optimizer.param_groups:
            group['params'] = [p for p in group['params'] if p in self.local_params]

    def register_gradient_hooks(self):
        """Register hooks to shard gradients during backward"""
        for param in self.params:
            if param in self.local_params:
                # Local parameters keep their gradients
                hook = lambda grad: grad
            else:
                # Non-local parameters discard gradients
                hook = lambda grad: None
            
            handle = param.register_hook(hook)
            self.grad_hooks[param] = handle

    def step(self, closure=None):
        """All-reduce gradients, then step optimizer"""
        # Collect gradients for all parameters (including non-local ones)
        step_start = time.perf_counter()
        comm_start = step_start

        for i, param in enumerate(self.params):
            grad = param.grad
            if grad is None:
                continue
                
            flattened_grad = grad.data.contiguous().view(-1)

            # Build input chunks: each rank contributes its grad for this param
            # at every slot, so reduce_scatter will sum them and hand out one
            # shard per rank.
            in_tensor = torch.cat([flattened_grad for _ in range(get("ws"))], dim=0)

            output_tensor = torch.empty_like(flattened_grad)
            dist.reduce_scatter_tensor(output_tensor, in_tensor, op=dist.ReduceOp.SUM)

            # Keep only the grads of params in this rank
            if i in self.local_param_indices:
                param.grad.data = (output_tensor / get("ws")).view_as(grad.data)
            else:
                param.grad = None

        torch.cuda.synchronize()  # Ensure communication is done
        self.communication_time += time.perf_counter() - comm_start

        
        # Only step on local parameters
        self.optimizer.step(closure)

        # Broadcast updated params
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


def train(model, optimizer, device, is_sharded=False):
    rank = get("rank")
    batch_size = 16
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
        optimizer.all_reduce_count = 0
        optimizer.broadcast_count = 0

    if rank == 0:
        print_memory_stats("Initial state", model, optimizer, rank, device)
    dist.barrier()

    peak_memories = []
    for i in range(10):
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)

        # Print memory before backward
        if rank == 0 and i == 0:
            print(f"\nStep {i} memory:")
            print(
                f"Before backward: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB"
            )

        loss.backward()
        torch.cuda.synchronize()  # Ensure gradients are computed

        # Print gradient memory after backward
        if rank == 0 and i == 0:
            grad_memory = sum(
                p.grad.numel() * p.grad.element_size() / 1024**2
                for p in model.parameters()
                if p.grad is not None
            )
            print(f"Gradient memory after backward: {grad_memory:.2f} MB")

        optimizer.step()
        current_peak = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_memories.append(current_peak)

        if rank == 0 and i == 0:
            print(f"Peak memory this step: {current_peak:.2f} MB")
        dist.barrier()

    if rank == 0:
        print(f"\nFinal peak memory: {max(peak_memories):.2f} MB")

    # Add timing stats at the end
    if is_sharded and rank == 0:
        avg_step_time = optimizer.step_time / 10  # Average over 10 steps
        avg_comm_time = optimizer.communication_time / 10
        print("\nTiming and Communication Stats (averaged over 10 steps):")
        print("-" * 40)
        print(f"All-reduce operations per step: {optimizer.all_reduce_count // 10}")
        print(f"Broadcast operations per step: {optimizer.broadcast_count // 10}")
        print(
            f"Total communications: {optimizer.all_reduce_count + optimizer.broadcast_count}"
        )
        print(f"\nAverage step time: {avg_step_time:.3f}s")
        print(f"Average communication time: {avg_comm_time:.3f}s")
        print(f"Average compute time: {avg_step_time - avg_comm_time:.3f}s")
        print(f"Communication overhead: {(avg_comm_time / avg_step_time) * 100:.1f}%")

    return model, optimizer, max(peak_memories)


def test_zero2():
    dist.init_process_group("nccl")
    rank = get("rank")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

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
    model, regular_optimizer, peak_memory_adam = train(
        model, regular_optimizer, device, is_sharded=False
    )

    # Clear memory before testing sharded optimizer
    del model, regular_optimizer
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()

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
    model, sharded_optimizer, peak_memory_z1 = train(
        model, sharded_optimizer, device, is_sharded=True
    )

    # Print final memory comparison
    if rank == 0:
        print("\nMemory Usage Summary:")
        print("-" * 40)
        print(f"Peak memory with regular Adam: {peak_memory_adam:.2f} MB")
        print(f"Peak memory with sharded Adam: {peak_memory_z1:.2f} MB")
        print(
            f"Memory reduction: {(peak_memory_adam - peak_memory_z1):.2f} MB ({((peak_memory_adam - peak_memory_z1) / peak_memory_adam * 100):.2f}%)"
        )


if __name__ == "__main__":
    test_zero2()
