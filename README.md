# Distributed Training with PyTorch

Repo for short learning scripts from https://maven.com/walk-with-code/scratch-to-scale, adapted and extended (ie profiling insertions, comments, modal scripts) for my own noodling. 

Built for learning with velocity, not polish. Expect hacks. 

---

## Scattered Learnings

- [torch.distributed.barrier](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.barrier) is implemented as an all_reduce of a 1-element tensor
    - https://github.com/pytorch/pytorch/pull/140785: will show as `nccl:all_reduce_barrier` in the future


### Zero
- zero1.py with standard Adam: 5 nccl:all_reduce from barrier calls
- zero1.py with custom ShardedOptimizer wrapper of Adam: adds 60 all_reduce (for grad) and 60 broadcast (for param) for the 12 parameters in 6 linear layers
- zero2 toy script has higher peak memory (need to use bucketed reduce-scatter) than zero1 but shows the drop at steady-state
- zero3.py: adds 60 all_reduce (optimizer step) and 120 all_gather (forward pre hook and backward pre hook)

### fsdp with A100-80GB:2 (no activation checkpointing)
- Static memory at rest (SmolLM3-3B) ~12GB: 3.1GB param shard (BF16) + 6.2 GB AdamW state (BF16) + fsdp metadata / flat params + CUDA caching allocator fragmentation
- Per step peak 
    - during each module forward: static shards, full params for 1 FSDP group, activations across all layers so far
        - Forward activations scale linearly with sequence length (8192)
        - across 36 layers ~10-20GB
    - final loss
        - `aten::_to_copy (aten::empty_strided)` allocates FP32 logits (~4GB: 8192 x 128k x 4)
        - `aten::_log_softmax` allocates log probs (~4GB) and frees FP32 logits
    - gradient at the start of backward
        - `aten::nll_loss_backward` allocates grad wrt log probs (~4GB)
        - `attn::log_softmax_backward_data` calculates grad wrt logits (~4GB, can reuse) and frees log probs and grad wrt log probs
