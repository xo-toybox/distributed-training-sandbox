# DDP utilities

This directory contains the PyTorch DDP training example plus helper scripts for profiling runs on Modal GPUs and viewing the resulting traces locally.

## Components

- `modal_app.py` packages the repo into a Modal image and exposes the `run_ddp` function plus a local launcher. GPU shape is configured via `MODAL_GPU_SPEC`. The launcher creates a per-run directory inside `MODAL_TRACE_VOLUME` and sets `DDP_TRACE_DIR` so `ddp.py` writes its profiler output there.
- `ddp.py` is the actual training script: it boots a tiny distributed model via `accelerate.PartialState`, shards the dataset per-rank, runs a few profiled steps under `torch.profiler`, and writes traces into the `DDP_TRACE_DIR` supplied by the launcher.

## Prerequisites

1. Install project dependencies `uv pip install -r DDP/requirements.local.txt`
2. Authenticate: `modal token new`.
3. Verify/update `.env.public` configs

## Workflow

Use `DDP/scripts/profile.sh` to manage the full loop.

`profile.sh all` chains the three commands


```sh
./scripts/profile.sh run --run-name my-sweep
```
Runs on Modal and writes traces into
`${MODAL_TRACE_VOLUME}/<run_id>` inside the Modal volume.

```sh
./scripts/profile.sh sync /tmp/ddp_traces
```
Downloads the volume contents into the specified directory (defaults to
`${TRACE_OUTPUT_DIR}` from `.env.public`).

```sh
./scripts/profile.sh view --logdir /tmp/ddp_traces --port 7000
```
Starts TensorBoard pointing at the downloaded traces. 
