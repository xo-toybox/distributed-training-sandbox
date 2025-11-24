# Modal Deployment

Config-driven distributed training for FSDP, ZeRO, Pipeline Parallelism, and FP8.

## Usage

```bash
# ZeRO
modal run zero/modal_app.py --script zero2.py --run-name zero2


# FSDP
modal run fsdp/modal_app.py --run-name fsdp

modal volume get fsdp-traces / ./fsdp/traces/
# modal volume get fsdp-traces <run-id> ./fsdp/traces/<run-id>
tensorboard --logdir ./traces/fsdp/  # Compares all runs
# View large traces with ui.perfetto.dev


# Pipeline Parallelism
modal run pp/modal_app.py


# FP8 (requires H100)
modal run fp8/modal_app.py::sweep


```

## Strategies

| Directory | Strategy | GPUs |
|-----------|----------|------|
| `fsdp/` | Fully Sharded Data Parallel | A100-80GB:2 |
| `zero/` | ZeRO-1/2/3 | A10G:2 |
| `pp/` | Pipeline Parallelism | A100-80GB:2 |
| `fp8/` | FP8 Training | H100:2 |

## Structure

Each directory has:
- `modal_app.py` - config + launcher in one file
- `requirements.txt` - dependencies

Shared infrastructure in `modal_utils.py`.

## Configuration

All settings are in `modal_app.py` as a Python dict - simple and no extra files!
