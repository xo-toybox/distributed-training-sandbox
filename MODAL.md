# Modal Deployment

Config-driven distributed training for DDP, FSDP, ZeRO.

## Usage

```bash
# Run training
modal run zero/modal_app.py --script zero2.py --run-name my-exp
modal run fsdp/modal_app.py --run-name my-exp

# Download profiler traces
# Single run (get run-id from output above)
modal volume get zero-traces <run-id> ./traces/zero/<run-id>
# All runs
modal volume get zero-traces / ./traces/zero/

# View with TensorBoard
tensorboard --logdir ./traces/zero/  # Compares all runs
```

## Structure

Each strategy has:
- `modal_app.py` - config + launcher in one file
- `requirements.txt` - dependencies

Shared infrastructure in `modal_utils.py`.

## Config Example

`zero/modal_app.py` (config embedded as Python dict):
```python
config = {
    "app": {"name": "zero-training", "training_script": "zero1.py"},
    "gpu": {"default_spec": "A10G:2"},
    "launcher": {"type": "torchrun"},
    "entrypoint": {"script_options": ["zero1.py", "zero2.py", "zero3.py"]},
}
app, _, _ = create_modal_app(config)
```

## Add New Strategy

```bash
# Copy template
cp zero/modal_app.py my_strategy/modal_app.py
# Edit config dict in my_strategy/modal_app.py
modal run my_strategy/modal_app.py
```

## Configuration

All settings are in `modal_app.py` as a Python dict - simple and no extra files!
