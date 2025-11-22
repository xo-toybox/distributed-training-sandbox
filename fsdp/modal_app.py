#!/usr/bin/env python3
"""
FSDP training on Modal.

Run: modal run fsdp/modal_app.py
With custom steps: modal run fsdp/modal_app.py --num-steps 20

Memory profiler schedule:
  - Steps 0-4: wait (no profiling)
  - Steps 5-9: warmup (profile but don't record)
  - Steps 10-19: active (record detailed profiles)
  - Steps 20+: continue training without profiling overhead

For quick memory profiling, use --num-steps 20
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from modal_utils import create_modal_app, _print_completion_message

config = {
    "app": {"name": "fsdp-training", "script_dir": str(Path(__file__).parent), "training_script": "train_fsdp.py", "requirements_file": "requirements.txt"},
    "remote": {"code_path": "/root/fsdp", "trace_path": "/root/traces"},
    "gpu": {"default_spec": "A100-80GB:2", "timeout": 600},
    "volume": {"name": "fsdp-traces", "local_dir": "./traces"},
    "pytorch": {"whl_index": "https://download.pytorch.org/whl/cu121", "pypi_index": "https://pypi.org/simple"},
    "launcher": {"type": "accelerate", "args": [], "env": {"TOKENIZERS_PARALLELISM": "false"}},
}

app, cfg, train = create_modal_app(config)

@app.local_entrypoint()
def main(run_name: str | None = None, num_steps: int = 20) -> None:
    """
    Launch FSDP training.

    Args:
        run_name: Optional name for this training run
        num_steps: Number of training steps (default: 20 for quick profiling)
    """
    print(f"Launching {cfg.training_script} on Modal with {num_steps} steps")
    run_id = train.remote(run_name=run_name, num_steps=num_steps)
    _print_completion_message(cfg, run_id, cfg.training_script)
