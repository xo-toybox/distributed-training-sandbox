#!/usr/bin/env python3
"""FSDP training on Modal. Run: modal run fsdp/modal.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from modal_utils import create_modal_app, _print_completion_message

config = {
    "app": {"name": "fsdp-training", "script_dir": str(Path(__file__).parent), "training_script": "train_fsdp.py", "requirements_file": "requirements.txt"},
    "remote": {"code_path": "/root/fsdp", "trace_path": "/root/traces"},
    "gpu": {"default_spec": "A10G:2", "timeout": 1800},
    "volume": {"name": "fsdp-traces", "local_dir": "./traces"},
    "pytorch": {"whl_index": "https://download.pytorch.org/whl/cu121", "pypi_index": "https://pypi.org/simple"},
    "launcher": {"type": "accelerate", "args": [], "env": {"TOKENIZERS_PARALLELISM": "false"}},
}

app, cfg, train = create_modal_app(config)

@app.local_entrypoint()
def main(run_name: str | None = None) -> None:
    """Launch FSDP training."""
    print(f"Launching {cfg.training_script} on Modal")
    run_id = train.remote(run_name=run_name)
    _print_completion_message(cfg, run_id, cfg.training_script)
