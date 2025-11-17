#!/usr/bin/env python3
"""ZeRO training on Modal. Run: modal run zero/modal.py --script zero2.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from modal_utils import create_modal_app, _print_completion_message

config = {
    "app": {"name": "zero-training", "script_dir": str(Path(__file__).parent), "training_script": "zero1.py", "requirements_file": "requirements.txt"},
    "remote": {"code_path": "/root/zero", "trace_path": "/root/traces"},
    "gpu": {"default_spec": "A10G:2", "timeout": 1800},
    "volume": {"name": "zero-traces", "local_dir": "./traces"},
    "pytorch": {"whl_index": "https://download.pytorch.org/whl/cu121", "pypi_index": "https://pypi.org/simple"},
    "launcher": {"type": "torchrun", "args": [], "env": {"TOKENIZERS_PARALLELISM": "false"}},
    "entrypoint": {"script_options": ["zero1.py", "zero2.py", "zero3.py"], "default_script": "zero1.py"},
}

app, cfg, train = create_modal_app(config)

@app.local_entrypoint()
def main(run_name: str | None = None, script: str | None = None) -> None:
    """Launch ZeRO training."""
    script_options = config["entrypoint"]["script_options"]
    selected = script or config["entrypoint"]["default_script"]
    if selected not in script_options:
        print(f"Error: script must be one of {script_options}")
        sys.exit(1)
    print(f"Launching {selected} on Modal")
    run_id = train.remote(run_name=run_name, script=selected)
    _print_completion_message(cfg, run_id, selected)
