#!/usr/bin/env python3
"""
Pipeline Parallelism training on Modal - Compare GPipe vs 1F1B.

Run comparison (default):
  modal run pp/modal_app.py

Run with more epochs:
  modal run pp/modal_app.py --num-epochs 10

Run single strategy:
  modal run pp/modal_app.py --compare=false
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from modal_utils import create_modal_app

config = {
    "app": {"name": "pp-training", "script_dir": str(Path(__file__).parent), "training_script": "1f1b.py", "requirements_file": "requirements.txt"},
    "remote": {"code_path": "/root/pp"},
    "gpu": {"default_spec": "T4:2", "timeout": 600},
    "pytorch": {"whl_index": "https://download.pytorch.org/whl/cu121", "pypi_index": "https://pypi.org/simple"},
    "launcher": {"type": "python", "args": [], "env": {"TOKENIZERS_PARALLELISM": "false"}},
}

app, cfg, train = create_modal_app(config)

@app.local_entrypoint()
def main(run_name: str | None = None, num_epochs: int = 5, compare: bool = True) -> None:
    """
    Launch Pipeline Parallelism training and compare GPipe vs 1F1B.

    Args:
        run_name: Optional name for this training run
        num_epochs: Number of training epochs (default: 5 for quick comparison)
        compare: Run both GPipe and 1F1B for comparison (default: True)
    """
    if compare:
        print(f"\n{'='*70}")
        print("Pipeline Parallelism Comparison: GPipe vs 1F1B")
        print(f"{'='*70}")
        print(f"Running {num_epochs} epochs on each strategy...\n")

        # Run both strategies
        print("Running GPipe...")
        train.remote(run_name=run_name, script="gpipe.py", num_epochs=num_epochs)

        print("\nRunning 1F1B...")
        train.remote(run_name=run_name, script="1f1b.py", num_epochs=num_epochs)
   else:
        # Single script mode
        print(f"Launching {cfg.training_script} on Modal with {num_epochs} epochs")
        train.remote(run_name=run_name, num_epochs=num_epochs)
