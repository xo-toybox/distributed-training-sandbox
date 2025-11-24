#!/usr/bin/env python3
"""
FP8 training on Modal.

NOTE: FP8 requires H100 GPUs (or newer) for hardware support.
      A100 GPUs do not support the fp8e4nv dtype used by torchao.
      Use --precision bf16 if you only have access to A100 GPUs.

Single run:
  modal run fp8/modal_app.py --model-name Qwen/Qwen3-4B

Parameter sweep:
  modal run fp8/modal_app.py::sweep --model-name Qwen/Qwen3-4B

Custom parameters:
  modal run fp8/modal_app.py --model-name Qwen/Qwen3-4B --num-steps 50 --sequence-length 4096 --precision fp8

BF16 on A100 (if H100 not available):
  modal run fp8/modal_app.py --model-name Qwen/Qwen3-4B --precision bf16
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from modal_utils import create_modal_app, _print_completion_message

config = {
    "app": {"name": "fp8-training", "script_dir": str(Path(__file__).parent), "training_script": "fp8_benchmark.py", "requirements_file": "requirements.txt"},
    "remote": {"code_path": "/root/fp8", "trace_path": "/root/traces"},
    "gpu": {"default_spec": "H100:2", "timeout": 600},
    "volume": {"name": "fp8-traces", "local_dir": "./traces"},
    "pytorch": {"whl_index": "https://download.pytorch.org/whl/cu121", "pypi_index": "https://pypi.org/simple"},
    "launcher": {"type": "accelerate", "args": [], "env": {"TOKENIZERS_PARALLELISM": "false"}},
}

app, cfg, train = create_modal_app(config)

@app.local_entrypoint()
def main(
    model_name: str = "Qwen/Qwen3-4B",
    run_name: str | None = None,
    num_steps: int = 50,
    sequence_length: int = 8192,
    precision: str = "fp8"
) -> None:
    """
    Launch FP8 training with specified parameters.

    Args:
        model_name: Model name/path (default: Qwen/Qwen3-4B)
        run_name: Optional name for this training run
        num_steps: Number of training steps (default: 50)
        sequence_length: Sequence length (default: 8192)
        precision: Precision mode - fp8 or bf16 (default: fp8)
    """
    extra_args = [
        model_name,
        "--sequence-length", str(sequence_length),
        "--precision", precision,
    ]

    print(f"Launching {cfg.training_script} on Modal")
    print(f"  Model: {model_name}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Precision: {precision}")
    print(f"  Steps: {num_steps}")

    run_id = train.remote(run_name=run_name, num_steps=num_steps, extra_args=extra_args)
    _print_completion_message(cfg, run_id, cfg.training_script)


@app.local_entrypoint()
def sweep(
    model_name: str = "Qwen/Qwen3-4B",
    run_name: str | None = None,
    num_steps: int = 50
) -> None:
    """
    Launch parameter sweep across sequence lengths and precisions.

    Sweeps over:
    - Sequence lengths: [2048, 4096, 8192]
    - Precisions: [bf16, fp8]

    Args:
        model_name: Model name/path (default: Qwen/Qwen3-4B)
        run_name: Optional base name for sweep runs
        num_steps: Number of training steps per config (default: 50)
    """
    seq_lengths = [2048, 4096, 8192]
    precisions = ["bf16", "fp8"]

    print(f"Starting parameter sweep for {model_name}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Precisions: {precisions}")
    print(f"Steps per run: {num_steps}")
    print(f"Total runs: {len(seq_lengths) * len(precisions)}")
    print()

    for seq_len in seq_lengths:
        for prec in precisions:
            sweep_run_name = f"{run_name or 'sweep'}-{prec}-seq{seq_len}"
            extra_args = [
                model_name,
                "--sequence-length", str(seq_len),
                "--precision", prec,
            ]

            print(f"=== Launching: seq_len={seq_len}, precision={prec} ===")
            run_id = train.remote(run_name=sweep_run_name, num_steps=num_steps, extra_args=extra_args)
            print(f"Run ID: {run_id}")
            print()

    print("All sweep runs launched!")
    print(f"View traces: modal volume ls {cfg.trace_volume_name}")
    print(f"Download traces: modal volume get {cfg.trace_volume_name} <remote_path> <local_path>")
