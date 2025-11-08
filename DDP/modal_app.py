#!/usr/bin/env python3
"""
Modal cloud deployment for DDP training.

- Launches DDP training on Modal GPUs via torchrun
- Manages profiler trace storage in shared volumes
- Builds container images with PyTorch and dependencies
- Syncs parent directory to remote worker
- Configures GPU resources and environment variables
- Provides local entrypoint for remote job execution
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from modal import App, Image, Volume

APP_NAME = "ddp-trace-runner"

DDP_DIR = Path(__file__).resolve().parent
TRAINING_SCRIPT = "ddp.py"
REQUIREMENTS_FILE = DDP_DIR / "requirements.image.txt"

REMOTE_DDP_PATH = "/root/ddp"
TRACE_MOUNT_PATH = "/root/traces"


def _load_config() -> dict[str, str]:
    keys = ("MODAL_GPU_SPEC", "MODAL_TRACE_VOLUME", "TRACE_OUTPUT_DIR")
    missing = [name for name in keys if name not in os.environ]
    if missing:
        raise RuntimeError(
            f"Missing environment value(s): {', '.join(missing)}. "
            "Run via scripts/profile.sh so it exports the required vars."
        )
    return {name: os.environ[name] for name in keys}


CONFIG = _load_config()
GPU_SPEC = CONFIG["MODAL_GPU_SPEC"]
TRACE_VOLUME_NAME = CONFIG["MODAL_TRACE_VOLUME"]
TRACE_OUTPUT_DIR = CONFIG["TRACE_OUTPUT_DIR"]

PYTORCH_WHL_INDEX = "https://download.pytorch.org/whl/cu121"
PYPI_INDEX = "https://pypi.org/simple"

app = App(APP_NAME)

trace_volume = Volume.from_name(TRACE_VOLUME_NAME, create_if_missing=True)

image = (
    Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements(
        str(REQUIREMENTS_FILE),
        index_url=PYTORCH_WHL_INDEX,
        extra_index_url=PYPI_INDEX,
    )
    .env(CONFIG)
    .add_local_dir(str(DDP_DIR), remote_path=REMOTE_DDP_PATH)
)


def _gpu_count_from_spec(spec: str) -> int:
    if ":" not in spec:
        return 1
    _, count_str = spec.split(":", 1)
    try:
        count = int(count_str)
    except ValueError as exc:
        raise ValueError(
            f"Invalid GPU spec '{spec}'. Expected format 'TYPE:COUNT'."
        ) from exc
    if count < 1:
        raise ValueError("GPU count must be >= 1.")
    return count


GPUS_PER_WORKER = _gpu_count_from_spec(GPU_SPEC)


def _build_run_id(label: str | None) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if not label:
        return timestamp
    suffix = "".join(c for c in label if c.isalnum() or c in ("-", "_"))
    return f"{timestamp}-{suffix}" if suffix else timestamp


@app.function(
    image=image,
    gpu=GPU_SPEC,
    timeout=60 * 30,
    volumes={TRACE_MOUNT_PATH: trace_volume},
)
def run_ddp(num_processes: int | None = None, run_name: str | None = None) -> str:
    resolved_processes = GPUS_PER_WORKER if num_processes is None else num_processes
    if resolved_processes < 1:
        raise ValueError("`num_processes` must be >= 1")

    if resolved_processes != GPUS_PER_WORKER:
        raise ValueError(
            "`num_processes` must equal the GPUs provisioned per worker "
            f"({GPUS_PER_WORKER}). Set `GPU_SPEC` or the flag so they match."
        )

    run_id = _build_run_id(run_name)
    trace_root = Path(TRACE_MOUNT_PATH) / run_id
    trace_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "DDP_TRACE_DIR": str(trace_root),
            "TOKENIZERS_PARALLELISM": env.get("TOKENIZERS_PARALLELISM", "false"),
        }
    )

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={resolved_processes}",
        TRAINING_SCRIPT,
    ]

    subprocess.run(cmd, check=True, cwd=REMOTE_DDP_PATH, env=env)
    print(
        "Profiler trace stored at volume "
        f"'{TRACE_VOLUME_NAME}' in {trace_root.name}"
    )
    return run_id


@app.local_entrypoint()
def launch(run_name: str | None = None) -> None:
    """
    Local helper so `modal run path/to/modal_app.py::launch --run-name myrun` kicks off a job.
    """

    run_id = run_ddp.remote(run_name=run_name)
    local_trace_root = TRACE_OUTPUT_DIR
    print(
        "Training complete.\n"
        f"Run id: {run_id}\n"
        f"Retrieve traces via:\n"
        f"  modal volume get {TRACE_VOLUME_NAME} {run_id} {local_trace_root}/{run_id}\n"
        f"then launch TensorBoard with:\n"
        f"  tensorboard --logdir {local_trace_root}/{run_id}"
    )
