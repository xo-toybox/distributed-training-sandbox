#!/usr/bin/env python3
"""
Reusable Modal deployment utilities for distributed training.

This module provides a config-driven approach to creating Modal apps
for different training strategies (DDP, FSDP, ZeRO, etc.).
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from modal import App, Image, Volume


class ModalConfig:
    """Configuration for Modal app deployment."""

    def __init__(self, config_path):
        """Load configuration from YAML file or dict."""
        if isinstance(config_path, dict):
            self.config = config_path
        else:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)

        self.app_name = self.config["app"]["name"]
        self.script_dir = Path(self.config["app"]["script_dir"]).resolve()
        self.training_script = self.config["app"]["training_script"]
        self.requirements_file = self.script_dir / self.config["app"].get(
            "requirements_file", "requirements.txt"
        )

        # Remote paths
        self.remote_code_path = self.config["remote"]["code_path"]
        self.remote_trace_path = self.config["remote"].get("trace_path", "/root/traces")

        # GPU configuration (from config only)
        self.gpu_spec = self.config["gpu"].get("default_spec", "A10G")
        self.timeout = self.config["gpu"].get("timeout", 60 * 30)

        # Volume configuration (from config, not env vars)
        volume_config = self.config.get("volume", {})
        self.trace_volume_name = volume_config.get("name", f"{self.app_name}-traces")
        self.trace_output_dir = volume_config.get("local_dir", "./traces")

        # PyTorch configuration
        pytorch_config = self.config.get("pytorch", {})
        self.pytorch_whl_index = pytorch_config.get("whl_index", "https://download.pytorch.org/whl/cu121")
        self.pypi_index = pytorch_config.get("pypi_index", "https://pypi.org/simple")

        # Launcher configuration (torchrun, accelerate, etc.)
        self.launcher = self.config.get("launcher", {})

    def get_gpu_count(self) -> int:
        """Parse GPU count from spec (e.g., 'A10G:2' -> 2)."""
        spec = self.gpu_spec
        if ":" not in spec:
            return 1
        _, count_str = spec.split(":", 1)
        try:
            count = int(count_str)
        except ValueError as exc:
            raise ValueError(f"Invalid GPU spec '{spec}'. Expected format 'TYPE:COUNT'.") from exc
        if count < 1:
            raise ValueError("GPU count must be >= 1.")
        return count


def build_image(config: ModalConfig, extra_packages: list[str] | None = None) -> Image:
    """Build Modal container image with dependencies."""
    image = Image.debian_slim().apt_install("git")

    # Install Python dependencies
    if config.requirements_file.exists():
        image = image.pip_install_from_requirements(
            str(config.requirements_file),
            index_url=config.pytorch_whl_index,
            extra_index_url=config.pypi_index,
        )

    # Install extra packages if provided
    if extra_packages:
        image = image.pip_install(*extra_packages)
    image = image.add_local_file("modal_utils.py", remote_path="/root/modal_utils.py")

    # Sync local directory to remote
    image = image.add_local_dir(str(config.script_dir), remote_path=config.remote_code_path)

    return image


def build_run_id(label: str | None = None) -> str:
    """Generate a unique run ID with timestamp."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if not label:
        return timestamp
    suffix = "".join(c for c in label if c.isalnum() or c in ("-", "_"))
    return f"{timestamp}-{suffix}" if suffix else timestamp


def build_launch_command(
    config: ModalConfig,
    num_processes: int,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the command to launch training."""
    launcher_type = config.launcher.get("type", "torchrun")

    if launcher_type == "torchrun":
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={num_processes}",
        ]
        # Add additional torchrun args if specified
        torchrun_args = config.launcher.get("args", [])
        cmd.extend(torchrun_args)

    elif launcher_type == "accelerate":
        cmd = [
            "accelerate",
            "launch",
            "--num_processes",
            str(num_processes),
        ]
        # Add additional accelerate args if specified
        accelerate_args = config.launcher.get("args", [])
        cmd.extend(accelerate_args)

    elif launcher_type == "python":
        cmd = ["python"]

    else:
        raise ValueError(f"Unsupported launcher type: {launcher_type}")

    cmd.append(config.training_script)

    # Add extra arguments to the training script
    if extra_args:
        cmd.extend(extra_args)

    return cmd


def run_training(
    config: ModalConfig,
    num_processes: int | None = None,
    run_name: str | None = None,
    extra_args: list[str] | None = None,
) -> str:
    """Execute training job on Modal."""
    gpu_count = config.get_gpu_count()
    resolved_processes = gpu_count if num_processes is None else num_processes

    if resolved_processes < 1:
        raise ValueError("`num_processes` must be >= 1")

    if resolved_processes != gpu_count:
        raise ValueError(
            f"`num_processes` must equal the GPUs provisioned per worker "
            f"({gpu_count}). Set `MODAL_GPU_SPEC` or the flag so they match."
        )

    # Create run ID and trace directory
    run_id = build_run_id(run_name)
    trace_root = Path(config.remote_trace_path) / run_id
    trace_root.mkdir(parents=True, exist_ok=True)

    # Setup environment
    env = os.environ.copy()
    env.update(
        {
            "TRACE_DIR": str(trace_root),
            "TOKENIZERS_PARALLELISM": env.get("TOKENIZERS_PARALLELISM", "false"),
        }
    )

    # Add any additional env vars from config
    additional_env = config.launcher.get("env", {})
    env.update(additional_env)

    # Build launch command
    cmd = build_launch_command(config, resolved_processes, extra_args)

    # Execute training
    print(f"Launching training with command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=config.remote_code_path, env=env)

    print(f"Training complete. Traces stored at volume '{config.trace_volume_name}' in {run_id}")
    return run_id


def create_modal_app(config_path):
    """
    Create a Modal app from configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Tuple of (Modal App instance, ModalConfig instance, train function)
    """
    config = ModalConfig(Path(config_path) if not isinstance(config_path, dict) else config_path)
    app = App(config.app_name)

    # Create volume for traces
    trace_volume = Volume.from_name(config.trace_volume_name, create_if_missing=True)

    # Build image
    image = build_image(config)

    # Create the training function with serialized=True to allow non-global scope
    @app.function(
        image=image,
        gpu=config.gpu_spec,
        timeout=config.timeout,
        volumes={config.remote_trace_path: trace_volume},
        serialized=True,
    )
    def train(
        num_processes: int | None = None,
        run_name: str | None = None,
        script: str | None = None,
    ) -> str:
        # If script is provided, update config
        if script:
            config.training_script = script
        return run_training(config, num_processes, run_name)

    # Note: local_entrypoint is defined at global scope (bottom of file)
    # to satisfy Modal's requirement for global scope functions

    return app, config, train


def _print_completion_message(config: ModalConfig, run_id: str, script: str) -> None:
    """Print completion message with trace retrieval instructions."""
    local_trace_root = config.trace_output_dir
    print(
        f"\nTraining complete!\n"
        f"Run ID: {run_id}\n"
        f"Script: {script}\n"
        f"\nRetrieve traces via:\n"
        f"  modal volume get {config.trace_volume_name} {run_id} {local_trace_root}/{run_id}\n"
        f"\nThen launch TensorBoard with:\n"
        f"  tensorboard --logdir {local_trace_root}/{run_id}"
    )


def main() -> None:
    """
    Main entrypoint for running Modal apps from config files.

    Usage:
        modal run modal_utils.py --config path/to/modal_config.yaml
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run Modal app from configuration file")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to modal_config.yaml file",
    )
    args = parser.parse_args()

    # Create app from config
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading Modal app from config: {config_path}")
    app, config, _ = create_modal_app(config_path)
    print(f"Modal app '{config.app_name}' loaded successfully")
