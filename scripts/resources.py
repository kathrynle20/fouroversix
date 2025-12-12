from __future__ import annotations

import subprocess
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from collections.abc import Callable

FOUROVERSIX_CACHE_PATH = Path("/fouroversix")
FOUROVERSIX_INSTALL_PATH = Path("/root/fouroversix")

app = modal.App("fouroversix")
cache_volume = modal.Volume.from_name("fouroversix", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")


class Dependency(str, Enum):
    """Dependencies to add to the base image."""

    fast_hadamard_transform = "fast_hadamard_transform"
    flash_attention = "flash_attention"
    fouroversix = "fouroversix"


cuda_version_to_image_tag = {
    "12.8": "nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-devel-ubuntu24.04",
    "12.9": "nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04",
    "13.0": "nvcr.io/nvidia/cuda-dl-base:25.09-cuda13.0-devel-ubuntu24.04",
}


def install_flash_attn() -> None:
    subprocess.run(
        ["pip", "install", "flash-attn", "--no-build-isolation"],  # noqa: S607
        check=False,
    )


def install_fouroversix() -> None:
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            "-e",
            FOUROVERSIX_INSTALL_PATH.as_posix(),
        ],
        check=False,
    )


def get_image(  # noqa: C901
    dependencies: list[Dependency] | None = None,
    *,
    cuda_version: str = "12.8",
    deploy: bool = False,
    disable_kernels: bool = False,
    extra_env: dict[str, str] | None = None,
    extra_pip_dependencies: list[str] | None = None,
    python_version: str = "3.12",
    pytorch_version: str = "2.9.1",
    run_before_copy: Callable[[modal.Image], modal.Image] | None = None,
) -> modal.Image:
    if dependencies is None:
        dependencies = [Dependency.fouroversix]

    img = (
        modal.Image.from_registry(
            cuda_version_to_image_tag[cuda_version],
            add_python=python_version,
        )
        .entrypoint([])
        .apt_install("clang", "git")
        .pip_install_from_pyproject(
            "pyproject.toml",
            optional_dependencies=["fast-build", "test"],
        )
        .uv_pip_install("numpy", "wheel")
        .uv_pip_install(
            f"torch=={pytorch_version}",
            extra_index_url=(
                f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
            ),
        )
    )

    for dependency in dependencies:
        if dependency == Dependency.fast_hadamard_transform:
            img = img.run_commands(
                "git clone https://github.com/Dao-AILab/fast-hadamard-transform.git "
                f"{FOUROVERSIX_INSTALL_PATH}/fast-hadamard-transform",
                f"pip install {FOUROVERSIX_INSTALL_PATH}/fast-hadamard-transform",
            )

        if dependency == Dependency.flash_attention:
            img = img.run_function(
                install_flash_attn,
                cpu=64,
                memory=128 * 1024,
                gpu="B200",
            )

        if dependency == Dependency.fouroversix:
            img = (
                img.run_commands(
                    "git clone https://github.com/NVIDIA/cutlass.git "
                    f"{FOUROVERSIX_INSTALL_PATH}/third_party/cutlass",
                )
                .add_local_file(
                    "pyproject.toml",
                    f"{FOUROVERSIX_INSTALL_PATH}/pyproject.toml",
                    copy=True,
                )
                .add_local_file(
                    "setup.py",
                    f"{FOUROVERSIX_INSTALL_PATH}/setup.py",
                    copy=True,
                )
                .add_local_file(
                    "src/fouroversix/__init__.py",
                    f"{FOUROVERSIX_INSTALL_PATH}/src/fouroversix/__init__.py",
                    copy=True,
                )
                .add_local_dir(
                    "src/fouroversix/csrc",
                    f"{FOUROVERSIX_INSTALL_PATH}/src/fouroversix/csrc",
                    copy=True,
                )
            )

            if disable_kernels:
                img = img.env({"DISABLE_KERNEL_COMPILATION": "1"})

            img = img.run_function(install_fouroversix, cpu=8, memory=32 * 1024)

    if extra_pip_dependencies is not None:
        img = img.uv_pip_install(*extra_pip_dependencies)

    img = img.env({"HF_HOME": FOUROVERSIX_CACHE_PATH.as_posix(), **(extra_env or {})})

    if run_before_copy is not None:
        img = run_before_copy(img)

    # Add source files after all dependencies are added so we can avoid rebuilding when
    # they change
    for dependency in dependencies:
        if dependency == Dependency.fouroversix:
            img = img.add_local_dir(
                "src",
                f"{FOUROVERSIX_INSTALL_PATH}/src",
                copy=deploy,
            )

    return img
