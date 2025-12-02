import os
import warnings
from pathlib import Path
from typing import Any

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


class NinjaBuildExtension(BuildExtension):
    """
    Custom build extension that tells Ninja how many jobs to run.

    Credit: https://github.com/Dao-AILab/flash-attention/blob/main/setup.py
    """

    def __init__(self, *args: list[Any], **kwargs: dict[str, Any]) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            try:
                import psutil

                # calculate the maximum allowed NUM_JOBS based on cores
                max_num_jobs_cores = max(1, os.cpu_count() // 2)

                # calculate the maximum allowed NUM_JOBS based on free memory
                free_memory_gb = psutil.virtual_memory().available / (
                    1024**3
                )  # free memory in GB
                max_num_jobs_memory = int(
                    free_memory_gb / 9,
                )  # each JOB peak memory cost is ~8-9GB when threads = 4

                # pick lower value of jobs based on cores vs memory metric to minimize
                # oom and swap usage during compilation
                max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
                os.environ["MAX_JOBS"] = str(max_jobs)
            except ImportError:
                warnings.warn(
                    "psutil not found, install psutil and ninja to get better build "
                    "performance",
                    stacklevel=1,
                )

        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    ext_modules = None

    if os.getenv("DISABLE_KERNEL_COMPILATION", "0") == "1":
        warnings.warn(
            "DISABLE_KERNEL_COMPILATION is set to 1, installing fouroversix without "
            "quantization and matmul kernels",
            stacklevel=1,
        )
    else:
        setup_dir = Path(__file__).parent
        kernels_dir = setup_dir / "src" / "fouroversix" / "csrc"
        sources = [
            path.relative_to(Path(__file__).parent).as_posix()
            for ext in ["**/*.cu", "**/*.cpp"]
            for path in kernels_dir.glob(ext)
        ]
        ext_modules = [
            CUDAExtension(
                "fouroversix._C",
                sources,
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": [
                        "-O3",
                        "-std=c++17",
                        "-gencode",
                        "arch=compute_100a,code=sm_100a",
                        "--expt-relaxed-constexpr",
                        "--use_fast_math",
                        "-DNDEBUG",
                        "-Xcompiler",
                        "-funroll-loops",
                        "-Xcompiler",
                        "-ffast-math",
                        "-Xcompiler",
                        "-finline-functions",
                    ],
                },
                include_dirs=[
                    setup_dir / "third_party/cutlass/examples/common",
                    setup_dir / "third_party/cutlass/include",
                    setup_dir / "third_party/cutlass/tools/util/include",
                    kernels_dir / "include",
                ],
            ),
        ]

    setup(
        name="fouroversix",
        ext_modules=ext_modules,
        cmdclass={"build_ext": NinjaBuildExtension},
    )
