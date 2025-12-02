from enum import Enum

import torch


class BlockScaleSelectionRule(str, Enum):
    """
    Block scale selection rules for NVFP4 quantization.

    - `abs_max`: Between 4 and 6, select the block scale that minimizes the maximum
        absolute quantization error.
    - `always_4`: Select 4 for all blocks.
    - `always_6`: Select 6 for all blocks (normal NVFP4 quantization).
    - `l1_norm`: Between 4 and 6, select the block scale that minimizes the L1 norm of
        the quantization error.
    - `mse`: Between 4 and 6, select the block scale that minimizes the mean squared
        quantization error.
    """

    abs_max = "abs_max"
    always_4 = "always_4"
    always_6 = "always_6"
    l1_norm = "l1_norm"
    mse = "mse"

    def cuda_id(self) -> int:
        """ID for the rule in the CUDA implementation."""

        return {
            BlockScaleSelectionRule.abs_max: 4,
            BlockScaleSelectionRule.always_4: 1,
            BlockScaleSelectionRule.always_6: 0,
            BlockScaleSelectionRule.l1_norm: 2,
            BlockScaleSelectionRule.mse: 3,
        }[self]


class DataType(str, Enum):
    """High-precision data types."""

    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"

    def torch(self) -> torch.dtype:
        """Return the corresponding torch.dtype."""

        if self == DataType.bfloat16:
            return torch.bfloat16
        if self == DataType.float16:
            return torch.float16
        if self == DataType.float32:
            return torch.float32
        msg = f"Invalid data type: {self}"
        raise ValueError(msg)


class FP4Format(str, Enum):
    """FP4 formats."""

    mxfp4 = "mxfp4"
    nvfp4 = "nvfp4"


class RoundStyle(str, Enum):
    """
    Rounding styles for quantization.

    - `nearest`: Round to the nearest FP4 value.
    - `stochastic`: Round to the nearest FP4 value after applying random noise to each
        value.
    """

    nearest = "nearest"
    stochastic = "stochastic"
