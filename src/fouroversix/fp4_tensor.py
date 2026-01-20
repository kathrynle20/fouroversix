import torch

from .utils import AdaptiveBlockScalingRule, FP4Format


class FP4Tensor:
    e2m1_values: torch.Tensor
    scale_factors: torch.Tensor
    amax: torch.Tensor

    fp4_format: FP4Format
    original_shape: tuple[int, int]
    scale_rule: AdaptiveBlockScalingRule

    def __init__(
        self,
        e2m1_values: torch.Tensor,
        scale_factors: torch.Tensor,
        amax: torch.Tensor,
        fp4_format: FP4Format,
        original_shape: tuple[int, int],
        scale_rule: AdaptiveBlockScalingRule,
    ) -> None:
        self.e2m1_values = e2m1_values
        self.scale_factors = scale_factors
        self.amax = amax
        self.fp4_format = fp4_format
        self.original_shape = original_shape
        self.scale_rule = scale_rule
