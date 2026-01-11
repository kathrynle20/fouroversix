from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn  # noqa: PLR0402

from .frontend import fp4_matmul, quantize_to_fp4
from .utils import AdaptiveBlockScalingRule, DataType, FP4Format

if TYPE_CHECKING:
    from collections.abc import Callable

    from .backend import MatmulBackend


def build_forward(
    *,
    device: str,
    dtype: DataType,
    fp4_format: FP4Format,
    a_scale_rule: AdaptiveBlockScalingRule,
    w_scale_rule: AdaptiveBlockScalingRule,
    w_scale_2d: bool,
    matmul_backend: MatmulBackend | None,
    a_quantize_kwargs: dict[str, Any],
    w_quantize_kwargs: dict[str, Any],
    name: str,
    module: str,
    **kwargs: dict[str, Any],  # noqa: ARG001
) -> Callable:
    def forward(
        self,  # noqa: ANN001
        input: tuple[torch.Tensor, ...],  # noqa: A002
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:

        if not hasattr(self, "weight_e2m1"):
            # Store original output features before quantization (padding may change shape)
            self._original_out_features = self.weight.shape[0]
            self.weight_e2m1, self.weight_sf, self.weight_normconst = quantize_to_fp4(
                self.weight,
                scale_rule=w_scale_rule,
                block_scale_2d=w_scale_2d,
                fp4_format=fp4_format,
                **w_quantize_kwargs,
            )
            del self.weight

        out_n = (
            # self.weight_e2m1.shape[0]
            self._original_out_features
            if hasattr(self, "weight_e2m1") and self.weight_e2m1 is not None
            else self.weight.shape[0]
        )

        out = torch.empty(
            *input.shape[:-1],
            out_n,
            device=input.device,
            dtype=dtype.torch(),
        )

        # Slow bmm
        for i in range(input.shape[0]):
            out[i] = fp4_matmul(
                input[i],
                backend=matmul_backend,
                b_e2m1=self.weight_e2m1,
                b_sf=self.weight_sf,
                b_normconst=self.weight_normconst,
                fp4_format=fp4_format,
                out_dtype=dtype,
                out_shape=(input.shape[1], out_n),
                a_quantize_kwargs={
                    "scale_rule": a_scale_rule,
                    "fp4_format": fp4_format,
                    **a_quantize_kwargs,
                },
                b_quantize_kwargs={
                    "fp4_format": fp4_format,
                },
            )

        if hasattr(self, "bias") and self.bias is not None:
            if out.shape[-1] > self.bias.shape[0]:
                self.bias = nn.functional.pad(self.bias, (0, self.bias.shape[0] - out.shape[-1]))
            if out.shape[-1] < self.bias.shape[0]:
                out = nn.functional.pad(out, (0, out.shape[-1] - self.bias.shape[0]))
            out = out + self.bias

        return out

    return forward

def requantize_experts(
    model: nn.Module,
    *,
    layer_pattern: str = "mlp.experts",
    weight_names: list[str] | None = None,
    scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    fp4_format: FP4Format = FP4Format.nvfp4,
) -> None:
    """Re-quantize dequantized MXFP4 weights to NVFP4 (fake quantize)."""
    from fouroversix.quantize.reference import (
        quantize_to_fp4 as quantize_to_fp4_ref,
    )

    if weight_names is None:
        weight_names = ["gate_up_proj", "down_proj"]
    
    for name, module in model.named_modules():
        if layer_pattern not in name:
            continue
        
        print(f"Re-quantizing to {fp4_format}: {name}")
        
        for weight_name in weight_names:
            if hasattr(module, weight_name):
                weight = getattr(module, weight_name)
                if isinstance(weight, torch.Tensor) and weight.dtype in (torch.bfloat16, torch.float16, torch.float32):
                    # quantized = fake_quantize_to_fp4(
                    #     weight.data,
                    #     fp4_format=fp4_format,  # Re-quantize to NVFP4
                    #     scale_rule=scale_rule,
                    # )
                    out_e2m1, out_sf, out_normconst, *out_extras = quantize_to_fp4_ref(
                        weight,
                        fp4_format=fp4_format,
                        scale_rule=scale_rule,
                    )
                    # Replace original weight with quantized version
                    if isinstance(weight, nn.Parameter):
                        setattr(module, weight_name, nn.Parameter(quantized, requires_grad=weight.requires_grad))
                    else:
                        setattr(module, weight_name, quantized)
                    del weight  # Delete reference to original weight
                    print(f"  - {weight_name}: {quantized.shape} -> NVFP4 fake-quantized")

def apply_ptq(
    model: nn.Module,
    *,
    exclude_layers: list[str] | None = None,
    allow_layers: list[str] | None = None, # exclude layers override allow_layers
    device: str = "cuda",
    dtype: DataType = DataType.bfloat16,
    fp4_format: FP4Format = FP4Format.nvfp4,
    a_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    w_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    w_scale_2d: bool = False,
    matmul_backend: MatmulBackend | None = None,
    a_quantize_kwargs: dict[str, Any] | None = None,
    w_quantize_kwargs: dict[str, Any] | None = None,
    build_forward_fn: Callable | None = None,
    **kwargs: dict[str, Any],
) -> None:
    if exclude_layers is None:
        exclude_layers = ["lm_head"]

    if allow_layers is None:
        allow_layers = []

    if a_quantize_kwargs is None:
        a_quantize_kwargs = {}

    if w_quantize_kwargs is None:
        w_quantize_kwargs = {}

    if build_forward_fn is None:
        build_forward_fn = build_forward

    requantize_experts(model, fp4_format=fp4_format)

    for name, module in model.named_modules():
        exclude = False
        if not isinstance(module, nn.Linear):
            exclude = True
        for allow_name in allow_layers:
            if allow_name in name:
                exclude = False
        for exclude_name in exclude_layers:
            if exclude_name in name:
                exclude = True
        
        if exclude:
            continue

        print(f"Layer: {name} module {module} Quantizing: {not exclude}")

        module.forward = types.MethodType(
            build_forward_fn(
                device=device,
                dtype=dtype,
                fp4_format=fp4_format,
                a_scale_rule=a_scale_rule,
                w_scale_rule=w_scale_rule,
                w_scale_2d=w_scale_2d,
                matmul_backend=matmul_backend,
                a_quantize_kwargs=a_quantize_kwargs,
                w_quantize_kwargs=w_quantize_kwargs,
                name=name,
                module=module,
                **kwargs,
            ),
            module,
        )
