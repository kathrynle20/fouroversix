from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import modal

from ..resources import FOUROVERSIX_CACHE_PATH, app, cache_volume, hf_secret
from .rtn import RTNEvaluatorImpl, rtn_img

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fouroversix.utils import AdaptiveBlockScalingRule, DataType, FP4Format

with rtn_img.imports():
    import torch
    from fouroversix import fp4_matmul
    from fouroversix.ptq import apply_ptq
    from transformers import AutoModelForCausalLM


def build_forward(
    *,
    device: str,
    dtype: DataType,
    fp4_format: FP4Format,
    a_scale_rule: AdaptiveBlockScalingRule,
    w_scale_rule: AdaptiveBlockScalingRule,
    a_quantize_kwargs: dict[str, Any],
    w_quantize_kwargs: dict[str, Any],
    smoothquant_alpha: float,
    **kwargs: dict[str, Any],  # noqa: ARG001
) -> Callable:
    def forward(
        self,  # noqa: ANN001
        input: tuple[torch.Tensor, ...],  # noqa: A002
    ) -> torch.Tensor:
        out = torch.empty(
            *input.shape[:-1],
            self.weight.shape[0],
            device=device,
            dtype=dtype.torch(),
        )

        # Slow bmm
        for i in range(input.shape[0]):
            s = (input[i].abs().max(dim=0).values ** smoothquant_alpha) / (
                self.weight.abs().max(dim=0).values ** (1 - smoothquant_alpha)
            )

            out[i] = fp4_matmul(
                input[i] / s[None, :],
                self.weight * s[None, :],
                fp4_format=fp4_format,
                out_dtype=dtype,
                out_shape=(input.shape[1], self.weight.shape[0]),
                a_quantize_kwargs={
                    "scale_rule": a_scale_rule,
                    **a_quantize_kwargs,
                },
                b_quantize_kwargs={
                    "scale_rule": w_scale_rule,
                    **w_quantize_kwargs,
                },
            )

        if hasattr(self, "bias") and self.bias is not None:
            out = out + self.bias

        return out

    return forward


@app.cls(
    image=rtn_img,
    gpu="B200",
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class SmoothQuantEvaluator(RTNEvaluatorImpl):
    """Evaluate a model using SmoothQuant."""

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        dtype: DataType,
        model_kwargs: dict[str, Any] | None = None,
        smoothquant_alpha: float,
        **kwargs: dict[str, Any],
    ) -> AutoModelForCausalLM:
        """Quantize a model using SmoothQuant."""

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            dtype=dtype.torch(),
            **(model_kwargs or {}),
        )
        apply_ptq(
            model,
            device=device,
            dtype=dtype,
            build_forward_fn=build_forward,
            smoothquant_alpha=smoothquant_alpha,
            **kwargs,
        )
        return model

    @modal.method()
    def smoothquant_evaluate(
        self,
        model_name: str,
        smoothquant_alpha: float,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a model using SmoothQuant."""

        return super().evaluate_impl(
            model_name,
            smoothquant_alpha=smoothquant_alpha,
            **kwargs,
        )


@app.cls(
    image=rtn_img,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
    timeout=24 * 60 * 60,
    nonpreemptible=True,
)
class SmoothQuantAutoAlphaEvaluator:
    """Evaluate a model using SmoothQuant."""

    @modal.method()
    def evaluate(
        self,
        model_name: str,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Quantize a model using SmoothQuant."""

        a_scale_rule = kwargs.get("a_scale_rule")
        w_scale_rule = kwargs.get("w_scale_rule")

        smoothquant_alpha = get_smoothquant_alpha(
            model_name,
            a_scale_rule,
            w_scale_rule,
        )

        if smoothquant_alpha is None:
            alpha_candidates = [x / 10 for x in range(11)]

            best_ppl = None

            for i, result in enumerate(
                SmoothQuantEvaluator().smoothquant_evaluate.starmap(
                    [(model_name, alpha) for alpha in alpha_candidates],
                    kwargs={
                        **kwargs,
                        "tasks": ["wikitext_train"],
                    },
                ),
            ):
                ppl = result["results"]["wikitext_train"]["word_perplexity,none"]

                if smoothquant_alpha is None or ppl < best_ppl:
                    smoothquant_alpha = alpha_candidates[i]
                    best_ppl = ppl

                print(f"alpha={alpha_candidates[i]}, ppl={ppl}")

            save_path = get_smoothquant_save_path(
                model_name,
                a_scale_rule,
                w_scale_rule,
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w") as f:
                f.write(str(smoothquant_alpha))

        return SmoothQuantEvaluator().smoothquant_evaluate.remote(
            model_name,
            smoothquant_alpha=smoothquant_alpha,
            **kwargs,
        )


def get_smoothquant_save_path(
    model_name: str,
    a_scale_rule: AdaptiveBlockScalingRule,
    w_scale_rule: AdaptiveBlockScalingRule,
) -> Path:
    return (
        FOUROVERSIX_CACHE_PATH
        / "ptq"
        / "smoothquant"
        / f"{model_name}-{a_scale_rule.value}-{w_scale_rule.value}"
    )


def get_smoothquant_alpha(
    model_name: str,
    a_scale_rule: AdaptiveBlockScalingRule,
    w_scale_rule: AdaptiveBlockScalingRule,
) -> float | None:
    save_path = get_smoothquant_save_path(model_name, a_scale_rule, w_scale_rule)
    smoothquant_alpha = None

    if save_path.exists():
        with save_path.open("r") as f, contextlib.suppress(ValueError):
            smoothquant_alpha = float(f.read())

    return smoothquant_alpha
