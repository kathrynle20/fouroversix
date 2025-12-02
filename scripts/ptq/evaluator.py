from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import modal

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM


class PTQEvaluatorImpl(ABC):
    """Base class for post-training quantization evaluators."""

    @abstractmethod
    def quantize_model(self, **kwargs: dict[str, Any]) -> AutoModelForCausalLM:
        """Quantize a model."""

    def evaluate_impl(
        self,
        model_name: str,
        *,
        device: str,
        dtype: str,
        task: list[str],
        limit: int | None = None,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a quantized model with lm-eval."""

        from lm_eval import evaluator, models

        if isinstance(model_name, str):
            model = self.quantize_model(
                model_name=model_name,
                device=device,
                dtype=dtype,
                **kwargs,
            )
        else:
            model = model_name

        return evaluator.simple_evaluate(
            model=models.huggingface.HFLM(pretrained=model, device=device),
            tasks=task,
            device=device,
            limit=limit,
        )


class PTQEvaluator(PTQEvaluatorImpl):
    """Base class for post-training quantization evaluators."""

    @modal.method()
    def evaluate(self, *args: list[str], **kwargs: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a quantized model."""
        return self.evaluate_impl(*args, **kwargs)
