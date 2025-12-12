from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
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
        tasks: list[str],
        limit: int | None = None,
        trust_remote_code: bool = False,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a quantized model with lm-eval."""

        from lm_eval import evaluator, models
        from lm_eval.tasks import TaskManager

        if isinstance(model_name, str):
            model = self.quantize_model(
                model_name=model_name,
                device=device,
                dtype=dtype,
                model_kwargs={"trust_remote_code": trust_remote_code},
                **kwargs,
            )
        else:
            model = model_name

        return evaluator.simple_evaluate(
            model=models.huggingface.HFLM(pretrained=model, device=device),
            tasks=tasks,
            device=device,
            limit=limit,
            task_manager=TaskManager(
                include_path=(Path(__file__).parent / "tasks").as_posix(),
            ),
        )


class PTQEvaluator(PTQEvaluatorImpl):
    """Base class for post-training quantization evaluators."""

    @modal.method()
    def evaluate(self, *args: list[str], **kwargs: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a quantized model."""
        return self.evaluate_impl(*args, **kwargs)
