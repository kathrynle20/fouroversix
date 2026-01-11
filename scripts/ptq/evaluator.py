from __future__ import annotations

import json
import torch
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import modal
from dateutil.tz import tzlocal

from ..resources import FOUROVERSIX_CACHE_PATH

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM


def print_gpu_memory(tag: str, device: str = "cuda") -> None:
    """Print GPU memory usage in a readable format."""
    if not torch.cuda.is_initialized():
        print(f"[{tag}] CUDA not initialized yet")
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"[{tag}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {max_allocated:.2f}GB")


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles torch.dtype."""

    def default(self, obj: Any) -> Any:  # noqa: ANN401
        """Convert value to a JSON serializable type."""

        import torch

        if isinstance(obj, torch.dtype):
            return str(obj)

        return json.JSONEncoder.default(self, obj)


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
        trust_remote_code: bool = False,
        **kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a quantized model with lm-eval."""

        from lm_eval import evaluator, models
        from lm_eval.tasks import TaskManager

        print(f"Tasks: {tasks}")

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

        model.eval()
        
        print_gpu_memory("After quantize", device)
        lm = models.huggingface.HFLM(
            pretrained=model,
            device=device,
            max_length=1024
        )
        print_gpu_memory("After HFLM", device)

        lm._model.config.use_cache = False

        # Clear memory before evaluation
        torch.cuda.empty_cache()
        print_gpu_memory("Before evaluate", device)

        return evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            device=device,
            # batch_size=1,
            task_manager=TaskManager(
                include_path=(Path(__file__).parent / "tasks").as_posix(),
            ),
        )


class PTQEvaluator(PTQEvaluatorImpl):
    """Base class for post-training quantization evaluators."""

    @modal.method()
    def evaluate(self, *args: list[str], **kwargs: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a quantized model."""
        model_name = kwargs["model_name"]
        ptq_method = kwargs["ptq_method"]

        results = self.evaluate_impl(*args, **kwargs)

        logs_path = (
            # FOUROVERSIX_CACHE_PATH
            # / "ptq_logs"
            Path("ptq_logs")
            / ptq_method.value
            / f"{datetime.now(tz=tzlocal()).strftime('%Y%m%d%H%M%S')}_{model_name}.json"
        )
        logs_path.parent.mkdir(parents=True, exist_ok=True)

        # results = self.evaluate_impl(*args, **kwargs)

        with logs_path.open("w") as f:
            json.dump(
                {
                    "model_name": model_name,
                    "ptq_method": ptq_method.value,
                    "kwargs": kwargs,
                    "results": results,
                },
                f,
                indent=4,
                cls=CustomJSONEncoder,
            )

        return results
