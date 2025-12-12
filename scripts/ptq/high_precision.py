from typing import Any

from ..resources import (
    FOUROVERSIX_CACHE_PATH,
    app,
    cache_volume,
    get_image,
    hf_secret,
)
from .evaluator import PTQEvaluator

hp_img = get_image()

with hp_img.imports():
    from fouroversix.utils import AdaptiveBlockScalingRule
    from transformers import AutoModelForCausalLM


@app.cls(
    image=hp_img,
    gpu="B200",
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
class HighPrecisionEvaluator(PTQEvaluator):
    """Evaluate a model while keeping it in high precision."""

    def quantize_model(
        self,
        model_name: str,
        *,
        device: str,
        dtype: str,
        a_scale_rule: AdaptiveBlockScalingRule,
        w_scale_rule: AdaptiveBlockScalingRule,
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> "AutoModelForCausalLM":
        """Return a model without any quantization."""

        if (
            a_scale_rule != AdaptiveBlockScalingRule.always_6
            or w_scale_rule != AdaptiveBlockScalingRule.always_6
        ):
            msg = (
                "Setting a block scale selection rule is not supported when keeping "
                "models in high precision"
            )
            raise ValueError(msg)

        return AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            dtype=dtype,
        )
