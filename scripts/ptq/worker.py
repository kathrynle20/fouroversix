from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dateutil.tz import tzlocal

from .high_precision import HighPrecisionEvaluator
from .rtn import RTNEvaluator
from .utils import PTQMethod

if TYPE_CHECKING:
    import multiprocessing

    from .evaluator import PTQEvaluator


def get_evaluator(ptq_method: PTQMethod) -> type[PTQEvaluator]:
    """Get the evaluator class for the given PTQ method."""

    if ptq_method == PTQMethod.high_precision:
        return HighPrecisionEvaluator
    if ptq_method == PTQMethod.rtn:
        return RTNEvaluator

    msg = f"Unsupported PTQ method: {ptq_method}"
    raise ValueError(msg)


def worker(gpu_id: str, task_queue: multiprocessing.Queue) -> dict[str, Any]:
    while True:
        task = task_queue.get()

        if task is None:
            break

        model_name, ptq_method, kwargs = task

        evaluator_cls = get_evaluator(ptq_method)
        results = evaluator_cls().evaluate.local(
            model_name=model_name,
            ptq_method=ptq_method,
            **{
                **kwargs,
                "device": f"cuda:{gpu_id}",
            },
        )

        logs_path = Path("ptq_logs") / (
            f"{model_name}_{ptq_method.value}-{datetime.now(tz=tzlocal).strftime('%Y%m%d%H%M%S')}.json"
        )
        logs_path.parent.mkdir(parents=True, exist_ok=True)

        with logs_path.open("w") as f:
            json.dump(
                {
                    "model_name": model_name,
                    "ptq_method": ptq_method.value,
                    "kwargs": kwargs,
                    "results": results["results"],
                },
                f,
                indent=4,
            )
