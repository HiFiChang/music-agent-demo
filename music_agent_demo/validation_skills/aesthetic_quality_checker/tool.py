from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common import build_result, get_audiobox_predictor
from ...schemas import PromptBrief, ValidationCheck


def validate(
    *,
    check: ValidationCheck,
    wav_path: Path,
    brief: PromptBrief,
    original_query: str,
    settings: Any,
    llm: Any,
    state: dict[str, Any],
):
    predictor = get_audiobox_predictor(state)
    prediction = predictor.forward([{"path": str(wav_path)}])
    if isinstance(prediction, list):
        prediction = prediction[0]

    axes = {
        "CE": float(prediction.get("CE", 0.0)),
        "CU": float(prediction.get("CU", 0.0)),
        "PC": float(prediction.get("PC", 0.0)),
        "PQ": float(prediction.get("PQ", 0.0)),
    }
    score = (
        0.35 * axes["CE"]
        + 0.15 * axes["CU"]
        + 0.15 * axes["PC"]
        + 0.35 * axes["PQ"]
    ) / 10.0
    threshold = 0.55
    return build_result(
        check=check,
        skill_id="aesthetic_quality_checker",
        passed=score >= threshold,
        score=score,
        summary=f"Aesthetic quality score {score:.3f}.",
        evidence={"axes": axes, "threshold": threshold},
    )
