from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..common import build_result, clap_text_similarity
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
    text = check.text.lower()
    required = not any(token in text for token in ["instrumental", "no vocals", "without vocals", "无人声", "纯音乐"])
    gender = "female" if any(token in text for token in ["female", "woman", "girl", "女声"]) else (
        "male" if any(token in text for token in ["male", "man", "boy", "男声"]) else "any"
    )

    labels = ["instrumental music only", "a song with singing vocals"]
    if gender == "female":
        labels.append("a song with female singing vocals")
    elif gender == "male":
        labels.append("a song with male singing vocals")

    _, score_map = clap_text_similarity(wav_path=wav_path, texts=labels, state=state)
    instrumental_score = float(score_map.get("instrumental music only", 0.0))
    vocal_score = float(score_map.get("a song with singing vocals", 0.0))
    if gender == "female":
        vocal_score = max(vocal_score, float(score_map.get("a song with female singing vocals", 0.0)))
    elif gender == "male":
        vocal_score = max(vocal_score, float(score_map.get("a song with male singing vocals", 0.0)))

    margin = (vocal_score - instrumental_score) if required else (instrumental_score - vocal_score)
    score = float(np.clip((margin + 1.0) / 2.0, 0.0, 1.0))
    return build_result(
        check=check,
        skill_id="vocal_presence_checker",
        passed=margin >= 0.03,
        score=score,
        summary=f"Vocal presence score {score:.3f}.",
        evidence={
            "required": required,
            "gender": gender,
            "instrumental_score": instrumental_score,
            "vocal_score": vocal_score,
        },
    )
