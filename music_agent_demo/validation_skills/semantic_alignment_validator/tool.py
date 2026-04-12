from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common import build_result, clap_text_similarity, default_semantic_texts
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
    texts = default_semantic_texts(check, brief, original_query)
    score, score_map = clap_text_similarity(wav_path=wav_path, texts=texts, state=state)
    threshold = 0.62
    return build_result(
        check=check,
        skill_id="semantic_alignment_validator",
        passed=score >= threshold,
        score=score,
        summary=f"Semantic alignment {score:.3f}.",
        evidence={"text_scores": score_map, "threshold": threshold},
    )
