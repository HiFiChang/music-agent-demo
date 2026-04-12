from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np

from ..common import build_result
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
    y, _ = librosa.load(wav_path, sr=None, mono=True)
    frame_rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    sections = np.array_split(frame_rms, 4)
    section_means = [float(np.mean(item)) if len(item) else 0.0 for item in sections]
    early_reference = max(float(np.mean(section_means[:2])), 1e-9)
    late_peak = max(section_means[2:]) if len(section_means) >= 4 else max(section_means)
    late_peak_index = int(np.argmax(section_means))
    ratio = float(late_peak / early_reference)

    text = check.text.lower()
    min_ratio = 1.25 if any(token in text for token in ["明显", "dramatic", "explodes", "maximal", "open up clearly", "明显打开"]) else 1.10
    should_peak_late = not any(token in text for token in ["early peak", "early lift"])
    passed = ratio >= min_ratio and (late_peak_index >= 2 if should_peak_late else True)
    score = min(1.0, ratio / max(min_ratio, 1e-9))
    return build_result(
        check=check,
        skill_id="section_energy_checker",
        passed=passed,
        score=score,
        summary=f"Section lift ratio {ratio:.3f}, peak section {late_peak_index}.",
        evidence={
            "section_mean_rms": section_means,
            "lift_ratio": ratio,
            "late_peak_index": late_peak_index,
            "min_ratio": min_ratio,
            "peak_should_be_late": should_peak_late,
        },
    )
