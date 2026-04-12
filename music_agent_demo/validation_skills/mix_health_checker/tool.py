from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np

from ..common import build_result
from ...schemas import PromptBrief, ValidationCheck


def _band_score(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, value / low)
    return max(0.0, 1.0 - ((value - high) / max(high, 1e-6)))


def _duration_score(duration: float) -> float:
    if duration <= 0:
        return 0.0
    if duration < 8:
        return duration / 8.0
    if duration <= 90:
        return 1.0
    if duration <= 180:
        return max(0.5, 1.0 - ((duration - 90.0) / 180.0))
    return 0.3


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
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = float(len(y) / sr) if sr else 0.0
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms)) if len(rms) else 0.0
    silence_ratio = float(np.mean(rms < 0.01)) if len(rms) else 1.0
    peak_abs = float(np.max(np.abs(y))) if len(y) else 0.0
    clipping_ratio = float(np.mean(np.abs(y) > 0.99)) if len(y) else 1.0

    duration_score = _duration_score(duration)
    loudness_score = _band_score(mean_rms, low=0.03, high=0.25)
    silence_score = max(0.0, 1.0 - silence_ratio)
    clipping_score = max(0.0, 1.0 - min(clipping_ratio * 20.0, 1.0))
    score = float(np.mean([duration_score, loudness_score, silence_score, clipping_score]))
    evidence = {
        "duration_seconds": duration,
        "duration_score": duration_score,
        "mean_rms": mean_rms,
        "loudness_score": loudness_score,
        "silence_ratio": silence_ratio,
        "silence_score": silence_score,
        "peak_abs": peak_abs,
        "clipping_ratio": clipping_ratio,
        "clipping_score": clipping_score,
    }
    return build_result(
        check=check,
        skill_id="mix_health_checker",
        passed=score >= 0.60,
        score=score,
        summary=f"Mix health score {score:.3f}.",
        evidence=evidence,
    )
