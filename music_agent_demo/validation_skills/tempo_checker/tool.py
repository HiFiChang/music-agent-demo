from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa

from ..common import build_result, parse_bpm
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
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    target_bpm = parse_bpm(check.text)
    if target_bpm is None:
        return build_result(
            check=check,
            skill_id="tempo_checker",
            passed=False,
            score=0.0,
            summary="Tempo checker could not extract a BPM target from the checklist sentence.",
            evidence={"estimated_bpm": float(tempo), "beat_count": int(len(beats))},
        )

    tolerance = 8.0
    deviation = abs(float(tempo) - target_bpm)
    score = max(0.0, 1.0 - deviation / tolerance)
    return build_result(
        check=check,
        skill_id="tempo_checker",
        passed=deviation <= tolerance,
        score=score,
        summary=f"Estimated tempo {float(tempo):.1f} BPM vs target {target_bpm:.1f} BPM.",
        evidence={
            "estimated_bpm": float(tempo),
            "target_bpm": target_bpm,
            "tolerance_bpm": tolerance,
            "beat_count": int(len(beats)),
        },
    )
