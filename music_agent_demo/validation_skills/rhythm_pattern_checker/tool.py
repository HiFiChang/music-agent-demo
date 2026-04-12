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
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beat_times) < 4:
        return build_result(
            check=check,
            skill_id="rhythm_pattern_checker",
            passed=False,
            score=0.0,
            summary="Not enough beats detected for rhythm validation.",
            evidence={"beat_count": int(len(beat_times))},
        )

    intervals = np.diff(beat_times)
    regularity = float(1.0 / (1.0 + np.std(intervals) / (np.mean(intervals) + 1e-9)))
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_band = stft[freqs < 150].mean(axis=0)
    times = librosa.frames_to_time(np.arange(low_band.shape[0]), sr=sr, hop_length=512)
    beat_period = float(np.mean(intervals))
    onbeat_values = []
    offbeat_values = []
    for beat in beat_times:
        on_mask = np.abs(times - beat) <= 0.08
        off_mask = np.abs(times - (beat + beat_period / 2.0)) <= 0.08
        if np.any(on_mask):
            onbeat_values.append(float(np.mean(low_band[on_mask])))
        if np.any(off_mask):
            offbeat_values.append(float(np.mean(low_band[off_mask])))

    pulse_strength = float(
        np.mean(onbeat_values) / (np.mean(onbeat_values) + np.mean(offbeat_values) + 1e-9)
    ) if onbeat_values and offbeat_values else regularity

    text = check.text.lower()
    pattern = "four_on_the_floor" if "four on the floor" in text or "four-on-the-floor" in text or "四踩" in text else "steady_pulse"
    score = 0.55 * pulse_strength + 0.45 * regularity if pattern == "four_on_the_floor" else regularity
    min_score = 0.65 if any(token in text for token in ["稳定", "steady", "grounded", "别飘", "tight"]) else 0.58
    return build_result(
        check=check,
        skill_id="rhythm_pattern_checker",
        passed=score >= min_score,
        score=score,
        summary=f"Rhythm pattern score {score:.3f} for {pattern}.",
        evidence={
            "pattern": pattern,
            "regularity": regularity,
            "pulse_strength": pulse_strength,
            "min_score": min_score,
        },
    )
