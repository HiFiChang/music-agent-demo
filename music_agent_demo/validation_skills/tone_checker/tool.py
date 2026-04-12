from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np

from ..common import build_result, parse_tones
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
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_energy = float(stft[freqs < 250].mean()) if np.any(freqs < 250) else 0.0
    high_energy = float(stft[freqs > 4000].mean()) if np.any(freqs > 4000) else 0.0
    low_high_ratio = low_energy / (high_energy + 1e-9)

    tones = parse_tones(check.text)
    scores = []
    for tone in tones:
        if tone in {"warm", "soft", "dark"}:
            score = 0.5 * np.clip(1.0 - centroid / 3500.0, 0.0, 1.0) + 0.5 * np.clip(low_high_ratio / 3.0, 0.0, 1.0)
        elif tone in {"bright", "airy"}:
            score = 0.5 * np.clip(centroid / 3500.0, 0.0, 1.0) + 0.5 * np.clip(rolloff / 8000.0, 0.0, 1.0)
        elif tone == "aggressive":
            score = 0.5 * np.clip(centroid / 4000.0, 0.0, 1.0) + 0.5 * np.clip(high_energy / (low_energy + 1e-9), 0.0, 1.0)
        else:
            score = 0.5
        scores.append(float(score))
    final_score = float(np.mean(scores)) if scores else 0.5
    return build_result(
        check=check,
        skill_id="tone_checker",
        passed=final_score >= 0.55,
        score=final_score,
        summary=f"Tone profile score {final_score:.3f}.",
        evidence={
            "tones": tones,
            "spectral_centroid": centroid,
            "spectral_rolloff": rolloff,
            "low_high_ratio": low_high_ratio,
        },
    )
