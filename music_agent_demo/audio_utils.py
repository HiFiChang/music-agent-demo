from __future__ import annotations

import subprocess
from pathlib import Path
import shutil
import sys


def _resolve_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    candidate = Path(sys.executable).resolve().parent / "ffmpeg"
    if candidate.exists():
        return str(candidate)

    raise FileNotFoundError("ffmpeg not found in PATH or current environment bin directory.")


def transcode_to_wav(audio_path: Path, target_sample_rate: int = 48000) -> Path:
    if audio_path.suffix.lower() == ".wav":
        return audio_path

    wav_path = audio_path.with_suffix(".wav")
    ffmpeg = _resolve_ffmpeg()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            str(target_sample_rate),
            str(wav_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return wav_path
