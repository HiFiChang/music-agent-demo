from __future__ import annotations

import subprocess
from pathlib import Path


def transcode_to_wav(audio_path: Path, target_sample_rate: int = 48000) -> Path:
    if audio_path.suffix.lower() == ".wav":
        return audio_path

    wav_path = audio_path.with_suffix(".wav")
    subprocess.run(
        [
            "ffmpeg",
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

