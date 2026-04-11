from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
RUNS_ROOT = PROJECT_ROOT / "runs"


@dataclass
class Settings:
    minimax_api_key: str
    moonshot_api_key: str
    minimax_api_base: str = "https://api.minimaxi.com/v1"
    minimax_model: str = "music-2.6"
    minimax_output_format: str = "url"
    kimi_api_base: str = "https://api.moonshot.cn/v1"
    kimi_model: str = "kimi-k2.5"
    sample_rate: int = 44100
    bitrate: int = 256000
    audio_format: str = "mp3"
    max_iterations: int = 3
    target_score: float = 0.82
    clap_ckpt_path: str = ""
    enable_clap: bool = True
    enable_audiobox: bool = True

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env")
        return cls(
            minimax_api_key=os.getenv("MINIMAX_API_KEY", ""),
            moonshot_api_key=os.getenv("MOONSHOT_API_KEY", ""),
            minimax_api_base=os.getenv("MINIMAX_API_BASE", "https://api.minimaxi.com/v1"),
            minimax_model=os.getenv("MINIMAX_MODEL", "music-2.6"),
            minimax_output_format=os.getenv("MINIMAX_OUTPUT_FORMAT", "url"),
            kimi_api_base=os.getenv("KIMI_API_BASE", "https://api.moonshot.cn/v1"),
            kimi_model=os.getenv("KIMI_MODEL", "kimi-k2.5"),
            sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "44100")),
            bitrate=int(os.getenv("AUDIO_BITRATE", "256000")),
            audio_format=os.getenv("AUDIO_FORMAT", "mp3"),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "3")),
            target_score=float(os.getenv("TARGET_SCORE", "0.82")),
            clap_ckpt_path=os.getenv("CLAP_CKPT_PATH", ""),
            enable_clap=os.getenv("ENABLE_CLAP", "true").lower() != "false",
            enable_audiobox=os.getenv("ENABLE_AUDIOBOX", "true").lower() != "false",
        )

    def validate_for_live_run(self) -> None:
        missing = []
        if not self.moonshot_api_key:
            missing.append("MOONSHOT_API_KEY")
        if not self.minimax_api_key:
            missing.append("MINIMAX_API_KEY")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


RUNS_ROOT.mkdir(parents=True, exist_ok=True)

