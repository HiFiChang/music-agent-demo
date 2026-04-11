from __future__ import annotations

import binascii
from pathlib import Path

import requests

from .config import Settings
from .schemas import PromptBrief
from .utils import dump_json


class MiniMaxMusicClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def build_payload(self, brief: PromptBrief) -> dict:
        payload = {
            "model": self.settings.minimax_model,
            "prompt": brief.generation_prompt,
            "output_format": self.settings.minimax_output_format,
            "audio_setting": {
                "sample_rate": self.settings.sample_rate,
                "bitrate": self.settings.bitrate,
                "format": self.settings.audio_format,
                "aigc_watermark": False,
                "lyrics_optimizer": brief.use_lyrics_optimizer,
                "is_instrumental": brief.is_instrumental,
            },
        }
        # MiniMax API currently requires 'lyrics' field to exist and be string, even if instrumental.
        payload["lyrics"] = brief.lyrics if brief.lyrics else (brief.intent_summary or "Instrumental music, no lyrics.")
        return payload

    def generate(self, brief: PromptBrief, attempt_dir: Path) -> tuple[Path, dict]:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        payload = self.build_payload(brief)
        dump_json(attempt_dir / "request.json", payload)

        response = requests.post(
            f"{self.settings.minimax_api_base}/music_generation",
            headers={
                "Authorization": f"Bearer {self.settings.minimax_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        data = response.json()
        dump_json(attempt_dir / "response.json", data)

        base_resp = data.get("base_resp", {})
        if base_resp.get("status_code") not in {0, "0", None}:
            raise RuntimeError(f"MiniMax generation failed: {base_resp}")

        audio_blob = data.get("data", {}).get("audio")
        if not audio_blob:
            raise RuntimeError(f"MiniMax response missing audio: {data}")

        audio_path = attempt_dir / f"generated.{self.settings.audio_format}"
        if self.settings.minimax_output_format == "url":
            audio_response = requests.get(audio_blob, timeout=600)
            audio_response.raise_for_status()
            audio_path.write_bytes(audio_response.content)
        else:
            try:
                audio_bytes = binascii.unhexlify(audio_blob)
            except binascii.Error as exc:
                raise RuntimeError("Failed to decode MiniMax hex audio payload.") from exc
            audio_path.write_bytes(audio_bytes)
        return audio_path, data

