from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ..schemas import PromptBrief, ValidationCheck, ValidationCheckResult
from ..utils import dedupe_keep_order


SUPPORTED_TONES = {"warm", "bright", "dark", "soft", "aggressive", "airy"}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_clap(state: dict[str, Any]):
    if state.get("clap_model") is None:
        from transformers import ClapModel, ClapProcessor

        state["clap_processor"] = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        state["clap_model"] = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    return state["clap_model"], state["clap_processor"]


def get_audiobox_predictor(state: dict[str, Any]):
    if state.get("aes_predictor") is None:
        from audiobox_aesthetics.infer import initialize_predictor

        state["aes_predictor"] = initialize_predictor()
    return state["aes_predictor"]


def clap_text_similarity(
    *,
    wav_path: Path,
    texts: list[str],
    state: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    import librosa
    import torch

    clap_model, clap_processor = get_clap(state)
    normalized_texts = dedupe_keep_order([text for text in texts if str(text).strip()])
    if not normalized_texts:
        return 0.0, {}

    y, _ = librosa.load(wav_path, sr=48000, mono=True)
    inputs = clap_processor(
        text=normalized_texts,
        audio=y,
        return_tensors="pt",
        sampling_rate=48000,
        padding=True,
    )

    with torch.no_grad():
        outputs = clap_model(**inputs)
        audio_embeds = outputs.audio_embeds
        text_embeds = outputs.text_embeds
        audio_embeds = audio_embeds / (audio_embeds.norm(dim=-1, keepdim=True) + 1e-12)
        text_embeds = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-12)
        sims = torch.matmul(audio_embeds, text_embeds.T).squeeze(0)
        normalized = torch.clamp((sims + 1.0) / 2.0, 0.0, 1.0).cpu().numpy()

    score_map = {text: float(value) for text, value in zip(normalized_texts, normalized)}
    return float(np.mean(normalized)), score_map


def parse_bpm(check_text: str) -> float | None:
    text = check_text.lower()
    match = re.search(r"(\d{2,3})\s*bpm", text)
    if match:
        bpm = float(match.group(1))
        if 40 <= bpm <= 220:
            return bpm
    match = re.search(r"\b(\d{2,3})\b", text)
    if match:
        bpm = float(match.group(1))
        if 40 <= bpm <= 220:
            return bpm
    return None


def parse_tones(check_text: str) -> list[str]:
    text = check_text.lower()
    mappings = {
        "warm": "warm",
        "bright": "bright",
        "dark": "dark",
        "soft": "soft",
        "aggressive": "aggressive",
        "airy": "airy",
        "cold": "dark",
        "smoky": "dark",
        "spacious": "airy",
        "empty": "airy",
        "温暖": "warm",
        "明亮": "bright",
        "黑暗": "dark",
        "冷": "dark",
        "空": "airy",
        "柔和": "soft",
        "攻击性": "aggressive",
        "空气感": "airy",
    }
    tones = []
    for token, tone in mappings.items():
        if token in text:
            tones.append(tone)
    tones = dedupe_keep_order(tones)
    return [tone for tone in tones if tone in SUPPORTED_TONES]


def check_text_lower(check: ValidationCheck) -> str:
    return check.text.strip().lower()


def build_result(
    *,
    check: ValidationCheck,
    skill_id: str,
    passed: bool,
    score: float,
    summary: str,
    evidence: dict[str, Any] | None = None,
) -> ValidationCheckResult:
    return ValidationCheckResult(
        check_id=check.check_id,
        check_text=check.text,
        skill_id=skill_id,
        passed=passed,
        score=float(score),
        summary=summary,
        evidence=evidence or {},
    )


def default_semantic_texts(check: ValidationCheck, brief: PromptBrief, original_query: str) -> list[str]:
    return dedupe_keep_order([check.text, original_query, brief.intent_summary, *brief.evaluation_texts])
