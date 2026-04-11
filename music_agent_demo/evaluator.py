from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio_utils import transcode_to_wav
from .config import Settings
from .schemas import EvaluationResult, PromptBrief
from .utils import dedupe_keep_order


class AudioEvaluator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._clap_model = None
        self._aes_predictor = None

    def evaluate(self, audio_path: Path, brief: PromptBrief, original_query: str) -> tuple[EvaluationResult, Path]:
        notes: list[str] = []
        wav_path = transcode_to_wav(audio_path)

        clap_mean = None
        clap_scores: dict[str, float] = {}
        if self.settings.enable_clap:
            try:
                clap_mean, clap_scores = self._evaluate_clap(wav_path, brief, original_query)
            except Exception as exc:  # pragma: no cover - runtime backend variance
                notes.append(f"CLAP skipped: {exc}")

        aesthetic_score = None
        aesthetic_axes: dict[str, float] = {}
        if self.settings.enable_audiobox:
            try:
                aesthetic_score, aesthetic_axes = self._evaluate_audiobox(wav_path)
            except Exception as exc:  # pragma: no cover - runtime backend variance
                notes.append(f"Audiobox skipped: {exc}")

        heuristic_score, heuristics = self._evaluate_heuristics(wav_path)

        total_score = self._combine_scores(
            clap_mean=clap_mean,
            aesthetic_score=aesthetic_score,
            heuristic_score=heuristic_score,
        )

        return (
            EvaluationResult(
                total_score=total_score,
                clap_mean=clap_mean,
                clap_scores=clap_scores,
                aesthetic_score=aesthetic_score,
                aesthetic_axes=aesthetic_axes,
                heuristic_score=heuristic_score,
                heuristics=heuristics,
                notes=notes,
            ),
            wav_path,
        )

    def _evaluate_clap(
        self,
        wav_path: Path,
        brief: PromptBrief,
        original_query: str,
    ) -> tuple[float, dict[str, float]]:
        if self._clap_model is None:
            from transformers import ClapModel, ClapProcessor
            import torch

            model_id = "laion/clap-htsat-unfused"
            self._clap_processor = ClapProcessor.from_pretrained(model_id)
            self._clap_model = ClapModel.from_pretrained(model_id)

        texts = dedupe_keep_order(
            [original_query, brief.intent_summary, *brief.evaluation_texts]
        )
        if not texts:
            texts = [original_query]

        import librosa
        import torch

        # Load audio at 48000 Hz, which is typical for CLAP
        y, sr = librosa.load(wav_path, sr=48000, mono=True)

        inputs = self._clap_processor(
            text=texts,
            audios=y,
            return_tensors="pt",
            sampling_rate=48000,
            padding=True
        )

        with torch.no_grad():
            outputs = self._clap_model(**inputs)
            
            audio_embeds = outputs.audio_embeds
            text_embeds = outputs.text_embeds
            
            audio_embeds = audio_embeds / (audio_embeds.norm(dim=-1, keepdim=True) + 1e-12)
            text_embeds = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-12)
            
            sims = torch.matmul(audio_embeds, text_embeds.T).squeeze(0)
            normalized = torch.clamp((sims + 1.0) / 2.0, 0.0, 1.0).cpu().numpy()

        score_map = {text: float(score) for text, score in zip(texts, normalized)}
        return float(np.mean(normalized)), score_map

    def _evaluate_audiobox(self, wav_path: Path) -> tuple[float, dict[str, float]]:
        if self._aes_predictor is None:
            from audiobox_aesthetics.infer import initialize_predictor

            self._aes_predictor = initialize_predictor()

        prediction = self._aes_predictor.forward([{"path": str(wav_path)}])
        if isinstance(prediction, list):
            prediction = prediction[0]

        axes = {
            "CE": float(prediction.get("CE", 0.0)),
            "CU": float(prediction.get("CU", 0.0)),
            "PC": float(prediction.get("PC", 0.0)),
            "PQ": float(prediction.get("PQ", 0.0)),
        }
        weighted = (
            0.35 * axes["CE"]
            + 0.15 * axes["CU"]
            + 0.15 * axes["PC"]
            + 0.35 * axes["PQ"]
        ) / 10.0
        return weighted, axes

    def _evaluate_heuristics(self, wav_path: Path) -> tuple[float, dict[str, float]]:
        import librosa

        y, sr = librosa.load(wav_path, sr=None, mono=True)
        duration = float(len(y) / sr) if sr else 0.0
        rms = librosa.feature.rms(y=y)[0]
        mean_rms = float(np.mean(rms)) if len(rms) else 0.0
        silence_ratio = float(np.mean(rms < 0.01)) if len(rms) else 1.0
        peak_abs = float(np.max(np.abs(y))) if len(y) else 0.0
        clipping_ratio = float(np.mean(np.abs(y) > 0.99)) if len(y) else 1.0

        duration_score = self._duration_score(duration)
        loudness_score = self._band_score(mean_rms, low=0.03, high=0.25)
        silence_score = max(0.0, 1.0 - silence_ratio)
        clipping_score = max(0.0, 1.0 - min(clipping_ratio * 20.0, 1.0))

        heuristics = {
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
        heuristic_score = float(
            np.mean([duration_score, loudness_score, silence_score, clipping_score])
        )
        return heuristic_score, heuristics

    @staticmethod
    def _band_score(value: float, low: float, high: float) -> float:
        if low <= value <= high:
            return 1.0
        if value < low:
            return max(0.0, value / low)
        return max(0.0, 1.0 - ((value - high) / max(high, 1e-6)))

    @staticmethod
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

    @staticmethod
    def _combine_scores(
        clap_mean: float | None,
        aesthetic_score: float | None,
        heuristic_score: float,
    ) -> float:
        weighted_total = 0.0
        total_weight = 0.0

        if clap_mean is not None:
            weighted_total += 0.55 * clap_mean
            total_weight += 0.55
        if aesthetic_score is not None:
            weighted_total += 0.35 * aesthetic_score
            total_weight += 0.35

        weighted_total += 0.10 * heuristic_score
        total_weight += 0.10
        return weighted_total / total_weight
