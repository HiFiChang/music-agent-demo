from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


@dataclass
class PromptBrief:
    title: str
    intent_summary: str
    is_instrumental: bool
    generation_prompt: str
    lyrics: str = ""
    use_lyrics_optimizer: bool = False
    evaluation_texts: list[str] = field(default_factory=list)
    focus_tags: list[str] = field(default_factory=list)
    avoid_tags: list[str] = field(default_factory=list)
    selected_skill: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], selected_skill: str | None = None) -> "PromptBrief":
        generation_prompt = str(
            data.get("generation_prompt")
            or data.get("prompt")
            or data.get("optimized_prompt")
            or ""
        ).strip()
        intent_summary = str(data.get("intent_summary") or data.get("summary") or generation_prompt).strip()
        title = str(data.get("title") or "Untitled Demo Run").strip()
        return cls(
            title=title,
            intent_summary=intent_summary,
            is_instrumental=_as_bool(data.get("is_instrumental"), default=False),
            generation_prompt=generation_prompt,
            lyrics=str(data.get("lyrics") or "").strip(),
            use_lyrics_optimizer=_as_bool(data.get("use_lyrics_optimizer"), default=False),
            evaluation_texts=_as_str_list(data.get("evaluation_texts")),
            focus_tags=_as_str_list(data.get("focus_tags")),
            avoid_tags=_as_str_list(data.get("avoid_tags")),
            selected_skill=selected_skill,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationCheck:
    check_id: str
    text: str
    weight: float
    hard: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationPlan:
    checks: list[ValidationCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"checks": [item.to_dict() for item in self.checks]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class ValidationCheckResult:
    check_id: str
    check_text: str
    skill_id: str
    passed: bool
    score: float
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    total_score: float
    validator_score: float = 0.0
    clap_mean: float | None = None
    clap_scores: dict[str, float] = field(default_factory=dict)
    aesthetic_score: float | None = None
    aesthetic_axes: dict[str, float] = field(default_factory=dict)
    heuristic_score: float = 0.0
    heuristics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    validator_skill_ids: list[str] = field(default_factory=list)
    check_results: list[ValidationCheckResult] = field(default_factory=list)
    verifier_summary: str = ""
    next_prompt_guidance: list[str] = field(default_factory=list)
    hard_failures: list[str] = field(default_factory=list)
    protected_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["check_results"] = [item.to_dict() for item in self.check_results]
        return data


@dataclass
class AttemptRecord:
    iteration: int
    prompt: str
    audio_path: str
    wav_path: str
    evaluation: EvaluationResult

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evaluation"] = self.evaluation.to_dict()
        return data
