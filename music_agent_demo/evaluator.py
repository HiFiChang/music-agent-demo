from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from .audio_utils import transcode_to_wav
from .config import Settings
from .llm_client import KimiClient
from .schemas import (
    EvaluationResult,
    PromptBrief,
    ValidationCheck,
    ValidationCheckResult,
    ValidationPlan,
)
from .skill_manager import SkillManager


CHECKLIST_COMPILER_PROMPT = """You are compiling a validation checklist for a music generation optimization loop.

User request:
{user_prompt}

Current generation brief JSON:
{brief_json}

Available validator skills:
{validator_manifest}

Audiobox enabled:
{audiobox_enabled}

Return JSON only in this shape:
{{
  "checks": [
    {{
      "text": "One concrete natural-language validation sentence.",
      "hard": true,
      "weight": 0.25
    }}
  ]
}}

Rules:
- Each check must be one sentence of natural language.
- Each check must be atomic enough that exactly one validator skill can verify it.
- Always include one overall semantic-intent check.
- Always include one basic audio-health check.
- Add other checks only when they are clearly implied by the request or brief.
- Write checks so that one of the available validator skills can verify them.
- Make weights positive numbers; the system will normalize them later.
- Do not mention validator skill IDs in the checks.
- Keep checks audibly testable, concrete, and concise.
"""


VALIDATOR_ROUTER_PROMPT = """You are routing checklist items to validator skills.

Available validator skills:
{manifest}

Checklist:
{checks_json}

Return JSON only in this shape:
{{
  "routes": [
    {{"check_id": "check_01", "skill_id": "validator_skill_id"}}
  ]
}}

Rules:
- Every check_id must appear exactly once.
- Pick the validator skill whose description most directly matches the checklist sentence.
- Do not invent skill IDs.
"""


VERIFIER_PROMPT = """You are a music verification agent in a prompt-optimization loop.

Original user request:
{user_prompt}

Current generation prompt:
{generation_prompt}

Checklist:
{checks_json}

Validator results:
{results_json}

Return JSON only:
{{
  "summary": "short overall summary",
  "hard_failures": ["check failure 1"],
  "protected_checks": ["satisfied check 1"],
  "next_prompt_guidance": [
    "concrete prompt improvement 1",
    "concrete prompt improvement 2"
  ]
}}

Rules:
- Base everything on the validator evidence.
- Hard failures come first.
- Guidance must be concrete and generator-friendly.
- Preserve already-satisfied checks.
"""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


class AudioEvaluator:
    def __init__(self, settings: Settings, llm: KimiClient):
        self.settings = settings
        self.llm = llm
        self.validation_skill_manager = SkillManager(
            Path(__file__).resolve().parent / "validation_skills"
        )
        self._tool_state: dict[str, Any] = {}
        self._tool_modules: dict[str, ModuleType] = {}

    def build_validation_plan(self, user_prompt: str, brief: PromptBrief) -> ValidationPlan:
        prompt = CHECKLIST_COMPILER_PROMPT.format(
            user_prompt=user_prompt,
            brief_json=json.dumps(brief.to_dict(), ensure_ascii=False, indent=2),
            validator_manifest=self.validation_skill_manager.get_skill_manifest(),
            audiobox_enabled=str(self.settings.use_audiobox).lower(),
        )
        payload = self.llm.complete_json(prompt)
        if not isinstance(payload, dict) or "checks" not in payload or not isinstance(payload["checks"], list):
            raise ValueError("Checklist compiler must return a JSON object with a 'checks' list.")

        checks = self._normalize_compiled_checks(payload["checks"])
        total_weight = sum(item.weight for item in checks)
        if total_weight <= 0:
            raise ValueError("Checklist weights must sum to a positive value.")
        for item in checks:
            item.weight = item.weight / total_weight
        return ValidationPlan(checks=checks)

    def _normalize_compiled_checks(self, raw_checks: list[Any]) -> list[ValidationCheck]:
        if not raw_checks:
            raise ValueError("Checklist compiler returned no checks.")

        checks: list[ValidationCheck] = []
        for index, raw in enumerate(raw_checks, start=1):
            if not isinstance(raw, dict):
                raise ValueError("Each compiled check must be an object.")
            text = str(raw.get("text") or raw.get("check") or "").strip()
            weight = _safe_float(raw.get("weight"), 0.0)
            hard = _safe_bool(raw.get("hard"), default=False)
            if not text:
                raise ValueError("Compiled checklist item is missing text.")
            if weight <= 0:
                raise ValueError(f"Checklist item '{text}' must have a positive weight.")
            checks.append(
                ValidationCheck(
                    check_id=f"check_{index:02d}",
                    text=text,
                    weight=weight,
                    hard=hard,
                )
            )
        return checks

    def evaluate(
        self,
        audio_path: Path,
        brief: PromptBrief,
        original_query: str,
        validation_plan: ValidationPlan,
    ) -> tuple[EvaluationResult, Path]:
        wav_path = transcode_to_wav(audio_path)
        routes = self._route_validators(validation_plan)

        check_results: list[ValidationCheckResult] = []
        clap_mean = None
        clap_scores: dict[str, float] = {}
        aesthetic_score = None
        aesthetic_axes: dict[str, float] = {}
        heuristic_score = 0.0
        heuristics: dict[str, float] = {}

        checks_by_id = {item.check_id: item for item in validation_plan.checks}

        for route in routes:
            check = checks_by_id[route["check_id"]]
            result = self._run_validation_skill(
                skill_id=route["skill_id"],
                check=check,
                wav_path=wav_path,
                brief=brief,
                original_query=original_query,
            )
            check_results.append(result)

            if route["skill_id"] == "semantic_alignment_validator":
                clap_mean = result.score
                clap_scores = dict(result.evidence.get("text_scores", {}))
            elif route["skill_id"] == "aesthetic_quality_checker":
                aesthetic_score = result.score
                aesthetic_axes = dict(result.evidence.get("axes", {}))
            elif route["skill_id"] == "mix_health_checker":
                heuristic_score = result.score
                heuristics = dict(result.evidence)

        validator_score = float(
            sum(
                checks_by_id[item.check_id].weight * item.score
                for item in check_results
            )
        )

        verifier_payload = self._run_verifier_agent(
            user_prompt=original_query,
            generation_prompt=brief.generation_prompt,
            checks=validation_plan.checks,
            results=check_results,
        )

        return (
            EvaluationResult(
                total_score=validator_score,
                validator_score=validator_score,
                clap_mean=clap_mean,
                clap_scores=clap_scores,
                aesthetic_score=aesthetic_score,
                aesthetic_axes=aesthetic_axes,
                heuristic_score=heuristic_score,
                heuristics=heuristics,
                notes=[],
                validator_skill_ids=[item["skill_id"] for item in routes],
                check_results=check_results,
                verifier_summary=str(verifier_payload.get("summary", "")).strip(),
                next_prompt_guidance=[
                    str(item).strip()
                    for item in verifier_payload.get("next_prompt_guidance", [])
                    if str(item).strip()
                ],
                hard_failures=[
                    str(item).strip()
                    for item in verifier_payload.get("hard_failures", [])
                    if str(item).strip()
                ],
                protected_checks=[
                    str(item).strip()
                    for item in verifier_payload.get("protected_checks", [])
                    if str(item).strip()
                ],
            ),
            wav_path,
        )

    def _route_validators(self, validation_plan: ValidationPlan) -> list[dict[str, str]]:
        prompt = VALIDATOR_ROUTER_PROMPT.format(
            manifest=self.validation_skill_manager.get_skill_manifest(),
            checks_json=validation_plan.to_json(),
        )
        payload = self.llm.complete_json(prompt)
        if not isinstance(payload, dict) or "routes" not in payload or not isinstance(payload["routes"], list):
            raise ValueError("Validator router must return a JSON object with a 'routes' list.")

        valid_skill_ids = set(self.validation_skill_manager.skills.keys())
        valid_check_ids = {item.check_id for item in validation_plan.checks}
        routes: list[dict[str, str]] = []
        seen = set()
        for item in payload["routes"]:
            if not isinstance(item, dict):
                continue
            check_id = str(item.get("check_id", "")).strip()
            skill_id = str(item.get("skill_id", "")).strip()
            if check_id in valid_check_ids and skill_id in valid_skill_ids:
                routes.append({"check_id": check_id, "skill_id": skill_id})
                seen.add(check_id)

        missing = valid_check_ids - seen
        if missing:
            raise ValueError(f"Validator router did not assign all checks: {sorted(missing)}")
        return routes

    def _run_validation_skill(
        self,
        *,
        skill_id: str,
        check: ValidationCheck,
        wav_path: Path,
        brief: PromptBrief,
        original_query: str,
    ) -> ValidationCheckResult:
        skill = self.validation_skill_manager.get_skill(skill_id)
        if not skill:
            raise ValueError(f"Unknown validation skill: {skill_id}")
        tool_path = skill.get("tool_path", "")
        if not tool_path:
            raise ValueError(f"Validation skill '{skill_id}' does not provide a tool.py")

        module = self._load_tool_module(skill_id, Path(tool_path))
        validate_fn = getattr(module, "validate", None)
        if validate_fn is None:
            raise ValueError(f"Validation skill '{skill_id}' tool.py does not export validate()")

        result = validate_fn(
            check=check,
            wav_path=wav_path,
            brief=brief,
            original_query=original_query,
            settings=self.settings,
            llm=self.llm,
            state=self._tool_state,
        )
        if not isinstance(result, ValidationCheckResult):
            raise ValueError(f"Validation skill '{skill_id}' must return ValidationCheckResult")
        return result

    def _load_tool_module(self, skill_id: str, tool_path: Path) -> ModuleType:
        if skill_id in self._tool_modules:
            return self._tool_modules[skill_id]

        module_name = f"music_agent_demo.validation_skills.{skill_id}.tool"
        spec = importlib.util.spec_from_file_location(module_name, tool_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Unable to load validator tool module for '{skill_id}'")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._tool_modules[skill_id] = module
        return module

    def _run_verifier_agent(
        self,
        user_prompt: str,
        generation_prompt: str,
        checks: list[ValidationCheck],
        results: list[ValidationCheckResult],
    ) -> dict[str, Any]:
        payload = self.llm.complete_json(
            VERIFIER_PROMPT.format(
                user_prompt=user_prompt,
                generation_prompt=generation_prompt,
                checks_json=str([item.to_dict() for item in checks]),
                results_json=str([item.to_dict() for item in results]),
            )
        )
        if not isinstance(payload, dict):
            raise ValueError("Verifier agent must return a JSON object.")
        return payload
