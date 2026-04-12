from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .config import RUNS_ROOT, Settings
from .evaluator import AudioEvaluator
from .llm_client import KimiClient
from .music_client import MiniMaxMusicClient
from .schemas import AttemptRecord, PromptBrief
from .skill_manager import SkillManager
from .utils import dump_json, slugify


PLANNER_DECISION_PROMPT = """You are a skill router for a music generation agent.

Available skills:
{manifest}

User request:
{user_prompt}

Rules:
1. Pick exactly one SKILL_ID only if it gives clear added value.
2. If the request is generic, answer NONE.
3. Answer with the SKILL_ID or NONE only.
"""


INITIAL_BRIEF_PROMPT = """You are designing a generation brief for MiniMax music-2.6.

User request:
{user_prompt}

Selected skill:
{skill_id}

Skill instructions:
{skill_instructions}

Return JSON only with this schema:
{{
  "title": "short run title",
  "intent_summary": "one-sentence summary of the music target",
  "is_instrumental": true,
  "generation_prompt": "optimized MiniMax prompt text",
  "lyrics": "",
  "use_lyrics_optimizer": false,
  "evaluation_texts": ["3 to 5 short captions for CLAP evaluation"],
  "focus_tags": ["genre", "mood", "instrument", "vocal or instrumental trait"],
  "avoid_tags": ["conflicting element 1", "conflicting element 2"]
}}

Rules:
- Favor concrete genre, mood, instrumentation, arrangement, and production descriptors.
- Keep the generation_prompt compact and generator-friendly.
- If the user wants vocals and did not provide exact lyrics, set use_lyrics_optimizer=true and keep lyrics empty.
- If the user explicitly wants instrumental music, set is_instrumental=true and use_lyrics_optimizer=false.
- evaluation_texts should be concrete and audibly verifiable, not abstract poetry.
- avoid artist-name references unless the user explicitly asked for them.
"""


REFINE_PROMPT_TEMPLATE = """You are refining a MiniMax music generation prompt.

Original user request:
{user_prompt}

Stable evaluation rubric:
{evaluation_texts}

Active skill:
{skill_id}

Skill instructions:
{skill_instructions}

Attempt history:
{history_log}

Return ONLY a revised generation prompt string.

Rules:
- Preserve the user's core intent.
- Reinforce the failing checklist items, especially hard failures.
- Use verifier guidance and check-level evidence directly.
- Protect already-satisfied checks to avoid regressions.
- Remove redundant or conflicting descriptors.
- Keep the prompt concise and generator-friendly.
"""


class MusicGenerationAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = KimiClient(settings)
        self.music_client = MiniMaxMusicClient(settings)
        self.evaluator = AudioEvaluator(settings, self.llm)
        self.skill_manager = SkillManager()

    def route_skill(self, user_prompt: str) -> str:
        manifest = self.skill_manager.get_skill_manifest()
        if not manifest.strip():
            return "NONE"
        prompt = PLANNER_DECISION_PROMPT.format(
            manifest=manifest,
            user_prompt=user_prompt,
        )
        decision = self.llm.complete_text(prompt).strip()
        return decision if decision in self.skill_manager.skills else "NONE"

    def build_initial_brief(
        self,
        user_prompt: str,
        skill_id: str,
        force_instrumental: bool = False,
    ) -> PromptBrief:
        skill = self.skill_manager.get_skill(skill_id)
        skill_instructions = skill["instructions"] if skill else "No skill selected."
        prompt = INITIAL_BRIEF_PROMPT.format(
            user_prompt=user_prompt,
            skill_id=skill_id,
            skill_instructions=skill_instructions,
        )
        data = self.llm.complete_json(prompt)
        if not isinstance(data, dict):
            raise ValueError("Initial brief prompt must return a JSON object.")

        brief = PromptBrief.from_dict(data, selected_skill=None if skill_id == "NONE" else skill_id)
        if force_instrumental:
            brief.is_instrumental = True
            brief.use_lyrics_optimizer = False
            brief.lyrics = ""
        if not brief.generation_prompt:
            raise ValueError("Initial brief is missing generation_prompt.")
        if not brief.evaluation_texts:
            raise ValueError("Initial brief is missing evaluation_texts.")
        return brief

    def refine_prompt(
        self,
        user_prompt: str,
        brief: PromptBrief,
        attempt_history: list[AttemptRecord],
    ) -> str:
        skill_id = brief.selected_skill or "NONE"
        skill = self.skill_manager.get_skill(skill_id) if skill_id != "NONE" else None
        skill_instructions = skill["instructions"] if skill else "No skill selected."

        history_chunks = []
        for record in attempt_history:
            evaluation = record.evaluation
            history_chunks.append(
                "\n".join(
                    [
                        f"Round {record.iteration}",
                        f"- Prompt: {record.prompt}",
                        f"- Total score: {evaluation.total_score:.4f}",
                        f"- Validator score: {evaluation.validator_score:.4f}",
                        f"- Hard failures: {evaluation.hard_failures or 'none'}",
                        f"- Protected checks: {evaluation.protected_checks or 'none'}",
                        f"- Verifier summary: {evaluation.verifier_summary or 'none'}",
                        f"- Next guidance: {evaluation.next_prompt_guidance or 'none'}",
                        f"- Check results: {[item.to_dict() for item in evaluation.check_results]}",
                    ]
                )
            )
        history_log = "\n\n".join(history_chunks)

        prompt = REFINE_PROMPT_TEMPLATE.format(
            user_prompt=user_prompt,
            evaluation_texts=brief.evaluation_texts,
            skill_id=skill_id,
            skill_instructions=skill_instructions,
            history_log=history_log,
        )
        return self.llm.complete_text(prompt).strip()

    def run(
        self,
        user_prompt: str,
        iterations: int | None = None,
        target_score: float | None = None,
        force_instrumental: bool = False,
        dry_run: bool = False,
    ) -> dict:
        if iterations is None:
            iterations = self.settings.max_iterations
        if target_score is None:
            target_score = self.settings.target_score

        print(f"\n[Agent] Starting run for: '{user_prompt}'")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = RUNS_ROOT / f"{timestamp}-{slugify(user_prompt)[:48]}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print("[Agent] Routing skill...")
        skill_id = self.route_skill(user_prompt)
        print(f"[Agent] Selected skill: {skill_id}")

        print("[Agent] Building initial brief...")
        brief = self.build_initial_brief(
            user_prompt=user_prompt,
            skill_id=skill_id,
            force_instrumental=force_instrumental,
        )
        print(f"[Agent] Initial prompt ready: {brief.generation_prompt}")
        print("[Agent] Building validation plan...")
        validation_plan = self.evaluator.build_validation_plan(user_prompt, brief)
        print(f"[Agent] Validation checks: {len(validation_plan.checks)}")

        dump_json(
            run_dir / "plan.json",
            {
                "user_prompt": user_prompt,
                "skill_id": skill_id,
                "brief": brief.to_dict(),
                "validation_plan": validation_plan.to_dict(),
                "dry_run": dry_run,
            },
        )

        if dry_run:
            print("[Agent] Dry run completed.")
            return {
                "run_dir": str(run_dir),
                "skill_id": skill_id,
                "brief": brief.to_dict(),
                "validation_plan": validation_plan.to_dict(),
                "attempts": [],
                "best_attempt": None,
            }

        attempt_history: list[AttemptRecord] = []
        best_attempt: AttemptRecord | None = None
        current_brief = replace(brief)

        for iteration in range(1, iterations + 1):
            print(f"\n======== Iteration {iteration}/{iterations} ========")
            attempt_dir = run_dir / f"attempt_{iteration:02d}"
            
            print(f"🚀 [{iteration}] Requesting Generation from MiniMax... (This takes 1-2 minutes)")
            audio_path, _ = self.music_client.generate(current_brief, attempt_dir)
            print(f"✅ [{iteration}] Generation Complete! Saved to {audio_path.name}")
            
            print(f"📊 [{iteration}] Running validator-agent evaluation...")
            evaluation, wav_path = self.evaluator.evaluate(
                audio_path,
                current_brief,
                user_prompt,
                validation_plan,
            )
            print(f"✅ [{iteration}] Evaluation Complete! Score: {evaluation.total_score:.4f}")
            print(f"   Validator score: {evaluation.validator_score:.4f}")
            if evaluation.hard_failures:
                print(f"   Hard failures: {evaluation.hard_failures}")
            if evaluation.verifier_summary:
                print(f"   Verifier: {evaluation.verifier_summary}")

            attempt = AttemptRecord(
                iteration=iteration,
                prompt=current_brief.generation_prompt,
                audio_path=str(audio_path),
                wav_path=str(wav_path),
                evaluation=evaluation,
            )
            attempt_history.append(attempt)
            dump_json(attempt_dir / "evaluation.json", evaluation.to_dict())

            if best_attempt is None or evaluation.total_score > best_attempt.evaluation.total_score:
                print(f"🌟 [{iteration}] New best attempt found!")
                best_attempt = attempt

            if evaluation.total_score >= target_score:
                print(f"\n🎯 [{iteration}] Target score ({target_score}) reached! Stopping early.")
                break
            elif iteration == iterations:
                print(f"\n🎯 Max iterations ({iterations}) hit. Stopping.")
                break

            print(f"🧠 [{iteration}] Target score not met. Refining prompt with Kimi K2.5...")
            refined_prompt = self.refine_prompt(user_prompt, brief, attempt_history)
            print(f"✨ [{iteration}] Refined prompt: {refined_prompt}")
            current_brief = replace(current_brief, generation_prompt=refined_prompt)

        print(f"\n🎉 [Agent] Run completed successfully. Best score: {best_attempt.evaluation.total_score:.4f}")
        summary = {
            "run_dir": str(run_dir),
            "skill_id": skill_id,
            "brief": brief.to_dict(),
            "validation_plan": validation_plan.to_dict(),
            "best_attempt": best_attempt.to_dict() if best_attempt else None,
            "attempts": [item.to_dict() for item in attempt_history],
        }
        dump_json(run_dir / "summary.json", summary)
        return summary
