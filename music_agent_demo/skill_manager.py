from __future__ import annotations

import re
from pathlib import Path


class SkillManager:
    def __init__(self, skills_dir: Path | None = None):
        self.skills_dir = skills_dir or (Path(__file__).resolve().parent / "skills")
        self.skills = self._load_skills()

    def _load_skills(self) -> dict[str, dict[str, str]]:
        skills_data: dict[str, dict[str, str]] = {}
        if not self.skills_dir.exists():
            return skills_data

        for md_path in sorted(self.skills_dir.glob("*/SKILL.md")):
            skill_id = md_path.parent.name
            content = md_path.read_text(encoding="utf-8")
            desc_match = re.search(r"## Description\n(.*?)\n## ", content, re.DOTALL)
            instr_match = re.search(r"## Instructions\n(.*)", content, re.DOTALL)
            tool_path = md_path.parent / "tool.py"
            skills_data[skill_id] = {
                "id": skill_id,
                "path": str(md_path),
                "tool_path": str(tool_path) if tool_path.exists() else "",
                "description": desc_match.group(1).strip() if desc_match else "",
                "instructions": instr_match.group(1).strip() if instr_match else "",
            }
        return skills_data

    def get_skill_manifest(self) -> str:
        lines = []
        for skill_id, data in self.skills.items():
            lines.append(f"- SKILL_ID: {skill_id}\n  DESCRIPTION: {data['description']}")
        return "\n".join(lines)

    def get_skill(self, skill_id: str) -> dict[str, str] | None:
        return self.skills.get(skill_id)
