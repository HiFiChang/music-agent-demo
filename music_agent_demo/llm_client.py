from __future__ import annotations

from openai import OpenAI

from .config import Settings
from .utils import extract_json_payload


class KimiClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.moonshot_api_key,
            base_url=settings.kimi_api_base,
        )

    def complete_text(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.settings.kimi_model,
            messages=messages,
            max_tokens=4096,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return (response.choices[0].message.content or "").strip()

    def complete_json(self, prompt: str, system_prompt: str | None = None) -> dict | list:
        text = self.complete_text(prompt, system_prompt=system_prompt)
        return extract_json_payload(text)

