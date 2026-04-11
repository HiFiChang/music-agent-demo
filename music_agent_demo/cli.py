from __future__ import annotations

import argparse
import json
import sys

from .agent import MusicGenerationAgent
from .config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal music generation agent demo.")
    parser.add_argument("query", help="User music request.")
    parser.add_argument("--iterations", type=int, default=None, help="Maximum optimization rounds.")
    parser.add_argument("--target-score", type=float, default=None, help="Stop once score reaches this value.")
    parser.add_argument("--instrumental", action="store_true", help="Force instrumental generation.")
    parser.add_argument("--dry-run", action="store_true", help="Only plan and route skills, do not call MiniMax.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = Settings.load()
    if not args.dry_run:
        settings.validate_for_live_run()
    elif not settings.moonshot_api_key:
        raise ValueError("MOONSHOT_API_KEY is required even for --dry-run because planning uses Kimi.")

    agent = MusicGenerationAgent(settings)
    result = agent.run(
        user_prompt=args.query,
        iterations=args.iterations,
        target_score=args.target_score,
        force_instrumental=args.instrumental,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0)

if __name__ == "__main__":
    main()

