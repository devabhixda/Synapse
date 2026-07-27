import argparse
import asyncio
import json
from pathlib import Path

from synapse.db.session import engine, session_factory
from synapse.episodic.service import SystemOneService
from synapse.inference.assertions import LLMAssertionInterpreter
from synapse.inference.chat import InferenceError, LLMInterpreter
from synapse.memory.mem0_store import Mem0EpisodeStore
from synapse.sources.slack_export import SlackExportSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a local Slack export into Synapse")
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument(
        "--group-id",
        help="Account these messages belong to. One export maps to one account. "
        "Required unless --dry-run, which writes nothing.",
    )
    parser.add_argument("--channel", action="append", dest="channels")
    parser.add_argument("--max-events", type=int, default=40)
    parser.add_argument("--max-per-channel", type=int, default=30)
    parser.add_argument("--min-score", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-assertions", action="store_true")
    return parser.parse_args()


async def import_events(args: argparse.Namespace) -> dict[str, object]:
    source = SlackExportSource(args.export_root)
    channels = set(args.channels) if args.channels else None
    if args.dry_run:
        return source.preview(
            channels=channels,
            max_events=args.max_events,
            min_score=args.min_score,
            max_per_channel=args.max_per_channel,
        )

    events = source.load_events(
        channels=channels,
        max_events=args.max_events,
        min_score=args.min_score,
        max_per_channel=args.max_per_channel,
    )
    system_one = SystemOneService(LLMInterpreter(), Mem0EpisodeStore())
    assertion_interpreter = LLMAssertionInterpreter()
    processed = 0
    skipped = 0
    indexed = 0
    assertion_failures = 0
    observation_failures = 0

    async with session_factory() as session:
        for event in events:
            if not args.skip_assertions:
                try:
                    assertions = await assertion_interpreter.extract(event)
                    event.metadata["semantic_assertions"] = assertions.model_dump(mode="json")
                except InferenceError:
                    assertion_failures += 1
            try:
                episode, created = await system_one.process(event, args.group_id, session)
            except InferenceError:
                observation_failures += 1
                continue
            processed += int(created)
            skipped += int(not created)
            indexed += int(episode.mem0_id is not None)

    return {
        "selected": len(events),
        "processed": processed,
        "skipped": skipped,
        "indexed": indexed,
        "assertion_failures": assertion_failures,
        "observation_failures": observation_failures,
    }


async def async_main() -> None:
    args = parse_args()
    if not args.dry_run and not args.group_id:
        raise ValueError("--group-id is required when importing")
    if args.max_events < 1:
        raise ValueError("--max-events must be at least 1")
    if args.min_score < 0:
        raise ValueError("--min-score cannot be negative")
    if args.max_per_channel < 1:
        raise ValueError("--max-per-channel must be at least 1")
    try:
        result = await import_events(args)
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        await engine.dispose()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
