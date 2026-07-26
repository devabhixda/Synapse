import json
from pathlib import Path

from synapse.sources.slack_export import SlackExportSource, sanitize_text


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def create_export(root: Path) -> None:
    write_json(
        root / "users.json",
        [
            {
                "id": "U1",
                "name": "maya",
                "deleted": False,
                "is_bot": False,
                "profile": {"display_name": "Maya", "email": "maya@example.com"},
            },
            {
                "id": "U2",
                "name": "jon",
                "deleted": False,
                "is_bot": False,
                "profile": {"real_name": "Jon Doe"},
            },
            {
                "id": "B1",
                "name": "buildbot",
                "deleted": False,
                "is_bot": True,
                "profile": {"display_name": "Build Bot"},
            },
        ],
    )
    write_json(
        root / "channels.json",
        [{"id": "C1", "name": "product", "is_archived": False}],
    )
    write_json(
        root / "product" / "2026-01-10.json",
        [
            {
                "type": "message",
                "user": "U1",
                "ts": "1768000000.000001",
                "thread_ts": "1768000000.000001",
                "text": (
                    "We decided <@U2> owns the launch blocker. Contact maya@example.com "
                    "or +1 415 555 0100. Details: <https://example.com/private|spec>."
                ),
                "reactions": [{"name": "white_check_mark", "count": 2}],
            },
            {
                "type": "message",
                "user": "U2",
                "ts": "1768000010.000002",
                "thread_ts": "1768000000.000001",
                "text": "Action item accepted. ETA is Friday.",
            },
            {
                "type": "message",
                "user": "U1",
                "ts": "1768000100.000003",
                "text": "Lunch was good.",
            },
            {
                "type": "message",
                "user": "B1",
                "bot_id": "BOT",
                "ts": "1768000200.000004",
                "text": "Build completed.",
            },
        ],
    )


def test_slack_export_bundles_threads_and_selects_work_signal(tmp_path: Path) -> None:
    create_export(tmp_path)
    source = SlackExportSource(tmp_path)

    events = source.load_events(max_events=10, min_score=4, max_per_channel=10)

    assert len(events) == 1
    event = events[0]
    assert event.source_id == "slack-export-product-1768000000-000001"
    assert event.participants == ["Maya", "Jon Doe"]
    assert "Maya: We decided @Jon Doe owns the launch blocker." in event.content
    assert "Jon Doe: Action item accepted. ETA is Friday." in event.content
    assert "[email]" in event.content
    assert "[phone]" in event.content
    assert "spec" in event.content
    assert "example.com" not in event.content
    assert event.metadata["message_count"] == 2
    assert event.metadata["source_adapter"] == "slack-export"


def test_slack_export_preview_contains_only_aggregate_metadata(tmp_path: Path) -> None:
    create_export(tmp_path)
    source = SlackExportSource(tmp_path)

    preview = source.preview(max_events=10, min_score=4, max_per_channel=10)

    assert preview["selected_conversations"] == 1
    assert preview["channels"] == {"product": 1}
    assert "selection_reasons" in preview
    assert "Maya" not in json.dumps(preview)


def test_sanitize_text_removes_secret_tokens(tmp_path: Path) -> None:
    create_export(tmp_path)
    source = SlackExportSource(tmp_path)

    cleaned = sanitize_text("token sk-exampleSecret123 and <@U1>", source.users)

    assert cleaned == "token [secret] and @Maya"
