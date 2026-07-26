import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synapse.db.models import SourceType
from synapse.domain.models import SourceEvent

SIGNALS: dict[str, tuple[int, tuple[str, ...]]] = {
    "decision": (8, ("decid", "agreed", "approved", "final decision", "we will")),
    "blocker": (7, ("block", "broken", "failing", "issue", "bug")),
    "owner_action": (6, ("owner", "action item", "todo", "to-do", "assigned")),
    "deadline": (5, ("deadline", "by friday", "by monday", "eta", "eod")),
    "status": (4, ("update", "progress", "done", "completed", "shipped")),
    "question": (1, ("?",)),
}

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
SECRET_PATTERN = re.compile(r"\b(?:sk-|xox[baprs]-|ghp_)[A-Za-z0-9_-]+\b")
USER_MENTION_PATTERN = re.compile(r"<@([A-Z0-9]+)>")
CHANNEL_MENTION_PATTERN = re.compile(r"<#[A-Z0-9]+\|([^>]+)>")
LINK_PATTERN = re.compile(r"<https?://[^>|]+(?:\|([^>]+))?>")


class SlackExportError(ValueError):
    pass


@dataclass(frozen=True)
class SlackUser:
    user_id: str
    name: str
    is_bot: bool
    is_deleted: bool


@dataclass(frozen=True)
class SlackMessage:
    channel: str
    timestamp: str
    thread_timestamp: str
    user_id: str
    text: str
    reaction_count: int
    file_count: int
    edited: bool


@dataclass(frozen=True)
class RankedConversation:
    event: SourceEvent
    score: int
    reasons: tuple[str, ...]


class SlackExportSource:
    def __init__(self, export_root: Path) -> None:
        self.export_root = export_root
        self.users = self._load_users()
        self.channel_names = self._load_channel_names()

    def load_events(
        self,
        *,
        channels: set[str] | None = None,
        max_events: int = 40,
        min_score: int = 4,
        max_per_channel: int | None = None,
    ) -> list[SourceEvent]:
        ranked = self.rank_conversations(channels=channels)
        selected = select_ranked_conversations(
            ranked,
            max_events=max_events,
            min_score=min_score,
            max_per_channel=max_per_channel,
        )
        return sorted(
            (item.event for item in selected),
            key=lambda event: (event.occurred_at, event.source_id),
        )

    def rank_conversations(
        self, *, channels: set[str] | None = None
    ) -> list[RankedConversation]:
        selected_channels = channels or self.channel_names
        unknown_channels = selected_channels - self.channel_names
        if unknown_channels:
            names = ", ".join(sorted(unknown_channels))
            raise SlackExportError(f"Unknown Slack export channels: {names}")

        conversations: dict[tuple[str, str], list[SlackMessage]] = defaultdict(list)
        seen_messages: set[tuple[str, str]] = set()
        for channel in sorted(selected_channels):
            for message in self._load_channel_messages(channel):
                identity = (channel, message.timestamp)
                if identity in seen_messages:
                    continue
                seen_messages.add(identity)
                conversations[(channel, message.thread_timestamp)].append(message)

        ranked = [
            self._rank_conversation(channel, thread_timestamp, messages)
            for (channel, thread_timestamp), messages in conversations.items()
        ]
        return sorted(
            ranked,
            key=lambda item: (-item.score, item.event.occurred_at, item.event.source_id),
        )

    def preview(
        self,
        *,
        channels: set[str] | None = None,
        max_events: int = 40,
        min_score: int = 4,
        max_per_channel: int | None = None,
    ) -> dict[str, Any]:
        ranked = self.rank_conversations(channels=channels)
        selected = select_ranked_conversations(
            ranked,
            max_events=max_events,
            min_score=min_score,
            max_per_channel=max_per_channel,
        )
        channel_counts = Counter(str(item.event.metadata["channel"]) for item in selected)
        reason_counts = Counter(reason for item in selected for reason in item.reasons)
        dates = [item.event.occurred_at for item in selected]
        return {
            "available_conversations": len(ranked),
            "selected_conversations": len(selected),
            "channels": dict(sorted(channel_counts.items())),
            "selection_reasons": dict(sorted(reason_counts.items())),
            "score_range": (
                [min(item.score for item in selected), max(item.score for item in selected)]
                if selected
                else []
            ),
            "date_range": (
                [min(dates).date().isoformat(), max(dates).date().isoformat()]
                if dates
                else []
            ),
        }

    def _load_users(self) -> dict[str, SlackUser]:
        payload = self._load_json_list(self.export_root / "users.json")
        users: dict[str, SlackUser] = {}
        for item in payload:
            user_id = string_value(item.get("id"))
            if not user_id:
                continue
            profile = item.get("profile")
            profile_data = profile if isinstance(profile, dict) else {}
            name = first_nonempty(
                profile_data.get("display_name"),
                profile_data.get("real_name"),
                item.get("real_name"),
                item.get("name"),
                user_id,
            )
            users[user_id] = SlackUser(
                user_id=user_id,
                name=name,
                is_bot=bool(item.get("is_bot")),
                is_deleted=bool(item.get("deleted")),
            )
        return users

    def _load_channel_names(self) -> set[str]:
        payload = self._load_json_list(self.export_root / "channels.json")
        return {
            name
            for item in payload
            if (name := string_value(item.get("name"))) and not item.get("is_archived")
        }

    def _load_channel_messages(self, channel: str) -> list[SlackMessage]:
        channel_path = self.export_root / channel
        if not channel_path.is_dir():
            return []
        messages: list[SlackMessage] = []
        for path in sorted(channel_path.glob("*.json")):
            if path.name.endswith("_file_shares.json"):
                continue
            for item in self._load_json_list(path):
                message = self._parse_message(channel, item)
                if message is not None:
                    messages.append(message)
        return messages

    def _parse_message(self, channel: str, item: dict[str, Any]) -> SlackMessage | None:
        if item.get("type") != "message":
            return None
        if item.get("subtype") in {"channel_join", "channel_leave"}:
            return None
        user_id = string_value(item.get("user"))
        user = self.users.get(user_id)
        if user is None or user.is_bot or user.is_deleted or item.get("bot_id"):
            return None
        timestamp = string_value(item.get("ts"))
        if not timestamp:
            return None
        raw_text = string_value(item.get("text")).strip()
        if not raw_text:
            return None
        text = sanitize_text(raw_text, self.users)
        if not text:
            return None
        reactions = item.get("reactions")
        reaction_count = sum(
            int(reaction.get("count") or 0)
            for reaction in reactions
            if isinstance(reaction, dict)
        ) if isinstance(reactions, list) else 0
        files = item.get("files")
        attachments = item.get("attachments")
        file_count = (
            len(files) if isinstance(files, list) else 0
        ) + (
            len(attachments) if isinstance(attachments, list) else 0
        )
        return SlackMessage(
            channel=channel,
            timestamp=timestamp,
            thread_timestamp=string_value(item.get("thread_ts")) or timestamp,
            user_id=user_id,
            text=text,
            reaction_count=reaction_count,
            file_count=file_count,
            edited=isinstance(item.get("edited"), dict),
        )

    def _rank_conversation(
        self,
        channel: str,
        thread_timestamp: str,
        messages: list[SlackMessage],
    ) -> RankedConversation:
        ordered = sorted(messages, key=lambda message: float(message.timestamp))
        combined_text = "\n".join(message.text for message in ordered).casefold()
        reasons: list[str] = []
        score = 0
        for reason, (weight, markers) in SIGNALS.items():
            if any(marker in combined_text for marker in markers):
                reasons.append(reason)
                score += weight
        reply_count = max(0, len(ordered) - 1)
        reaction_count = sum(message.reaction_count for message in ordered)
        file_count = sum(message.file_count for message in ordered)
        edited_count = sum(message.edited for message in ordered)
        if reply_count:
            reasons.append("thread")
            score += min(reply_count, 3)
        if reaction_count:
            reasons.append("reaction")
            score += min(reaction_count, 2)
        if file_count:
            reasons.append("attachment")
            score += 1
        if edited_count:
            reasons.append("edited")
            score += 1

        participants = list(
            dict.fromkeys(self.users[message.user_id].name for message in ordered)
        )
        lines = [
            f"{self.users[message.user_id].name}: {message.text[:1200]}"
            for message in ordered
        ]
        content = "\n".join(lines)
        if len(content) > 6000:
            content = f"{content[:5997]}..."
        occurred_at = datetime.fromtimestamp(float(ordered[0].timestamp), tz=UTC)
        stable_timestamp = thread_timestamp.replace(".", "-")
        event = SourceEvent(
            source_id=f"slack-export-{channel}-{stable_timestamp}",
            source_type=SourceType.SLACK,
            conversation_id=f"slack/{channel}/{thread_timestamp}",
            content=content,
            participants=participants,
            occurred_at=occurred_at,
            metadata={
                "channel": channel,
                "thread_ts": thread_timestamp,
                "message_count": len(ordered),
                "reply_count": reply_count,
                "reaction_count": reaction_count,
                "file_count": file_count,
                "edited_count": edited_count,
                "selection_score": score,
                "selection_reasons": reasons,
                "source_adapter": "slack-export",
            },
        )
        return RankedConversation(event=event, score=score, reasons=tuple(reasons))

    @staticmethod
    def _load_json_list(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise SlackExportError(f"Slack export file not found: {path.name}")
        try:
            payload: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SlackExportError(f"Invalid Slack export JSON: {path.name}") from error
        if not isinstance(payload, list):
            raise SlackExportError(f"Expected a JSON list in Slack export file: {path.name}")
        return [item for item in payload if isinstance(item, dict)]


def sanitize_text(value: str, users: dict[str, SlackUser]) -> str:
    def replace_user(match: re.Match[str]) -> str:
        user = users.get(match.group(1))
        return f"@{user.name}" if user is not None else "@unknown"

    cleaned = USER_MENTION_PATTERN.sub(replace_user, value)
    cleaned = CHANNEL_MENTION_PATTERN.sub(r"#\1", cleaned)
    cleaned = LINK_PATTERN.sub(lambda match: match.group(1) or "[link]", cleaned)
    cleaned = EMAIL_PATTERN.sub("[email]", cleaned)
    cleaned = PHONE_PATTERN.sub("[phone]", cleaned)
    cleaned = SECRET_PATTERN.sub("[secret]", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def select_ranked_conversations(
    ranked: list[RankedConversation],
    *,
    max_events: int,
    min_score: int,
    max_per_channel: int | None,
) -> list[RankedConversation]:
    selected: list[RankedConversation] = []
    channel_counts: Counter[str] = Counter()
    for item in ranked:
        if item.score < min_score:
            continue
        channel = str(item.event.metadata["channel"])
        if max_per_channel is not None and channel_counts[channel] >= max_per_channel:
            continue
        selected.append(item)
        channel_counts[channel] += 1
        if len(selected) >= max_events:
            break
    return selected


def string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


def first_nonempty(*values: object) -> str:
    return next((value.strip() for value in values if isinstance(value, str) and value.strip()), "")
