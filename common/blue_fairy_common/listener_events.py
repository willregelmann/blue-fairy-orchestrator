# common/blue_fairy_common/listener_events.py
"""Listener event protocol types.

Listeners emit newline-delimited JSON events to stdout.
Each event has: type, source, payload.
"""
import json
from typing import Any
from pydantic import BaseModel, Field


class ListenerEvent(BaseModel):
    """Base class for all listener events.

    Attributes:
        type: Event type (e.g., "message", "invite", "heartbeat")
        source: Plugin name that produced this event
        payload: Event-specific data
    """
    type: str
    source: str
    payload: dict[str, Any] = Field(default_factory=dict)


class MessageEvent(ListenerEvent):
    """A chat message event.

    Payload should contain:
        - room: Room identifier
        - sender: User who sent the message
        - content: Message text
        - event_id: Unique event identifier
    """
    type: str = "message"


class InviteEvent(ListenerEvent):
    """A room invite event.

    Payload should contain:
        - room: Room identifier
        - inviter: User who sent the invite
    """
    type: str = "invite"


class HeartbeatEvent(ListenerEvent):
    """A health check heartbeat.

    Emitted periodically to indicate listener is alive.
    """
    type: str = "heartbeat"


# Registry of event types for parsing
_EVENT_TYPES: dict[str, type[ListenerEvent]] = {
    "message": MessageEvent,
    "invite": InviteEvent,
    "heartbeat": HeartbeatEvent,
}


def parse_listener_event(json_str: str) -> ListenerEvent:
    """Parse a JSON string into a ListenerEvent.

    Args:
        json_str: Newline-delimited JSON from listener stdout

    Returns:
        Parsed event object

    Raises:
        ValueError: If JSON is invalid or missing required fields
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if "type" not in data:
        raise ValueError("Missing required field: type")
    if "source" not in data:
        raise ValueError("Missing required field: source")

    event_type = data.get("type")
    event_class = _EVENT_TYPES.get(event_type, ListenerEvent)

    return event_class(**data)
