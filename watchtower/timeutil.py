"""All times leave the API as ISO-8601 with an explicit offset."""
from __future__ import annotations

import re
from datetime import datetime, timezone


def now() -> datetime:
    """Current UTC time with microsecond precision (two changes in the same second stay distinguishable)."""
    return datetime.now(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()  # fractional seconds are kept (omitted by isoformat only when exactly zero)


def parse(value: str | None) -> datetime | None:
    """Parse an ISO datetime; naive values are treated as UTC. Raises ValueError if invalid."""
    if value is None:
        return None
    v = value.strip().replace("Z", "+00:00")
    # A '+' in a query string may arrive decoded as a space.
    v = re.sub(r"(T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?) (\d{2}:\d{2})$", r"\1+\2", v)  # also with fractional seconds
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
