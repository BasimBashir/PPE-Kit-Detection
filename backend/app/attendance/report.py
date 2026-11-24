"""Outside-time per working day against the person's allowed budget (ADR-016).

Presence intervals are clipped to each local (site time zone) day and to the requested range; an open interval
counts up to "now". Pure functions - the route only queries and serialises.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo


def _local_day_bounds(day: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time.min, tz)
    return start.astimezone(UTC), (start + timedelta(days=1)).astimezone(UTC)


def outside_by_day(intervals: Sequence[tuple[datetime, datetime | None]], first: date, last: date,
                   tz_name: str, allowed_minutes: int, now: datetime | None = None) -> list[dict]:
    """One row per day in [first, last] (site tz), with outside minutes capped by the interval data."""
    tz = ZoneInfo(tz_name)
    now = now or datetime.now(UTC)
    rows = []
    day = first
    while day <= last:
        lo, hi = _local_day_bounds(day, tz)
        seconds = 0.0
        for start, end in intervals:
            overlap = (min(end or now, hi) - max(start, lo)).total_seconds()
            if overlap > 0:
                seconds += overlap
        minutes = round(seconds / 60, 1)
        rows.append({"date": day.isoformat(), "outside_minutes": minutes, "allowed_minutes": allowed_minutes,
                     "over_minutes": round(max(0.0, minutes - allowed_minutes), 1)})
        day += timedelta(days=1)
    return rows
