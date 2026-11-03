"""Durable alert outbox (FR-ALT-008/009): rows are created in the event transaction; delivery is retried with
bounded exponential backoff and ends FAILED after max attempts (observable, never silent)."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config.settings import AlertsConfig
from app.models import AlertOutbox, utcnow


def claim_due(s: Session, limit: int = 50) -> list[AlertOutbox]:
    """Lock due PENDING rows for this transaction (SKIP LOCKED: safe with several dispatchers)."""
    q = (select(AlertOutbox).where(AlertOutbox.status == "PENDING", AlertOutbox.next_attempt_at <= utcnow())
         .order_by(AlertOutbox.id).limit(limit).with_for_update(skip_locked=True))
    return list(s.scalars(q))


def mark_delivered(s: Session, row: AlertOutbox) -> None:
    row.attempts += 1
    row.status, row.delivered_at, row.last_error = "DELIVERED", utcnow(), None


def mark_failed(s: Session, row: AlertOutbox, error: str, cfg: AlertsConfig) -> None:
    row.attempts += 1
    row.last_error = error[:500]
    if row.attempts >= cfg.outbox_max_attempts:
        row.status = "FAILED"
        return
    delay = min(cfg.outbox_backoff_max_s, cfg.outbox_backoff_base_s * 2 ** (row.attempts - 1))
    row.next_attempt_at = utcnow() + timedelta(seconds=delay)


def counts(s: Session) -> dict[str, int]:
    rows = s.execute(select(AlertOutbox.status, func.count()).group_by(AlertOutbox.status))
    return {k: v for k, v in rows}
