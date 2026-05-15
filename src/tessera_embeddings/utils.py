"""Shared utility functions used across the src package."""

from datetime import UTC, datetime


def utcnow_iso() -> str:
    """Get current UTC time as ISO format string."""
    return datetime.now(UTC).isoformat()
