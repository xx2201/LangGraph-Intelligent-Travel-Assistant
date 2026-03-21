from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from ..core.config import THREAD_STORE_DB_PATH


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def default_thread_title() -> str:
    """Return the default title used before a thread receives its first message."""

    return "新会话"


def shorten_title(text: str, limit: int = 24) -> str:
    """Build a short thread title from the first user message."""

    stripped = text.strip()
    if not stripped:
        return default_thread_title()
    return stripped[:limit]


@dataclass
class ThreadRecord:
    """Structured thread metadata stored in the backend SQLite table."""

    thread_id: str
    title: str
    created_at: str
    updated_at: str
    last_user_message: str
    last_assistant_message: str

    def to_dict(self) -> dict[str, str]:
        """Convert the record into a JSON-friendly dictionary."""

        return {
            "thread_id": self.thread_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_user_message": self.last_user_message,
            "last_assistant_message": self.last_assistant_message,
        }


async def connect_thread_store(db_path: str = THREAD_STORE_DB_PATH) -> aiosqlite.Connection:
    """Open the SQLite database used for backend thread metadata."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(path)
    connection.row_factory = aiosqlite.Row
    return connection


async def initialize_thread_store(connection: aiosqlite.Connection) -> None:
    """Create the thread metadata table if it does not already exist."""

    await connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_threads (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_user_message TEXT NOT NULL DEFAULT '',
            last_assistant_message TEXT NOT NULL DEFAULT ''
        )
        """
    )
    await connection.commit()


def row_to_thread(row: aiosqlite.Row) -> ThreadRecord:
    """Convert one SQLite row into a ``ThreadRecord`` object."""

    return ThreadRecord(
        thread_id=row["thread_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_user_message=row["last_user_message"],
        last_assistant_message=row["last_assistant_message"],
    )


async def create_thread(
    connection: aiosqlite.Connection,
    thread_id: str,
    title: str | None = None,
) -> ThreadRecord:
    """Create a new backend thread record."""

    now = utc_now_iso()
    thread_title = title or default_thread_title()
    await connection.execute(
        """
        INSERT INTO chat_threads (
            thread_id, title, created_at, updated_at, last_user_message, last_assistant_message
        )
        VALUES (?, ?, ?, ?, '', '')
        """,
        (thread_id, thread_title, now, now),
    )
    await connection.commit()
    return ThreadRecord(
        thread_id=thread_id,
        title=thread_title,
        created_at=now,
        updated_at=now,
        last_user_message="",
        last_assistant_message="",
    )


async def get_thread(
    connection: aiosqlite.Connection,
    thread_id: str,
) -> ThreadRecord | None:
    """Return thread metadata by ``thread_id`` or ``None`` if it does not exist."""

    cursor = await connection.execute(
        """
        SELECT thread_id, title, created_at, updated_at, last_user_message, last_assistant_message
        FROM chat_threads
        WHERE thread_id = ?
        """,
        (thread_id,),
    )
    row = await cursor.fetchone()
    return row_to_thread(row) if row else None


async def list_threads(connection: aiosqlite.Connection) -> list[ThreadRecord]:
    """Return all thread metadata ordered by most recently updated first."""

    cursor = await connection.execute(
        """
        SELECT thread_id, title, created_at, updated_at, last_user_message, last_assistant_message
        FROM chat_threads
        ORDER BY updated_at DESC
        """
    )
    rows = await cursor.fetchall()
    return [row_to_thread(row) for row in rows]


async def ensure_thread(
    connection: aiosqlite.Connection,
    thread_id: str,
) -> ThreadRecord:
    """Return an existing thread or create a new empty one if missing."""

    existing = await get_thread(connection, thread_id)
    if existing is not None:
        return existing
    return await create_thread(connection, thread_id)


async def update_thread_after_chat(
    connection: aiosqlite.Connection,
    thread_id: str,
    user_message: str,
    assistant_message: str,
) -> ThreadRecord:
    """Update thread summary fields after one successful chat turn."""

    existing = await ensure_thread(connection, thread_id)
    title = existing.title
    if title == default_thread_title() and user_message.strip():
        title = shorten_title(user_message)

    updated_at = utc_now_iso()
    await connection.execute(
        """
        UPDATE chat_threads
        SET title = ?, updated_at = ?, last_user_message = ?, last_assistant_message = ?
        WHERE thread_id = ?
        """,
        (title, updated_at, user_message, assistant_message, thread_id),
    )
    await connection.commit()
    return ThreadRecord(
        thread_id=thread_id,
        title=title,
        created_at=existing.created_at,
        updated_at=updated_at,
        last_user_message=user_message,
        last_assistant_message=assistant_message,
    )
