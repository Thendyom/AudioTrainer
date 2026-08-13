"""Versioned local SQLite practice history."""

from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Iterator
from uuid import uuid4

from audiotrainer.api.schemas import PracticeSession


def default_data_dir() -> Path:
    """Return the per-user AudioTrainer data directory."""

    configured = os.environ.get("AUDIOTRAINER_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    try:
        from platformdirs import user_data_dir

        return Path(user_data_dir("AudioTrainer", appauthor=False))
    except ImportError:  # pragma: no cover - fallback for minimal source checkouts
        return Path.home() / ".local" / "share" / "AudioTrainer"


class SessionRepository:
    """Small thread-safe-by-connection repository for local practice data."""

    SCHEMA_VERSION = 1

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.data_dir / "recordings"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.data_dir / "audiotrainer.sqlite3"
        self._migrate()

    def save(
        self,
        *,
        mode: str,
        duration: float,
        source: str,
        backend: str,
        settings: dict[str, Any],
        result: dict[str, Any],
        feedback: list[dict[str, Any]],
        audio_source: str | Path | None = None,
        retain_audio: bool = False,
    ) -> PracticeSession:
        """Persist one result and optionally copy its recording into managed storage."""

        session_id = str(uuid4())
        created_at = datetime.now(timezone.utc)
        audio_path = None
        if retain_audio and audio_source is not None:
            source_path = Path(audio_source)
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            suffix = source_path.suffix.lower() or ".wav"
            managed = self.audio_dir / f"{session_id}{suffix}"
            shutil.copy2(source_path, managed)
            audio_path = str(managed)
        session = PracticeSession(
            session_id=session_id,
            mode=mode,
            created_at=created_at,
            duration=duration,
            source=source,
            backend=backend,
            settings=settings,
            result=result,
            feedback=feedback,
            audio_path=audio_path,
        )
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO practice_sessions
                (session_id, mode, created_at, duration, source, backend, settings_json, result_json, feedback_json, audio_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session.session_id,
                    session.mode,
                    session.created_at.isoformat(),
                    session.duration,
                    session.source,
                    session.backend,
                    json.dumps(session.settings, ensure_ascii=False),
                    json.dumps(session.result, ensure_ascii=False),
                    json.dumps(session.feedback, ensure_ascii=False),
                    session.audio_path,
                ),
            )
        return session

    def list(
        self,
        *,
        mode: str | None = None,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[PracticeSession]:
        """List newest sessions with optional mode and date filters."""

        query = "SELECT * FROM practice_sessions WHERE 1=1"
        parameters: list[Any] = []
        if mode:
            query += " AND mode = ?"
            parameters.append(mode)
        if since:
            query += " AND created_at >= ?"
            parameters.append(since.isoformat())
        query += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(max(1, min(limit, 10_000)))
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._row_to_session(row) for row in rows]

    def get(self, session_id: str) -> PracticeSession | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM practice_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return self._row_to_session(row) if row else None

    def delete(self, session_id: str) -> bool:
        """Delete one session and its managed retained recording."""

        session = self.get(session_id)
        if session is None:
            return False
        with self._connect() as connection:
            connection.execute("DELETE FROM practice_sessions WHERE session_id = ?", (session_id,))
        self._delete_managed_audio(session.audio_path)
        return True

    def clear(self) -> int:
        """Delete all sessions and their managed recordings."""

        sessions = self.list(limit=10_000)
        with self._connect() as connection:
            connection.execute("DELETE FROM practice_sessions")
        for session in sessions:
            self._delete_managed_audio(session.audio_path)
        return len(sessions)

    def set_setting(self, key: str, value: Any) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO app_settings(key, value_json) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                (key, json.dumps(value, ensure_ascii=False)),
            )

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._connect() as connection:
            row = connection.execute("SELECT value_json FROM app_settings WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def export_json(self) -> str:
        return json.dumps([session.model_dump(mode="json") for session in self.list(limit=10_000)], indent=2)

    def export_csv(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["session_id", "mode", "created_at", "duration", "source", "backend", "audio_retained"],
            )
            writer.writeheader()
            for session in self.list(limit=10_000):
                writer.writerow(
                    {
                        "session_id": session.session_id,
                        "mode": session.mode,
                        "created_at": session.created_at.isoformat(),
                        "duration": session.duration,
                        "source": session.source,
                        "backend": session.backend,
                        "audio_retained": bool(session.audio_path),
                    }
                )
        return output

    def _migrate(self) -> None:
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version < 1:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS practice_sessions (
                        session_id TEXT PRIMARY KEY,
                        mode TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        duration REAL NOT NULL,
                        source TEXT NOT NULL,
                        backend TEXT NOT NULL,
                        settings_json TEXT NOT NULL,
                        result_json TEXT NOT NULL,
                        feedback_json TEXT NOT NULL,
                        audio_path TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_sessions_mode_created
                    ON practice_sessions(mode, created_at DESC);
                    CREATE TABLE IF NOT EXISTS app_settings (
                        key TEXT PRIMARY KEY,
                        value_json TEXT NOT NULL
                    );
                    PRAGMA user_version = 1;
                    """
                )
            if version > self.SCHEMA_VERSION:
                raise RuntimeError("Practice history was created by a newer AudioTrainer version")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _row_to_session(self, row: sqlite3.Row) -> PracticeSession:
        return PracticeSession(
            session_id=row["session_id"],
            mode=row["mode"],
            created_at=datetime.fromisoformat(row["created_at"]),
            duration=row["duration"],
            source=row["source"],
            backend=row["backend"],
            settings=json.loads(row["settings_json"]),
            result=json.loads(row["result_json"]),
            feedback=json.loads(row["feedback_json"]),
            audio_path=row["audio_path"],
        )

    def _delete_managed_audio(self, path: str | None) -> None:
        if not path:
            return
        target = Path(path).resolve()
        managed_root = self.audio_dir.resolve()
        if target.parent == managed_root and target.is_file():
            target.unlink()
