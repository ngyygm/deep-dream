#!/usr/bin/env python3
"""
Chat Session Manager — manages claude CLI processes for multi-session chat.

Each session launches a claude subprocess with stream-json I/O.
Messages are sent via stdin (NDJSON), responses read from stdout (NDJSON).
Session metadata persisted in SQLite.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "chat_sessions.db"
_DEFAULT_CLI = os.path.expanduser("~/.local/bin/claude")
_IDLE_TIMEOUT_S = 1800  # 30 minutes
_MAX_PROCESSES = 10
_REAPER_INTERVAL_S = 300  # 5 minutes


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    claude_session_id TEXT,
    graph_id TEXT NOT NULL DEFAULT 'default',
    title TEXT,
    created_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
)
"""


def _get_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE_SQL)
    # Migrate: rename dream_session_id -> claude_session_id if needed
    cols = [row[1] for row in conn.execute("PRAGMA table_info(chat_sessions)").fetchall()]
    if "dream_session_id" in cols and "claude_session_id" not in cols:
        conn.execute("ALTER TABLE chat_sessions RENAME COLUMN dream_session_id TO claude_session_id")
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Runtime session
# ---------------------------------------------------------------------------

_STREAM_SENTINEL = "__STREAM_DONE__"


@dataclass(slots=True)
class ChatSession:
    session_id: str
    graph_id: str
    title: str
    process: Optional[subprocess.Popen] = None
    stdin_lock: threading.Lock = field(default_factory=threading.Lock)
    event_queue: queue.Queue = field(default_factory=queue.Queue)
    stdout_thread: Optional[threading.Thread] = None
    created_at: str = ""
    last_active: float = field(default_factory=time.time)
    status: str = "active"
    claude_session_id: str = ""


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

class SessionManager:
    def __init__(
        self,
        db_path: str | Path | None = None,
        project_root: str | Path | None = None,
        idle_timeout: int = _IDLE_TIMEOUT_S,
        max_processes: int = _MAX_PROCESSES,
    ):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.project_root = Path(project_root) if project_root else _PROJECT_ROOT
        self.idle_timeout = idle_timeout
        self.max_processes = max_processes

        self._sessions: dict[str, ChatSession] = {}
        self._lock = threading.RLock()
        self._db = _get_db(self.db_path)
        self._reaper_stop = threading.Event()
        self._reaper_thread: threading.Thread | None = None
        self._cli = os.environ.get("CLAUDE_CLI", _DEFAULT_CLI)

        self._restore_sessions()

    # -- Public API --------------------------------------------------------

    def start(self):
        """Start the idle reaper background thread."""
        if self._reaper_thread is None or not self._reaper_thread.is_alive():
            self._reaper_thread = threading.Thread(target=self._idle_reaper, daemon=True)
            self._reaper_thread.start()

    def shutdown(self):
        """Stop reaper and terminate all processes."""
        self._reaper_stop.set()
        with self._lock:
            for sid, s in self._sessions.items():
                self._terminate_process(s)
        if self._db:
            self._db.close()

    def create_session(
        self,
        graph_id: str = "default",
        title: str | None = None,
    ) -> dict:
        """Create a new chat session and launch claude process."""
        sid = uuid.uuid4().hex[:16]
        claude_sid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        title = title or f"Session {sid[:8]}"

        # Persist to DB
        self._db.execute(
            "INSERT INTO chat_sessions (session_id, claude_session_id, graph_id, title, created_at, last_active_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, 'active')",
            (sid, claude_sid, graph_id, title, now, now),
        )
        self._db.commit()

        # Create runtime session + launch process
        session = ChatSession(
            session_id=sid,
            graph_id=graph_id,
            title=title,
            created_at=now,
            last_active=time.time(),
            status="active",
            claude_session_id=claude_sid,
        )

        with self._lock:
            self._evict_if_needed()
            self._launch_process(session)
            self._sessions[sid] = session

        return self._session_to_dict(session)

    def get_session(self, sid: str) -> dict | None:
        """Get session details (from memory or DB)."""
        with self._lock:
            session = self._sessions.get(sid)
            if session:
                return self._session_to_dict(session)

        # Check DB
        row = self._db.execute(
            "SELECT * FROM chat_sessions WHERE session_id = ?", (sid,)
        ).fetchone()
        if not row:
            return None
        return dict(row)

    def list_sessions(self, include_closed: bool = False) -> list[dict]:
        """List all sessions."""
        results = []
        with self._lock:
            for s in self._sessions.values():
                if not include_closed and s.status == "closed":
                    continue
                results.append(self._session_to_dict(s))

        # Add DB-only sessions (not yet loaded into memory)
        loaded_ids = set(self._sessions.keys())
        rows = self._db.execute("SELECT * FROM chat_sessions").fetchall()
        for row in rows:
            if row["session_id"] not in loaded_ids:
                if not include_closed and row["status"] == "closed":
                    continue
                results.append(dict(row))

        return results

    def close_session(self, sid: str) -> bool:
        """Close session — terminate process, keep DB record."""
        with self._lock:
            session = self._sessions.get(sid)
            if not session:
                return False
            self._terminate_process(session)
            session.status = "closed"

        self._db.execute(
            "UPDATE chat_sessions SET status = 'closed', last_active_at = ? WHERE session_id = ?",
            (datetime.now(timezone.utc).isoformat(), sid),
        )
        self._db.commit()
        return True

    def delete_session(self, sid: str) -> bool:
        """Delete session — terminate process, remove DB record."""
        with self._lock:
            session = self._sessions.pop(sid, None)
            if session:
                self._terminate_process(session)

        self._db.execute("DELETE FROM chat_sessions WHERE session_id = ?", (sid,))
        self._db.commit()
        return True

    def update_session(self, sid: str, **kwargs) -> bool:
        """Update session metadata (graph_id, title)."""
        allowed = {"graph_id", "title"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False

        with self._lock:
            session = self._sessions.get(sid)
            if session:
                for k, v in updates.items():
                    setattr(session, k, v)

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [datetime.now(timezone.utc).isoformat(), sid]
        self._db.execute(
            f"UPDATE chat_sessions SET {set_clause}, last_active_at = ? WHERE session_id = ?",
            values,
        )
        self._db.commit()
        return True

    def send_message(
        self,
        sid: str,
        message: str,
        attachments: list[dict] | None = None,
    ) -> queue.Queue | None:
        """Send a message to a session's claude process.

        Returns a queue that yields SSE event strings, or None if session not found.
        The caller should drain the queue until _STREAM_SENTINEL is received.
        """
        with self._lock:
            session = self._sessions.get(sid)
            if not session or session.status == "closed":
                return None

            # Ensure process is running (lazy resume)
            self._ensure_process(session)

            if not session.process or session.process.poll() is not None:
                return None

            session.last_active = time.time()

        # Build content
        content_parts = []
        if attachments:
            for att in attachments:
                content_parts.append({
                    "type": "text",
                    "text": f"[Attachment: {att.get('filename', 'file')}]\n{att.get('content', '')}",
                })
        content_parts.append({"type": "text", "text": message})

        # Create response queue for this request
        resp_queue: queue.Queue = queue.Queue()
        session.event_queue = resp_queue  # redirect events to this requestor

        # Send via stdin (NDJSON)
        user_msg = {
            "type": "user",
            "session_id": session.claude_session_id,
            "message": {
                "role": "user",
                "content": content_parts if len(content_parts) > 1 else message,
            },
            "parent_tool_use_id": None,
        }

        with session.stdin_lock:
            try:
                session.process.stdin.write(json.dumps(user_msg).encode() + b"\n")
                session.process.stdin.flush()
            except (BrokenPipeError, OSError):
                resp_queue.put(_STREAM_SENTINEL)
                return resp_queue

        return resp_queue

    def get_event_sentinel(self) -> str:
        return _STREAM_SENTINEL

    # -- Process management ------------------------------------------------

    def _build_env(self, session: ChatSession) -> dict:
        """Build environment variables for claude CLI process."""
        env = os.environ.copy()
        # Only pass graph ID so MCP tools know which graph to target
        env["DEEP_DREAM_GRAPH_ID"] = session.graph_id
        return env

    def _launch_process(self, session: ChatSession, resume: bool = False):
        """Launch claude CLI process for a session."""
        env = self._build_env(session)

        mcp_config = str(self.project_root / ".mcp.json")

        args = [
            self._cli,
            "--print",
            "--output-format", "stream-json",
            "--input-format", "stream-json",
            "--include-partial-messages",
            "--mcp-config", mcp_config,
            "--bare",
            "--dangerously-skip-permissions",
        ]

        if resume and session.claude_session_id:
            args.extend(["--resume", session.claude_session_id])
        else:
            args.extend(["--session-id", session.claude_session_id])

        try:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=str(self.project_root),
            )
        except FileNotFoundError:
            raise RuntimeError(f"claude CLI not found: {self._cli}")

        session.process = proc

        # Start stdout reader thread
        t = threading.Thread(
            target=self._read_stdout,
            args=(session,),
            daemon=True,
            name=f"claude-stdout-{session.session_id[:8]}",
        )
        t.start()
        session.stdout_thread = t

        # Start stderr reader thread (logs)
        t2 = threading.Thread(
            target=self._read_stderr,
            args=(session,),
            daemon=True,
            name=f"claude-stderr-{session.session_id[:8]}",
        )
        t2.start()

    def _ensure_process(self, session: ChatSession):
        """Ensure the session's claude process is running. Resume if needed."""
        if session.process and session.process.poll() is None:
            return  # Already running

        # Process dead or never started — launch with --resume
        self._launch_process(session, resume=True)

    def _terminate_process(self, session: ChatSession):
        """Terminate a session's claude process."""
        if session.process:
            try:
                session.process.terminate()
                session.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                session.process.kill()
                session.process.wait(timeout=2)
            except Exception as _e:
                logging.getLogger(__name__).warning("终止 claude 进程失败: %s", _e)
            session.process = None

    def _read_stdout(self, session: ChatSession):
        """Read NDJSON from claude stdout and forward to event queue."""
        proc = session.process
        if not proc or not proc.stdout:
            return

        try:
            for line in proc.stdout:
                if not line:
                    continue
                try:
                    line_str = line.decode().strip()
                except UnicodeDecodeError:
                    continue

                if not line_str:
                    continue

                try:
                    event = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                # Convert to SSE event string
                sse = self._event_to_sse(event)
                if sse is _STREAM_SENTINEL:
                    session.event_queue.put(_STREAM_SENTINEL)
                    continue
                if sse:
                    session.event_queue.put(sse)

        except (ValueError, OSError):
            pass
        finally:
            session.event_queue.put(_STREAM_SENTINEL)

    def _read_stderr(self, session: ChatSession):
        """Read stderr from claude for logging."""
        proc = session.process
        if not proc or not proc.stderr:
            return
        try:
            for line in proc.stderr:
                try:
                    msg = line.decode().strip()
                    if msg:
                        sys.stderr.write(f"[claude:{session.session_id[:8]}] {msg}\n")
                except UnicodeDecodeError:
                    pass
        except (ValueError, OSError):
            pass

    def _event_to_sse(self, event: dict) -> str | None:
        """Convert a claude stream-json event to an SSE event string.

        Returns:
            str: SSE event string to forward to client
            None: skip this event (e.g. system init, internal events)
            _STREAM_SENTINEL: signals end of response turn (do NOT send to client)
        """
        event_type = event.get("type", "")

        # Hide system init events from user
        if event_type == "system" and event.get("subtype") == "init":
            return None

        # --- Streaming token deltas (from --include-partial-messages) ---
        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                if text:
                    return f"event: text\ndata: {json.dumps({'content': text}, ensure_ascii=False)}\n\n"
            return None

        if event_type == "content_block_start":
            # New content block starting — if tool_use, notify frontend
            cb = event.get("content_block", {})
            if isinstance(cb, dict) and cb.get("type") == "tool_use":
                tool_name = cb.get("name", "unknown")
                tool_input = cb.get("input", {})
                return f"event: tool_call\ndata: {json.dumps({'tool': tool_name, 'args': tool_input}, ensure_ascii=False)}\n\n"
            return None

        if event_type == "content_block_stop":
            return None

        # --- Full assistant message (skip if we already streamed deltas) ---
        if event_type == "assistant":
            # When partial messages are enabled, deltas already streamed — skip
            return None

        elif event_type == "tool_use":
            tool_name = event.get("tool_name", event.get("name", "unknown"))
            tool_input = event.get("input", event.get("args", {}))
            return f"event: tool_call\ndata: {json.dumps({'tool': tool_name, 'args': tool_input}, ensure_ascii=False)}\n\n"

        elif event_type == "tool_result":
            tool_name = event.get("tool_name", "")
            result = event.get("result", event.get("content", ""))
            is_error = event.get("is_error", False)
            return f"event: tool_result\ndata: {json.dumps({'tool': tool_name, 'result': result, 'is_error': is_error}, ensure_ascii=False)}\n\n"

        elif event_type == "result":
            # Final result of a turn — signal end of stream
            return _STREAM_SENTINEL

        elif event_type == "error":
            error_msg = event.get("error", event.get("message", "Unknown error"))
            return f"event: error\ndata: {json.dumps({'error': str(error_msg)}, ensure_ascii=False)}\n\n"

        elif event_type == "system":
            return f"event: system\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

        # Skip other internal events
        return None

    # -- Maintenance -------------------------------------------------------

    def _idle_reaper(self):
        """Background thread: terminate idle processes."""
        while not self._reaper_stop.wait(_REAPER_INTERVAL_S):
            with self._lock:
                now = time.time()
                for sid, session in list(self._sessions.items()):
                    if session.status != "active":
                        continue
                    if session.process and session.process.poll() is None:
                        idle_time = now - session.last_active
                        if idle_time > self.idle_timeout:
                            sys.stderr.write(
                                f"[chat-session] Reaping idle session {sid[:8]} "
                                f"(idle {idle_time:.0f}s > {self.idle_timeout}s)\n"
                            )
                            self._terminate_process(session)

    def _evict_if_needed(self):
        """Evict least-recently-used process if pool is full."""
        active = [
            (s.last_active, sid, s)
            for sid, s in self._sessions.items()
            if s.process and s.process.poll() is None and s.status == "active"
        ]
        if len(active) < self.max_processes:
            return

        # Sort by last_active ascending (oldest first)
        active.sort()
        while len(active) >= self.max_processes:
            _, sid, session = active.pop(0)
            sys.stderr.write(f"[chat-session] Evicting session {sid[:8]} (pool full)\n")
            self._terminate_process(session)

    def _restore_sessions(self):
        """Restore session metadata from DB. Processes start lazily."""
        rows = self._db.execute(
            "SELECT * FROM chat_sessions WHERE status = 'active'"
        ).fetchall()

        for row in rows:
            session = ChatSession(
                session_id=row["session_id"],
                graph_id=row["graph_id"],
                title=row["title"] or "",
                created_at=row["created_at"],
                last_active=time.time(),  # Reset idle timer
                status="active",
                claude_session_id=row["claude_session_id"] or "",
            )
            self._sessions[session.session_id] = session

    # -- Serialization -----------------------------------------------------

    def _session_to_dict(self, session: ChatSession) -> dict:
        return {
            "session_id": session.session_id,
            "claude_session_id": session.claude_session_id,
            "graph_id": session.graph_id,
            "title": session.title,
            "created_at": session.created_at,
            "last_active_at": datetime.fromtimestamp(
                session.last_active, tz=timezone.utc
            ).isoformat(),
            "status": session.status,
            "process_alive": session.process is not None and session.process.poll() is None,
        }
