"""Chat session logger for PHA.

Provides structured logging of chat sessions with per-user directories,
per-conversation folders, summary statistics, and raw verbose logs.

Folder structure:
    chat_logs/
    └── {user_id}/
        └── {baseline}_{model_short}_{YYYY-MM-DD_HH-MM-SS}/
            ├── summary.json          # Session config, per-turn stats, aggregates
            └── raw_conversation.log  # Full verbose stdout/stderr capture
"""

import io
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 近似代币计数器( 避免对代币的依赖)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Stdout /stderr 抓取上下文管理器
# ---------------------------------------------------------------------------

class _OutputCapture:
    """Thread-aware stdout/stderr capture.

    Captures output from *all* threads while active (agent work often
    spawns ThreadPoolExecutor workers).  The captured text is stored in
    ``self.output`` and can be retrieved after the context exits.
    """

    def __init__(self):
        self.output = ""
        self._buf = io.StringIO()
        self._orig_stdout = None
        self._orig_stderr = None
        self._tee_stdout = None
        self._tee_stderr = None

    # ---- helpers --------------------------------------------------------

    class _Tee:
        """Write to both the original stream and a buffer simultaneously."""

        def __init__(self, original, buf: io.StringIO):
            self._original = original
            self._buf = buf
            self._lock = threading.Lock()

        def write(self, data):
            if data:
                with self._lock:
                    self._original.write(data)
                    self._buf.write(data)

        def flush(self):
            self._original.flush()

        # 将其它属性查询转发到原始流中
        # 诸如``sys.stdout.encoding``之类的东西仍然有效。
        def __getattr__(self, name):
            return getattr(self._original, name)

    # ---- context manager ------------------------------------------------

    def __enter__(self):
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._tee_stdout = self._Tee(self._orig_stdout, self._buf)
        self._tee_stderr = self._Tee(self._orig_stderr, self._buf)
        sys.stdout = self._tee_stdout
        sys.stderr = self._tee_stderr
        return self

    def __exit__(self, *_exc):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self.output = self._buf.getvalue()
        self._buf.close()


# ---------------------------------------------------------------------------
# 每回合记录
# ---------------------------------------------------------------------------

class TurnRecord:
    """Statistics for a single user→assistant turn."""

    def __init__(self, turn_number: int, user_message: str):
        self.turn_number = turn_number
        self.user_message = user_message
        self.user_tokens = _estimate_tokens(user_message)
        self.assistant_response: str = ""
        self.assistant_tokens: int = 0
        self.processing_time_ms: float = 0.0
        self.thinking_steps: List[Dict[str, str]] = []
        self.agents_called: List[str] = []
        self.tools_used: Dict[str, int] = {}
        self.errors: int = 0
        self.raw_output: str = ""
        self.timestamp: str = datetime.now().isoformat()

    def finalize(
        self,
        assistant_response: str,
        processing_time_ms: float,
        thinking_steps: Optional[List[Dict[str, str]]] = None,
        agents_called: Optional[List[str]] = None,
        tools_used: Optional[Dict[str, int]] = None,
        errors: int = 0,
        raw_output: str = "",
    ):
        self.assistant_response = assistant_response
        self.assistant_tokens = _estimate_tokens(assistant_response)
        self.processing_time_ms = processing_time_ms
        self.thinking_steps = thinking_steps or []
        self.agents_called = agents_called or []
        self.tools_used = tools_used or {}
        self.errors = errors
        self.raw_output = raw_output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_number": self.turn_number,
            "timestamp": self.timestamp,
            "user_message": self.user_message,
            "user_tokens_approx": self.user_tokens,
            "assistant_response": self.assistant_response,
            "assistant_tokens_approx": self.assistant_tokens,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "thinking_steps": self.thinking_steps,
            "agents_called": self.agents_called,
            "tools_used": self.tools_used,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# ChatSessionLogger
# ---------------------------------------------------------------------------

class ChatSessionLogger:
    """Manages on-disk logging for a single chat session.

    Call :meth:`begin_turn` before agent processing and :meth:`end_turn`
    afterwards.  :meth:`flush` writes the summary JSON and appends any
    new raw output to the log file.  :meth:`close` writes the final
    summary.
    """

    def __init__(
        self,
        user_id: str,
        session_id: str,
        baseline: str,
        provider: str,
        model_id: str,
        persona_id: str,
        logs_root: Optional[Path] = None,
    ):
        self.user_id = user_id or "anonymous"
        self.session_id = session_id
        self.baseline = baseline
        self.provider = provider
        self.model_id = model_id
        self.persona_id = persona_id
        self.created_at = datetime.now()

        # 构建文件夹名称：{ 基线 } model_short {YYYY-MM -DD_HH-MM-SS }
        model_short = model_id.split("/")[-1] if "/" in model_id else model_id
        ts = self.created_at.strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{baseline}_{model_short}_{ts}"

        if logs_root is None:
            # 默认：<project_root> /chat_logs (中文(简体) ).
            logs_root = Path(__file__).resolve().parent.parent.parent / "chat_logs"

        self.session_dir: Path = logs_root / self.user_id / folder_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.summary_path = self.session_dir / "summary.json"
        self.raw_log_path = self.session_dir / "raw_conversation.log"

        # 状态
        self.turns: List[TurnRecord] = []
        self._current_turn: Optional[TurnRecord] = None
        self._capture: Optional[_OutputCapture] = None

        # 将初始信头写入原始日志
        with open(self.raw_log_path, "w", encoding="utf-8") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"PHA Chat Session — Raw Verbose Log\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Session ID : {session_id}\n")
            f.write(f"User ID    : {self.user_id}\n")
            f.write(f"Baseline   : {baseline}\n")
            f.write(f"Provider   : {provider}\n")
            f.write(f"Model      : {model_id}\n")
            f.write(f"Persona    : {persona_id}\n")
            f.write(f"Created    : {self.created_at.isoformat()}\n")
            f.write(f"{'=' * 80}\n\n")

        # 写入初始摘要
        self._write_summary()

    # ------------------------------------------------------------------
    # 改变生命周期
    # ------------------------------------------------------------------

    def begin_turn(self, user_message: str) -> _OutputCapture:
        """Start recording a new turn.

        Returns an :class:`_OutputCapture` context-manager that the
        caller should use to wrap the agent-processing block so all
        stdout/stderr is captured for the raw log.
        """
        turn_number = len(self.turns) + 1
        self._current_turn = TurnRecord(turn_number, user_message)

        # 立即将用户消息写入原始日志
        with open(self.raw_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'─' * 80}\n")
            f.write(f"TURN {turn_number}  [{datetime.now().isoformat()}]\n")
            f.write(f"{'─' * 80}\n")
            f.write(f"USER: {user_message}\n")
            f.write(f"{'─' * 40} agent processing {'─' * 21}\n")

        self._capture = _OutputCapture()
        return self._capture

    def end_turn(
        self,
        assistant_response: str,
        processing_time_ms: float,
        thinking_steps: Optional[List[Dict[str, str]]] = None,
        agents_called: Optional[List[str]] = None,
        tools_used: Optional[Dict[str, int]] = None,
        errors: int = 0,
    ):
        """Finalize the current turn and write to disk."""
        if self._current_turn is None:
            return

        raw_output = ""
        if self._capture is not None:
            raw_output = self._capture.output
            self._capture = None

        self._current_turn.finalize(
            assistant_response=assistant_response,
            processing_time_ms=processing_time_ms,
            thinking_steps=thinking_steps,
            agents_called=agents_called,
            tools_used=tools_used,
            errors=errors,
            raw_output=raw_output,
        )

        self.turns.append(self._current_turn)

        # 附加到原始日志中— 完整的动词输出+助手响应
        with open(self.raw_log_path, "a", encoding="utf-8") as f:
            if raw_output:
                f.write(raw_output)
                if not raw_output.endswith("\n"):
                    f.write("\n")
            f.write(f"{'─' * 40} end agent processing {'─' * 18}\n")
            f.write(f"ASSISTANT: {assistant_response}\n")
            f.write(f"[processing_time={processing_time_ms:.0f}ms  "
                    f"agents={','.join(agents_called or [])}  "
                    f"tools={tools_used or {}}  "
                    f"errors={errors}]\n")

        self._current_turn = None
        self._write_summary()

    # ------------------------------------------------------------------
    # 总表一/O
    # ------------------------------------------------------------------

    def _build_summary(self) -> Dict[str, Any]:
        last_activity = self.turns[-1].timestamp if self.turns else self.created_at.isoformat()
        duration_s = (datetime.now() - self.created_at).total_seconds()

        total_user_tokens = sum(t.user_tokens for t in self.turns)
        total_asst_tokens = sum(t.assistant_tokens for t in self.turns)
        total_processing_ms = sum(t.processing_time_ms for t in self.turns)
        total_errors = sum(t.errors for t in self.turns)

        # 跨所有转弯的聚合剂和工具
        all_agents: Dict[str, int] = {}
        all_tools: Dict[str, int] = {}
        for t in self.turns:
            for a in t.agents_called:
                all_agents[a] = all_agents.get(a, 0) + 1
            for tool, count in t.tools_used.items():
                all_tools[tool] = all_tools.get(tool, 0) + count

        return {
            "session": {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "baseline": self.baseline,
                "provider": self.provider,
                "model_id": self.model_id,
                "persona_id": self.persona_id,
                "created_at": self.created_at.isoformat(),
                "last_activity": last_activity,
                "duration_seconds": round(duration_s, 1),
                "status": "active" if self._current_turn else "idle",
            },
            "aggregate": {
                "total_turns": len(self.turns),
                "total_user_tokens_approx": total_user_tokens,
                "total_assistant_tokens_approx": total_asst_tokens,
                "total_tokens_approx": total_user_tokens + total_asst_tokens,
                "total_processing_time_ms": round(total_processing_ms, 1),
                "avg_processing_time_ms": round(total_processing_ms / len(self.turns), 1) if self.turns else 0,
                "total_errors": total_errors,
                "agents_called": all_agents,
                "tools_used": all_tools,
            },
            "turns": [t.to_dict() for t in self.turns],
        }

    def _write_summary(self):
        """Write the summary JSON to disk (overwrites each time)."""
        summary = self._build_summary()
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 会议闭幕
    # ------------------------------------------------------------------

    def close(self):
        """Finalize the session log.

        Writes the final summary with ``status: closed`` and appends a
        footer to the raw log.
        """
        duration_s = (datetime.now() - self.created_at).total_seconds()

        # 最终原始日志页脚
        with open(self.raw_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"SESSION CLOSED  [{datetime.now().isoformat()}]\n")
            f.write(f"Total turns: {len(self.turns)}  "
                    f"Duration: {duration_s:.1f}s\n")
            f.write(f"{'=' * 80}\n")

        # 最后摘要
        summary = self._build_summary()
        summary["session"]["status"] = "closed"
        summary["session"]["closed_at"] = datetime.now().isoformat()
        summary["session"]["duration_seconds"] = round(duration_s, 1)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
