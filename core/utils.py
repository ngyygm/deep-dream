"""
DeepDream 通用工具函数。

放在此处的是被多个模块复用的纯函数，不依赖任何业务状态。
"""
from __future__ import annotations

import hashlib
import logging
import numpy as np
import os
import queue
import re
import sys
import threading
from datetime import datetime

# prompt 中用作分隔符的所有 XML 标签名（不含尖括号）
_SEPARATOR_TAG_NAMES = frozenset({
    "记忆缓存", "输入文本", "旧内容", "新内容", "上一文档记忆", "当前文档",
    "旧版本内容", "新版本内容", "概念实体列表", "已有关系", "已有关系列表",
    "新关系", "操作详情", "实体信息", "关系信息", "当前实体", "候选实体",
    "候选实体列表", "实体列表", "实体对", "新关系描述", "未覆盖实体",
    "已抽取实体", "指定实体名称", "关系描述列表", "实体内容列表",
})

# 预编译正则：匹配所有分隔符标签 <tag> 和 </tag>
_SEPARATOR_TAG_RE = re.compile(
    r'</?(?:' + '|'.join(re.escape(n) for n in _SEPARATOR_TAG_NAMES) + r')>\s*'
)

# Pre-compiled regex for markdown code block cleanup
_MD_CODE_OPEN_RE = re.compile(r'^```\s*markdown\s*\n?', re.MULTILINE | re.IGNORECASE)
_MD_CODE_FENCE_RE = re.compile(r'^```\s*\n?', re.MULTILINE)
_MD_CODE_CLOSE_RE = re.compile(r'\n?```\s*$', re.MULTILINE)
_EXCESS_NEWLINES_RE = re.compile(r'\n{3,}')


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """计算 Jaccard 相似度（基于 bigram 集合）。

    被 StorageManager 和 EntityProcessor 共用的纯函数。
    """
    s1 = (text1 or "").lower().strip()
    s2 = (text2 or "").lower().strip()
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    return _jaccard_from_bigrams(_bigrams(s1), _bigrams(s2))


def _bigrams(s: str) -> frozenset:
    """Pre-compute bigram set for Jaccard similarity."""
    if len(s) < 2:
        return frozenset(s)  # char-level fallback for single-char strings
    return frozenset(s[i:i+2] for i in range(len(s) - 1))


def _jaccard_from_bigrams(set1: frozenset, set2: frozenset) -> float:
    """Compute Jaccard from pre-computed bigram sets."""
    if not set1 or not set2:
        return 0.0
    union = len(set1 | set2)
    return len(set1 & set2) / union if union else 0.0


def cosine_similarity(vec1, vec2) -> float:
    """Compute cosine similarity between two numpy arrays.

    Shared by EntityProcessor and cross-window dedup.
    Fast path: skip asarray/flatten when inputs are already 1-D arrays.
    """
    if vec1 is None or vec2 is None:
        return 0.0
    # Fast path: both already 1-D ndarray (common case from embedding lookups)
    if isinstance(vec1, np.ndarray) and isinstance(vec2, np.ndarray) and vec1.ndim == 1 and vec2.ndim == 1:
        dot_ab = np.dot(vec1, vec2)
        denom = np.sqrt(np.dot(vec1, vec1) * np.dot(vec2, vec2)) + 1e-9
        return float(dot_ab / denom)
    a = vec1 if isinstance(vec1, np.ndarray) else np.asarray(vec1)
    b = vec2 if isinstance(vec2, np.ndarray) else np.asarray(vec2)
    if a.ndim > 1:
        a = a.ravel()
    if b.ndim > 1:
        b = b.ravel()
    dot_ab = np.dot(a, b)
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-9
    return float(dot_ab / denom)


def compute_doc_hash(text: str) -> str:
    """计算文本的 doc_hash（MD5 前12位），用于缓存去重和断点续传。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def normalize_entity_pair(entity1: str, entity2: str) -> tuple:
    """标准化实体对：按字典序排序，使无向边端点固定。

    多处复用的纯函数，用于确保 (A,B) 和 (B,A) 被视为同一关系。
    """
    a, b = (entity1 or "").strip(), (entity2 or "").strip()
    return (a, b) if a <= b else (b, a)


def clean_markdown_code_blocks(text: str) -> str:
    """清理文本中的 markdown 代码块标识符。

    移除 ````markdown` / ```` 等标记，返回纯净内容。
    """
    text = _MD_CODE_OPEN_RE.sub('', text)
    text = _MD_CODE_FENCE_RE.sub('', text)
    text = _MD_CODE_CLOSE_RE.sub('', text)
    return text.strip()


def clean_separator_tags(text: str) -> str:
    """清理 LLM 回显的 XML 分隔符标签。

    弱模型（如 qwen2.5-instruct）会把 prompt 中的 <记忆缓存>、<输入文本> 等
    XML 分隔符标签原样回显到输出中。此函数将这些标签移除，只保留实际内容。
    """
    text = _SEPARATOR_TAG_RE.sub('', text)
    # 清理标签移除后可能产生的多余空行
    text = _EXCESS_NEWLINES_RE.sub('\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# 线程局部窗口标签 + 流水线角色（并行 remember 时区分主线程 / 抽取 / 步骤9 / 10）
# ---------------------------------------------------------------------------

_window_local = threading.local()

# 并行时日志：单行原子输出，避免多线程 print 交错；可用 DEEPDREAM_LOG_SERIAL=0 关闭（直接 print）
_log_serial: bool = os.environ.get("DEEPDREAM_LOG_SERIAL", "1").strip().lower() not in ("0", "false", "no")
_log_queue: queue.Queue[str] | None = None
_log_writer_started = False
_log_writer_lock = threading.Lock()

_ROLE_ABBR = {
    "主线程": "MAIN",
    "抽取": "EXT",
    "步骤9": "S9",
    "步骤10": "S10",
}


def _abbr_role(role: str) -> str:
    if not role:
        return "----"
    return _ROLE_ABBR.get(role, role[:4])


def _emit_log_line(line: str) -> None:
    """Emit a formatted log line. Routes through logging if configured, else queue/print."""
    global _log_queue, _log_writer_started
    if not _log_serial:
        print(line, flush=True)
        return
    # Fast path: writer already started — no lock needed
    if _log_writer_started:
        _log_queue.put(line)
        return
    # Slow path: one-time initialization under lock
    with _log_writer_lock:
        if _log_queue is None:
            _log_queue = queue.Queue()
        if not _log_writer_started:
            _log_writer_started = True

            def _writer() -> None:
                assert _log_queue is not None
                while True:
                    item = _log_queue.get()
                    if item is None:
                        break
                    sys.stdout.write(item + "\n")
                    sys.stdout.flush()

            threading.Thread(target=_writer, name="tmg-log-writer", daemon=True).start()
    _log_queue.put(line)


def set_window_label(label: str | None) -> None:
    """设置当前线程的窗口标签（如 'W6/1426'），传 None 清除。"""
    _window_local.label = label


def get_window_label() -> str:
    """获取当前线程的窗口标签，无标签时返回空字符串。"""
    return getattr(_window_local, 'label', None) or ""


def set_pipeline_role(role: str | None) -> None:
    """设置当前线程的流水线角色（如「主线程」「抽取」「步骤9」「步骤10」），传 None 清除。"""
    _window_local.pipeline_role = role


def get_pipeline_role() -> str:
    return getattr(_window_local, 'pipeline_role', None) or ""


def clear_parallel_log_context() -> None:
    """清除窗号与角色（进入非并行段或流程入口时调用）。"""
    _window_local.label = None
    _window_local.pipeline_role = None


def remember_log(msg: str) -> None:
    """无窗号时的 Remember 级说明（入口、断点、提示）。"""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"{ts} {'Remember':>10} ---- | {msg}"
    _emit_log_line(line)


# ---------------------------------------------------------------------------
# Logging adapter: routes wprint-style messages through Python logging
# ---------------------------------------------------------------------------

class _QueueLogHandler(logging.Handler):
    """Logging handler that routes through _emit_log_line for serialized output."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            label = get_window_label() or "----"
            role = _abbr_role(get_pipeline_role())
            ts = datetime.now().strftime("%H:%M:%S")
            msg = self.format(record)
            line = f"{ts} {label:>10} {role:4} | {msg}"
            _emit_log_line(line)
        except Exception:
            self.handleError(record)


# Set up the pipeline logger with our custom handler
_pipeline_logger = logging.getLogger("tmg.pipeline")
if not _pipeline_logger.handlers:
    _handler = _QueueLogHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _pipeline_logger.addHandler(_handler)
    _pipeline_logger.setLevel(logging.DEBUG)
    _pipeline_logger.propagate = False  # Prevent duplicate output via root logger


def wprint(msg: str = "") -> None:
    """并行友好：固定列「时间 窗号 角色 | 正文」，经队列串行写出避免行级交错。"""
    label = get_window_label() or "----"
    role = _abbr_role(get_pipeline_role())
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"{ts} {label:>10} {role:4} | {msg}"
    _emit_log_line(line)


def wprint_debug(msg: str = "") -> None:
    """Level-aware version of wprint for debug/progress messages."""
    _pipeline_logger.debug(msg)


def wprint_info(msg: str = "") -> None:
    """Level-aware version of wprint for step milestones."""
    _pipeline_logger.info(msg)


def wprint_warn(msg: str = "") -> None:
    """Level-aware version of wprint for warnings."""
    _pipeline_logger.warning(msg)
