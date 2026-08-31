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
from functools import lru_cache

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


# ---------------------------------------------------------------------------
# 实体名身份归一化（全库唯一判定语义，P2 统一）
#
# 历史上存在 4 处语义不一致的实现（窗口去重 / 候选匹配 / FamilyWriteGate /
# 关系端点解析各判各的），导致 '张三教授'vs'张三'、'IBM'vs'ibm' 在不同层
# 得到不同的同一性结论。现在全部收敛到 entity_match_key。
# ---------------------------------------------------------------------------

# 括号注记（全角/半角）：LLM 常在名称尾部加场景注记，如 "曹操（魏王）"
_PAREN_ANNOTATION_RE = re.compile(r'[（(][^）)]+[)）]')

# 中文称号后缀：仅剥尾部，"张三教授" → "张三"
_TITLE_SUFFIXES_RE = re.compile(
    r'(?:教授|博士|先生|女士|同学|老师|工程师|经理|总监|院长|所长|主任|校长|站长|馆长|主编|首席|总裁'
    r'|部长|省长|市长|县长|区长|镇长|村长|将军|上校|中校|少校|大校|司令|参谋|政委|舰长|机长)$'
)

_MATCH_WS_RE = re.compile(r"\s+")


@lru_cache(maxsize=8192)
def entity_match_key(name: str) -> str:
    """实体身份匹配键：剥离括号注记与中文称号后缀 + 压缩空白 + casefold。

    窗口内去重、候选匹配、关系端点解析、FamilyWriteGate、跨窗口同名
    合并全部走这里——同一对名称在任意层得到同一结论。仅用于比较键，
    展示名（canonical_name）保留原文。
    """
    raw = _MATCH_WS_RE.sub(" ", str(name or "")).strip()
    s = _PAREN_ANNOTATION_RE.sub('', raw)
    # 叠加称号（"李四博士教授"）最多剥两层
    stripped = _TITLE_SUFFIXES_RE.sub('', s).strip()
    if stripped and stripped != s:
        again = _TITLE_SUFFIXES_RE.sub('', stripped).strip()
        if again:
            stripped = again
    s = stripped or s or raw
    return s.casefold()


def entity_name_variants(name: str) -> tuple:
    """返回名称的查找变体 (原文, 核心名)，用于 DB 按名查 family。

    核心名 = 剥括号注记与称号后缀、压缩空白，但保留大小写——DB 里存的
    canonical_name 是原文，大小写折叠交给 SQL COLLATE NOCASE。
    """
    raw = _MATCH_WS_RE.sub(" ", str(name or "")).strip()
    core = _PAREN_ANNOTATION_RE.sub('', raw)
    stripped = _TITLE_SUFFIXES_RE.sub('', core).strip()
    if stripped and stripped != core:
        again = _TITLE_SUFFIXES_RE.sub('', stripped).strip()
        if again:
            stripped = again
    return (raw, (stripped or core or raw))


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

# When set, pipeline logs go to stderr so stdout remains clean for JSON output.
_json_mode: bool = os.environ.get("DEEPDREAM_JSON_OUTPUT", "").strip().lower() in ("1", "true", "yes")
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
    _use_stderr = _json_mode or os.environ.get("DEEPDREAM_JSON_OUTPUT", "").strip().lower() in ("1", "true", "yes")
    if not _log_serial:
        print(line, file=sys.stderr if _use_stderr else sys.stdout, flush=True)
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
                _use_stderr = _json_mode or os.environ.get("DEEPDREAM_JSON_OUTPUT", "").strip().lower() in ("1", "true", "yes")
                _out = sys.stderr if _use_stderr else sys.stdout
                while True:
                    item = _log_queue.get()
                    if item is None:
                        break
                    try:
                        _out.write(item + "\n")
                        _out.flush()
                    except UnicodeEncodeError:
                        try:
                            _out.write(item.encode("utf-8", errors="replace").decode("utf-8", errors="replace") + "\n")
                            _out.flush()
                        except (OSError, ValueError):
                            # Test runners and embedders may close the stream
                            # while this daemon is draining its queue.
                            break
                    except (OSError, ValueError):
                        # Never leak a background-thread traceback when stdout
                        # is replaced/closed during shutdown.
                        break

            threading.Thread(target=_writer, name="tmg-log-writer", daemon=True).start()
    _log_queue.put(line)


def set_window_label(label: str | None) -> None:
    """设置当前线程的窗口标签（如 'W6/1426'），传 None 清除。"""
    _window_local.label = label


def capture_log_context() -> tuple:
    """捕获当前线程的日志上下文（窗口标签 + 流水线角色），供子线程恢复用。"""
    return (
        getattr(_window_local, 'label', None),
        getattr(_window_local, 'pipeline_role', None),
    )


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


# ---------------------------------------------------------------------------
# Episode type classification heuristic
# ---------------------------------------------------------------------------

# Conversation indicators: dialogue lines, attribution verbs, quotation patterns
_CONV_QUOTE_LINE_RE = re.compile(r'^\s*["“「『【].+[”“」』】]', re.MULTILINE)
_CONV_ATTRIBUTION_RE = re.compile(
    r'(?:said|replied|asked|answered|whispered|shouted|exclaimed|murmured|'
    r'说道|回答|问|喊|低声|大声|笑道|叫|答|喃喃)',
    re.IGNORECASE,
)
_CONV_FENCE_RE = re.compile(r'["“”「」].+["“”「」]')

# Narrative indicators: third-person past-tense verbs, story connectives
_NARRATIVE_PAST_RE = re.compile(
    r'(?:\b(?:he|she|it|they|the \w+)\s+'
    r'(?:was|had|went|came|took|saw|knew|felt|turned|looked|walked|ran|sat|stood|'
    r'stood|heard|felt|watched|smiled|nodded|opened|closed|reached|picked)\b)',
    re.IGNORECASE,
)
_NARRATIVE_CONNECTIVE_RE = re.compile(
    r'(?:\b(?:then|after|before|while|when|as|suddenly|meanwhile|later|finally)\b'
    r'|然后|接着|之后|突然|这时|终于|不久|过了一会儿)',
    re.IGNORECASE,
)


def classify_episode_type(text: str) -> str:
    """Classify episode type from source text using simple heuristics.

    Returns one of: 'conversation', 'narrative', 'fact'.

    - 'conversation': text contains dialogue patterns (quoted speech lines,
      attribution verbs like "said/replied", or multiple quotation fences).
    - 'narrative': text shows narrative structure (third-person past-tense
      verbs, story connectives) without dominant dialogue.
    - 'fact': default for everything else (factual/informational content).
    """
    if not text or not text.strip():
        return "fact"

    _conv_score = 0
    _has_attribution = bool(_CONV_ATTRIBUTION_RE.search(text))
    if _CONV_QUOTE_LINE_RE.search(text):
        _conv_score += 2
    if _has_attribution:
        _conv_score += 1
    # Quoted speech fragments
    _fences = _CONV_FENCE_RE.findall(text)
    if _fences:
        _conv_score += 1
    # Strong dialogue: attribution verb + quoted fragments together
    if _has_attribution and _fences:
        _conv_score += 1
    # Many quoted fragments alone strongly suggest dialogue
    if len(_fences) >= 3:
        _conv_score += 2

    if _conv_score >= 3:
        return "conversation"

    _narr_score = 0
    _past_matches = _NARRATIVE_PAST_RE.findall(text)
    if len(_past_matches) >= 2:
        _narr_score += 2
    if _NARRATIVE_CONNECTIVE_RE.search(text):
        _narr_score += 2

    if _narr_score >= 3:
        return "narrative"

    return "fact"


def wprint_debug(msg: str = "") -> None:
    """Level-aware version of wprint for debug/progress messages."""
    _pipeline_logger.debug(msg)


def wprint_info(msg: str = "") -> None:
    """Level-aware version of wprint for step milestones."""
    _pipeline_logger.info(msg)


def wprint_warn(msg: str = "") -> None:
    """Level-aware version of wprint for warnings."""
    _pipeline_logger.warning(msg)
