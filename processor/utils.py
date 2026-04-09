"""
DeepDream 通用工具函数。

放在此处的是被多个模块复用的纯函数，不依赖任何业务状态。
"""
from __future__ import annotations

import hashlib
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
    text = re.sub(r'^```\s*markdown\s*\n?', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^```\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def clean_separator_tags(text: str) -> str:
    """清理 LLM 回显的 XML 分隔符标签。

    弱模型（如 qwen2.5-instruct）会把 prompt 中的 <记忆缓存>、<输入文本> 等
    XML 分隔符标签原样回显到输出中。此函数将这些标签移除，只保留实际内容。
    """
    text = _SEPARATOR_TAG_RE.sub('', text)
    # 清理标签移除后可能产生的多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# 线程局部窗口标签 + 流水线角色（并行 remember 时区分主线程 / 抽取 / 步骤6 / 7）
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
    "步骤6": "S6",
    "步骤7": "S7",
}


def _abbr_role(role: str) -> str:
    if not role:
        return "----"
    return _ROLE_ABBR.get(role, role[:4])


def _emit_log_line(line: str) -> None:
    global _log_queue, _log_writer_started
    if not _log_serial:
        print(line, flush=True)
        return
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
    assert _log_queue is not None
    _log_queue.put(line)


def set_window_label(label: str | None) -> None:
    """设置当前线程的窗口标签（如 'W6/1426'），传 None 清除。"""
    _window_local.label = label


def get_window_label() -> str:
    """获取当前线程的窗口标签，无标签时返回空字符串。"""
    return getattr(_window_local, 'label', None) or ""


def set_pipeline_role(role: str | None) -> None:
    """设置当前线程的流水线角色（如「主线程」「抽取」「步骤6」「步骤7」），传 None 清除。"""
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


def wprint(msg: str = "") -> None:
    """并行友好：固定列「时间 窗号 角色 | 正文」，经队列串行写出避免行级交错。"""
    label = get_window_label() or "----"
    role = _abbr_role(get_pipeline_role())
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"{ts} {label:>10} {role:4} | {msg}"
    _emit_log_line(line)
