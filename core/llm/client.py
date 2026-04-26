"""
LLM客户端：封装LLM调用，实现三个核心任务。

请求方式：统一通过 `processor/ollama_chat_api.py` 访问：
- Ollama 走原生 `POST /api/chat`；
- OpenAI/GLM/LM Studio 等走 OpenAI 兼容接口。

think 模式由初始化参数 think_mode 控制；只有 Ollama 原生协议支持通过 `think: true/false` 显式开关思考模式。
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
import heapq
import json
import os
import re
import threading
import time

from ..models import Episode
from ..utils import clean_separator_tags, wprint_info
from .chat_api import ollama_chat, openai_compatible_chat
from .errors import LLMContextBudgetExceeded
from .memory_ops import _MemoryOpsMixin
from .content_merger import _ContentMergerMixin
from .consolidation import _ConsolidationMixin
from .summary_evolution import SummaryEvolutionMixin
from .contradiction import ContradictionDetectionMixin
from .agent_query import AgentQueryMixin
from .extraction import _LLMExtractionMixin

try:
    from openai import RateLimitError
except ImportError:  # pragma: no cover
    RateLimitError = None  # type: ignore[misc,assignment]

# 非 TPM 类错误：失败后等待 3^1, 3^2, … 秒再重试，最多 5 轮（第 6 次失败则放弃）
_LLM_BACKOFF_BASE = 3
_LLM_MAX_FAILURE_ROUNDS = 5
# 单次等待上限，避免 TPM 无限重试时指数爆炸占满进程
_LLM_TPM_SLEEP_CAP_SECONDS = 3600
_DISTILL_SKIP_STEPS = frozenset(("02_extract_entities", "03_extract_relations"))

# Static error keywords — computed once at import time, not per-call
_RATE_LIMIT_KEYWORDS = ("rate_limit", "rate limit", "tpm", "throttl", "capacity", "overloaded")
_CONNECTION_ERROR_KEYWORDS = frozenset((
    "connection refused", "connectionerror",
    "failed to establish a new connection", "newconnectionerror",
    "temporarily unreachable", "temporary failure in name resolution",
    "name or service not known", "connection aborted",
    "connection reset", "errno 111",
))
_CONTEXT_OVERFLOW_NEEDLES = (
    "context length", "maximum context", "max context", "context window",
    "token limit", "too many tokens", "maximum tokens", "exceeds the maximum",
    "prompt is too long", "input is too long", "input length", "length limit",
    "reduce the length", "payload too large", "请求过长", "上下文长度",
    "上下文超限", "tokens 超", "token 超", "invalid prompt", "context_limit",
)

# Pre-compiled regex for JSON cleanup
_TRAILING_COMMA_RE = re.compile(r',(\s*[}\]])')
_INVALID_UNICODE_ESCAPE_RE = re.compile(r'\\u([0-9a-fA-F]{0,3})(?![0-9a-fA-F])')
_CJK_PUNCT_RE = re.compile(r'[\uff1a\uff0c\uff1b]')  # ：，；
_CJK_PUNCT_MAP = {'\uff1a': ':', '\uff0c': ',', '\uff1b': ';'}
_CURRENT_ENTITY_NAME_RE = re.compile(r"<当前实体>.*?name:\s*(\S+)", re.DOTALL)
_FAMILY_ID_RE = re.compile(r"family_id:\s*(\S+)")
_ENTRY_NAME_RE = re.compile(r"name:\s*(\S+)")
# Single-pass JSON fence extraction — replaces 4 separate find() calls
_JSON_FENCE_RE = re.compile(r'```(?:json)?\s*\n?', re.DOTALL)

# JSON 解析失败时追加给模型的纠错提示（配合 call_llm_until_json_parses）
_JSON_RETRY_USER_MESSAGE = (
    "【输出格式纠错】上一条输出无法被解析为合法 JSON。"
    "请严格只输出一个 markdown `json` 代码块，不要任何解释文字；"
    "若是数组，代码块内部必须是合法 JSON 数组；若是对象，代码块内部必须是合法 JSON 对象。"
)
# 疑似截断（未闭合字符串等）时追加：引导缩短字段，避免再次超长
_JSON_RETRY_TRUNCATION_SUFFIX = (
    " 若疑似因输出过长在字符串中间被截断：请缩小每条 content 的篇幅（建议单字段不超过约 200 字），"
    "字符串内的换行必须写成转义 \\n；仍只输出一个合法的 ```json ... ``` 代码块。"
)
# Truncation detection keywords — computed once at import time
_TRUNCATION_KEYWORDS = (
    "Unterminated string",
    "Expecting value",
    "Expecting ',' delimiter",
    "Unterminated",
)


def _mock_json_fence(payload: Any) -> str:
    """将可 JSON 序列化的值包在单个 ```json 代码块内，与线上 prompt 约定一致。"""
    body = json.dumps(payload, ensure_ascii=False)
    return f"```json\n{body}\n```"


def _fix_unicode_escapes(text: str) -> str:
    """修复无效的 Unicode 转义序列（\\u 后不足 4 位十六进制）。"""
    def _replace_invalid_escape(match):
        hex_part = match.group(1)
        if not hex_part:
            return '\\u0020'
        elif len(hex_part) < 4:
            return '\\u' + hex_part.ljust(4, '0')
        else:
            return match.group(0)
    return _INVALID_UNICODE_ESCAPE_RE.sub(_replace_invalid_escape, text)


def _escape_control_chars_in_json_strings(text: str) -> str:
    """仅在 JSON 字符串内部转义裸控制字符，避免破坏结构字符。"""
    result = []
    in_string = False
    escaped = False

    for ch in text:
        if in_string:
            if escaped:
                result.append(ch)
                escaped = False
                continue
            if ch == '\\':
                result.append(ch)
                escaped = True
                continue
            if ch == '"':
                result.append(ch)
                in_string = False
                continue
            if ch == '\n':
                result.append('\\n')
                continue
            if ch == '\r':
                result.append('\\r')
                continue
            if ch == '\t':
                result.append('\\t')
                continue
            if ord(ch) < 0x20:
                result.append(f'\\u{ord(ch):04x}')
                continue
            result.append(ch)
        else:
            result.append(ch)
            if ch == '"':
                in_string = True
                escaped = False

    return ''.join(result)


def _is_rate_limit_tpm_error(exc: BaseException, _pre_lowered: str = None) -> bool:
    """429 / TPM / 速率限制：应长时间退避直至恢复，不计入普通重试上限。"""
    if RateLimitError is not None and isinstance(exc, RateLimitError):
        return True
    code = getattr(exc, "status_code", None)
    if code == 429:
        return True
    s = _pre_lowered if _pre_lowered is not None else str(exc).lower()
    # 检查 429 状态码相关字符串
    if "429" in s and ("error code" in s or "status code" in s):
        return True
    # 检查速率限制关键词（不依赖 429 状态码）
    return any(k in s for k in _RATE_LIMIT_KEYWORDS)


class PrioritySemaphore:
    """带优先级的信号量。priority 越小优先级越高，高优先级先获得锁。"""

    def __init__(self, value: int):
        if value < 1:
            raise ValueError("value must be >= 1")
        self._max_value = value
        self._value = value
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._heap: list = []  # [(priority, seq, event), ...]
        self._seq = 0

    @property
    def active_count(self) -> int:
        """当前正在使用的许可数（= 最大值 - 剩余值）。"""
        with self._lock:
            return self._max_value - self._value

    @property
    def max_value(self) -> int:
        return self._max_value

    def acquire(self, priority: int = 0):
        event = threading.Event()
        with self._cond:
            self._seq += 1
            if self._value > 0:
                self._value -= 1
                return
            heapq.heappush(self._heap, (priority, self._seq, event))
        event.wait()

    def release(self):
        with self._cond:
            while self._heap:
                _, _, event = heapq.heappop(self._heap)
                if not event.is_set():
                    event.set()
                    return
            self._value += 1


# 优先级常量（越小优先级越高）
LLM_PRIORITY_STEP1 = 0   # 步骤1: 更新缓存
LLM_PRIORITY_STEP2 = 1   # 步骤2: 抽取实体
LLM_PRIORITY_STEP3 = 2   # 步骤3: 抽取关系
LLM_PRIORITY_STEP4 = 3   # 步骤4: 补全实体
LLM_PRIORITY_STEP5 = 4   # 步骤5: 实体增强
LLM_PRIORITY_STEP6 = 5   # 步骤6: 实体对齐
LLM_PRIORITY_STEP7 = 6   # 步骤7: 关系对齐


class LLMClient(_MemoryOpsMixin, _ContentMergerMixin, _ConsolidationMixin,
                 SummaryEvolutionMixin, ContradictionDetectionMixin, AgentQueryMixin,
                 _LLMExtractionMixin):

    @staticmethod
    def _strip_opt_str(v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            t = v.strip()
            return t if t else None
        return None

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4", base_url: Optional[str] = None,
                 content_snippet_length: int = 50,
                 relation_content_snippet_length: int = 50,
                 relation_endpoint_jaccard_threshold: float = 0.9,
                 embedding_client: Any = None,
                 relation_endpoint_embedding_threshold: Optional[float] = 0.85,
                 think_mode: bool = False,
                 distill_data_dir: Optional[str] = None, max_tokens: Optional[int] = None,
                 context_window_tokens: Optional[int] = None,
                 timeout_seconds: Optional[int] = None,
                 connect_timeout_seconds: Optional[int] = None,
                 prompt_episode_max_chars: Optional[int] = None,
                 max_llm_concurrency: Optional[int] = None,
                 alignment_base_url: Optional[str] = None,
                 alignment_api_key: Optional[str] = None,
                 alignment_model: Optional[str] = None,
                 alignment_max_tokens: Optional[int] = None,
                 alignment_think_mode: Optional[bool] = None,
                 alignment_content_snippet_length: Optional[int] = None,
                 alignment_relation_content_snippet_length: Optional[int] = None,
                 alignment_enabled: bool = False,
                 alignment_max_llm_concurrency: Optional[int] = None):
        """
        初始化LLM客户端

        Args:
            api_key / model_name / base_url / content_snippet_length / relation_content_snippet_length / think_mode / max_tokens / context_window_tokens:
                步骤 1–5（上游滑窗与抽取）使用的配置；max_llm_concurrency 为步骤 1–5 的 LLM 并发上限。
                context_window_tokens：请求输入 prompt 的 token 预算上限；本地仅预检输入，不再用它压缩输出 max_tokens。
            timeout_seconds / connect_timeout_seconds:
                LLM API 请求超时配置（秒）。timeout_seconds 控制总请求超时（默认 300），connect_timeout_seconds 控制连接超时（默认 30）。
            prompt_episode_max_chars:
                进入抽取类 prompt 的记忆缓存最大字符数；超长时自动截断，避免异常缓存拖爆上下文预算。
            alignment_enabled:
                False 时忽略所有 alignment_*，步骤 6/7 与上游共用同一模型与（未拆分时）统一并发池。
            alignment_max_llm_concurrency:
                仅在 alignment_enabled 时生效：步骤 6/7 独立并发上限；未设时按原逻辑从 max_llm_concurrency 拆分下游槽位。
            alignment_*:
                步骤 6–7 可单独覆盖；未设置的项回退到上游对应项。
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.content_snippet_length = content_snippet_length
        self.relation_content_snippet_length = relation_content_snippet_length
        _jet = float(relation_endpoint_jaccard_threshold)
        self.relation_endpoint_jaccard_threshold = min(1.0, max(0.0, _jet))
        self._relation_embedding_client = embedding_client
        if relation_endpoint_embedding_threshold is None:
            self.relation_endpoint_embedding_threshold = None
        else:
            self.relation_endpoint_embedding_threshold = min(
                1.0, max(0.0, float(relation_endpoint_embedding_threshold))
            )
        self.think_mode = think_mode
        self.max_tokens = max_tokens
        if context_window_tokens is None:
            raise ValueError(
                "context_window_tokens 未设置。请在 service_config.json 的 llm 中配置 context_window_tokens，"
                "并由 TemporalMemoryGraphProcessor 传入 LLMClient。"
            )
        self.context_window_tokens = max(256, int(context_window_tokens))
        # Timeout configuration for LLM requests
        self.timeout_seconds = max(10, int(timeout_seconds)) if timeout_seconds is not None else 300
        self.connect_timeout_seconds = max(5, int(connect_timeout_seconds)) if connect_timeout_seconds is not None else 30
        if prompt_episode_max_chars is None:
            self.prompt_episode_max_chars = 2000
        else:
            self.prompt_episode_max_chars = max(0, int(prompt_episode_max_chars))

        self.alignment_base_url = self._strip_opt_str(alignment_base_url)
        if alignment_api_key is None:
            self.alignment_api_key = None
        elif isinstance(alignment_api_key, str):
            self.alignment_api_key = alignment_api_key.strip()
        else:
            self.alignment_api_key = alignment_api_key
        self.alignment_model = self._strip_opt_str(alignment_model)
        self.alignment_max_tokens = alignment_max_tokens
        self.alignment_think_mode = alignment_think_mode
        self.alignment_content_snippet_length = (
            int(alignment_content_snippet_length) if alignment_content_snippet_length is not None else None
        )
        self.alignment_relation_content_snippet_length = (
            int(alignment_relation_content_snippet_length)
            if alignment_relation_content_snippet_length is not None else None
        )
        self.alignment_enabled = bool(alignment_enabled)
        self._alignment_max_llm_concurrency: Optional[int] = None
        if alignment_max_llm_concurrency is not None:
            self._alignment_max_llm_concurrency = max(1, int(alignment_max_llm_concurrency))

        # 统一使用 Python SDK（openai>=1.0）访问；任一端点有 api/base 则非模拟模式
        self._endpoint_available = bool(
            api_key or base_url or self.alignment_base_url or (self.alignment_api_key is not None)
        )
        if not self._endpoint_available:
            wprint_info("提示：未提供 API key 或任一 base_url，将使用模拟响应模式")

        # LLM 并发：上游（步骤1–5）与下游（步骤6–7）两池
        self._max_llm_concurrency: int = max_llm_concurrency or 0
        self._llm_upstream_slot_max: int = 0
        self._llm_downstream_slot_max: int = 0
        self._llm_sem_upstream: Optional[PrioritySemaphore] = None
        self._llm_sem_downstream: Optional[PrioritySemaphore] = None
        self._llm_semaphore: Optional[PrioritySemaphore] = None  # 兼容旧代码/测试：与上游相同或总池
        mc = max_llm_concurrency or 0
        amc = self._alignment_max_llm_concurrency
        if self.alignment_enabled and mc >= 1 and amc is not None:
            # 对齐开启且单独指定下游并发：上游 = 步骤1–5，下游 = 步骤6–7
            self._llm_upstream_slot_max = int(mc)
            self._llm_downstream_slot_max = int(amc)
            self._llm_sem_upstream = PrioritySemaphore(self._llm_upstream_slot_max)
            self._llm_sem_downstream = PrioritySemaphore(self._llm_downstream_slot_max)
            self._llm_semaphore = self._llm_sem_upstream
        elif self.alignment_enabled and mc >= 1:
            # 对齐开启但未指定 alignment_max_concurrency：从上游总数中拆分下游（与旧版比例一致）
            if mc == 1:
                self._llm_upstream_slot_max = 1
                self._llm_downstream_slot_max = 1
                self._llm_sem_upstream = PrioritySemaphore(1)
                self._llm_sem_downstream = PrioritySemaphore(1)
            else:
                _r = max(1, min(mc // 4, mc - 1))
                _up = mc - _r
                self._llm_upstream_slot_max = _up
                self._llm_downstream_slot_max = _r
                self._llm_sem_upstream = PrioritySemaphore(_up)
                self._llm_sem_downstream = PrioritySemaphore(_r)
            self._llm_semaphore = self._llm_sem_upstream
        elif mc >= 1:
            # 未启用对齐专用通道：与旧版相同，从 max_llm_concurrency 总数拆分
            if mc == 1:
                self._llm_upstream_slot_max = 1
                self._llm_sem_upstream = PrioritySemaphore(1)
                self._llm_semaphore = self._llm_sem_upstream
            else:
                _r = max(1, min(mc // 4, mc - 1))
                _up = mc - _r
                self._llm_upstream_slot_max = _up
                self._llm_downstream_slot_max = _r
                self._llm_sem_upstream = PrioritySemaphore(_up)
                self._llm_sem_downstream = PrioritySemaphore(_r)
                self._llm_semaphore = self._llm_sem_upstream
        # 线程局部变量：当前 LLM 调用优先级
        self._priority_local = threading.local()

        # 蒸馏数据保存
        self._distill_data_dir = distill_data_dir
        self._distill_task_id = None  # task_id 由 step1 生成，全局共享
        self._distill_lock = threading.Lock()
        # 线程局部变量：distill step（step6/step7 并行线程各自独立）
        self._distill_local = threading.local()

    @property
    def _current_distill_step(self) -> Optional[str]:
        return getattr(self._distill_local, 'step', None)

    @_current_distill_step.setter
    def _current_distill_step(self, value: Optional[str]):
        self._distill_local.step = value

    @staticmethod
    def _ollama_root_from(base: Optional[str]) -> str:
        """将 base_url 规范化为 Ollama 根地址（不含 /v1），供 /api/chat 使用。"""
        b = (base or "http://localhost:11434").rstrip("/")
        if b.endswith("/v1"):
            b = b[:-3]
        return b

    def _in_alignment_phase(self, priority: int) -> bool:
        return priority >= LLM_PRIORITY_STEP6

    def _use_alignment_llm_endpoint(self, priority: int) -> bool:
        """是否对本次请求使用对齐专用 LLM 配置（需显式开启 alignment_enabled）。"""
        return bool(self.alignment_enabled) and self._in_alignment_phase(priority)

    def _effective_base_url(self, priority: int) -> Optional[str]:
        if self._use_alignment_llm_endpoint(priority) and self.alignment_base_url:
            return self.alignment_base_url
        return self.base_url

    def _effective_api_key(self, priority: int) -> Optional[str]:
        if self._use_alignment_llm_endpoint(priority) and self.alignment_api_key is not None:
            return self.alignment_api_key
        return self.api_key

    def _effective_model(self, priority: int) -> str:
        if self._use_alignment_llm_endpoint(priority) and self.alignment_model:
            return self.alignment_model
        return self.model_name

    def _effective_think_mode(self, priority: int) -> bool:
        if self._use_alignment_llm_endpoint(priority) and self.alignment_think_mode is not None:
            return bool(self.alignment_think_mode)
        return bool(self.think_mode)

    def _effective_max_tokens_base(self, priority: int) -> Optional[int]:
        if self._use_alignment_llm_endpoint(priority) and self.alignment_max_tokens is not None:
            return int(self.alignment_max_tokens)
        if self.max_tokens is not None:
            return int(self.max_tokens)
        return None

    @staticmethod
    def _estimate_text_token_count(text: Any) -> int:
        """保守估算 token 数。

        这里不追求精确 tokenizer 一致性，只需要在请求前避免总预算超过 8K。
        对中文与 JSON 来说，字符数近似 token 数，适合做服务保护上限。
        """
        if text is None:
            return 0
        if not isinstance(text, str):
            text = str(text)
        return len(text)

    def _estimate_messages_token_count(self, messages: List[Dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += 8  # role / 分隔符等固定开销
            total += self._estimate_text_token_count(msg.get("role", ""))
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    # Fast length estimation — avoid json.dumps per part
                    if isinstance(part, dict):
                        part_len = sum(len(v) for v in part.values() if isinstance(v, str))
                    elif isinstance(part, str):
                        part_len = len(part)
                    else:
                        part_len = len(str(part))
                    total += part_len  # _estimate_text_token_count uses len() ≈ tokens
            else:
                total += self._estimate_text_token_count(content)
        return total + 16  # 请求包尾部保留固定开销

    def _prepare_episode_for_prompt(self, episode: Optional[Episode]) -> str:
        """将记忆缓存裁剪到 prompt 可接受长度，避免异常膨胀拖爆上下文。"""
        content = ""
        if episode is not None:
            content = getattr(episode, "content", "") or ""
        limit = self.prompt_episode_max_chars
        if limit is None or limit <= 0 or len(content) <= limit:
            return content

        marker = "\n\n...[记忆缓存过长，已截断]...\n\n"
        if limit <= len(marker) + 32:
            trimmed = content[:limit]
        else:
            head = int((limit - len(marker)) * 0.7)
            tail = max(0, limit - len(marker) - head)
            trimmed = content[:head] + marker
            if tail > 0:
                trimmed += content[-tail:]
        wprint_info(
            f"[DeepDream] 记忆缓存过长：{len(content)} 字符，"
            f"已截断为 {len(trimmed)} 字符后再注入抽取 prompt"
        )
        return trimmed

    def _can_continue_multi_round(
        self,
        messages: List[Dict[str, Any]],
        *,
        next_user_content: str,
        stage_label: str,
    ) -> bool:
        """续轮前先做预算预检；若已无法容纳下一轮请求，则直接正常停止。"""
        # Estimate tokens without copying the messages list
        existing_tokens = self._estimate_messages_token_count(messages)
        next_user_msg = {"role": "user", "content": next_user_content}
        extra_tokens = 8 + self._estimate_text_token_count(next_user_content)
        total_estimated = existing_tokens + extra_tokens
        if total_estimated >= self.context_window_tokens:
            wprint_info(
                f"[DeepDream] {stage_label} 多轮预检停止：下一轮估算输入约 {total_estimated} tokens，"
                f"已触达输入上限 {self.context_window_tokens}"
            )
            return False
        return True

    @staticmethod
    def _error_suggests_context_overflow(err: BaseException) -> bool:
        """服务端错误是否与上下文/token/长度相关（仅此类错误才转储完整 messages）。"""
        sc = getattr(err, "status_code", None)
        if sc == 413:
            return True
        # Fast path: check primary error string before building full concat
        primary = str(err).lower()
        if any(n in primary for n in _CONTEXT_OVERFLOW_NEEDLES):
            return True
        # Slower path: check repr and nested body/response
        chunks: List[str] = [repr(err)]
        body = getattr(err, "body", None)
        if body is not None:
            chunks.append(str(body))
        response = getattr(err, "response", None)
        if response is not None:
            text = getattr(response, "text", None)
            if text:
                chunks.append(str(text)[:4000])
        s = "\n".join(chunks).lower()
        return any(n in s for n in _CONTEXT_OVERFLOW_NEEDLES)

    def _resolve_request_max_tokens(
        self,
        messages: List[Dict[str, Any]],
        desired_max_tokens: int,
    ) -> int:
        """仅预检输入 prompt 是否超限；输出上限按期望值直接传给模型。"""
        context_cap = self.context_window_tokens
        prompt_tokens = self._estimate_messages_token_count(messages)
        if prompt_tokens >= context_cap:
            wprint_info(
                f"[DeepDream] 输入上下文超限: 估算输入 tokens: {prompt_tokens}, "
                f"输入上限: {context_cap}, 期望输出上限: {desired_max_tokens}, "
                f"消息条数: {len(messages)}"
            )
            raise LLMContextBudgetExceeded(
                f"LLM 输入上下文超限：估算输入约 {prompt_tokens} tokens，"
                f"已达到或超过输入上限 {context_cap}。请缩短输入、减少多轮历史，"
                "或下调窗口大小 / 提示长度。"
            )
        return max(1, int(desired_max_tokens))

    def effective_entity_snippet_length(self) -> int:
        """按当前线程优先级返回实体 content 截断长度（步骤6–7 可走 alignment 配置）。"""
        p = getattr(self._priority_local, "priority", LLM_PRIORITY_STEP1)
        if self._use_alignment_llm_endpoint(p) and self.alignment_content_snippet_length is not None:
            return int(self.alignment_content_snippet_length)
        return int(self.content_snippet_length or 50)

    def _use_openai_compatible_url(self, url: Optional[str], api_key: Optional[str]) -> bool:
        """是否为 OpenAI 兼容接口；url / api_key 为本次请求实际使用的值。"""
        key = api_key
        eff = url if url is not None else self.base_url
        if not key or not eff:
            return False
        u = (eff or "").rstrip("/").lower()
        # 约定：api_key=ollama 表示使用 Ollama（即使是远端 /v1）
        if (key or "").strip().lower() == "ollama":
            return False
        # 本地 Ollama 默认端口：一律走 Ollama /api/chat，不走 /v1/chat/completions
        if ":11434" in u and ("127.0.0.1" in u or "localhost" in u):
            return False
        if "open.bigmodel.cn" in u or "bigmodel.cn" in u:
            return True
        if "openai.com" in u or "api.openai.com" in u:
            return True
        if u.endswith("/v4") or u.endswith("/v1"):
            return True
        return False

    def _is_valid_utf8(self, text: str) -> bool:
        """检测文本是否包含 Unicode 替换字符（乱码标志）。

        Python str 始终是有效 Unicode，无需 encode/decode 往返。
        仅需检测 \\ufffd 替换字符（编码错误的标志）。
        """
        if not text:
            return True
        # Unicode 替换字符是编码错误的标志
        return '\ufffd' not in text

    def _select_llm_semaphore(self, priority: int) -> Optional[PrioritySemaphore]:
        """步骤1–5 用上游池，步骤6–7 用下游池；未拆分（单槽）时仅上游。"""
        if self._llm_sem_upstream is None:
            return None
        if self._llm_sem_downstream is None:
            return self._llm_sem_upstream
        if priority >= LLM_PRIORITY_STEP6:
            return self._llm_sem_downstream
        return self._llm_sem_upstream

    def get_llm_semaphore_active_count(self) -> int:
        u = self._llm_sem_upstream.active_count if self._llm_sem_upstream else 0
        d = self._llm_sem_downstream.active_count if self._llm_sem_downstream else 0
        return u + d

    def get_llm_semaphore_max(self) -> int:
        u = self._llm_upstream_slot_max or 0
        d = self._llm_downstream_slot_max or 0
        if u or d:
            return u + d
        return self._max_llm_concurrency

    def _save_distill_conversation(self, messages: List[Dict[str, str]]):
        """保存一次 LLM 对话到 JSONL 文件（OpenAI fine-tuning 格式）。"""
        if not self._distill_data_dir or not self._current_distill_step or not self._distill_task_id:
            return
        step_dir = os.path.join(self._distill_data_dir, self._current_distill_step)
        os.makedirs(step_dir, exist_ok=True)
        filepath = os.path.join(step_dir, f"{self._distill_task_id}.jsonl")
        line = json.dumps({"messages": messages}, ensure_ascii=False)
        try:
            with self._distill_lock:
                with open(filepath, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError:
            pass

    def call_llm_until_json_parses(
        self,
        messages: List[Dict[str, str]],
        *,
        parse_fn: Callable[[str], Any],
        json_parse_retries: int = 2,
        timeout: int = 300,
        allow_mock_fallback: bool = True,
        json_retry_user_message: Optional[str] = None,
    ) -> Tuple[Any, str]:
        """
        调用 LLM，若 parse_fn(response) 因非法 JSON 抛出 json.JSONDecodeError，则追加纠错提示后重试。

        用于模型偶发输出非 JSON、截断残留、或夹杂说明文字等情况；不计入 _call_llm 的网络退避重试次数。

        Args:
            json_retry_user_message: 解析失败时追加的用户纠错句；默认使用通用「必须以 [ 或 { 开头结尾」提示。
        """
        max_attempts = 1 + max(0, int(json_parse_retries))
        last_response = ""
        last_err: Optional[BaseException] = None

        def _looks_like_truncation_json_err(err: BaseException) -> bool:
            s = str(err)
            return any(x in s for x in _TRUNCATION_KEYWORDS)

        for attempt in range(max_attempts):
            # 解析重试时若疑似截断，临时提高 max_tokens，减轻超大实体列表被截断
            scale = 1.0
            if attempt > 0 and last_err is not None and _looks_like_truncation_json_err(last_err):
                scale = min(16.0, 2.0 ** attempt)

            last_response = self._call_llm(
                "",
                messages=messages,
                timeout=timeout,
                allow_mock_fallback=allow_mock_fallback,
                request_max_tokens_scale=scale,
            )
            try:
                return parse_fn(last_response), last_response
            except json.JSONDecodeError as e:
                last_err = e
                if attempt >= max_attempts - 1:
                    wprint_info(
                        f"[DeepDream] JSON 解析失败，已达最大重试次数（{max_attempts}）: {e}"
                    )
                    raise
                wprint_info(
                    f"[DeepDream] JSON 解析失败，将重试 LLM（{attempt + 2}/{max_attempts}）: {e}"
                )
                is_truncation = _looks_like_truncation_json_err(e)
                if is_truncation and len(last_response) > 2000:
                    # 截断时响应可能达 20K+ 字符，完整追加会导致上下文溢出
                    # 仅追加前 1500 字符 + 截断标记
                    truncated_summary = (
                        last_response[:1500]
                        + "\n... [输出被截断，共 "
                        + str(len(last_response))
                        + " 字符，仅保留前 1500 字符] ..."
                    )
                    messages.append({"role": "assistant", "content": truncated_summary})
                else:
                    messages.append({"role": "assistant", "content": last_response})
                base_retry = json_retry_user_message or _JSON_RETRY_USER_MESSAGE
                retry_hint = base_retry
                if is_truncation:
                    retry_hint = base_retry + _JSON_RETRY_TRUNCATION_SUFFIX
                messages.append({"role": "user", "content": retry_hint})
                time.sleep(0.3)
        raise last_err if last_err else RuntimeError("call_llm_until_json_parses: unreachable")

    def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        timeout: Optional[int] = None,
        allow_mock_fallback: bool = True,
        messages: Optional[List[Dict[str, str]]] = None,
        *,
        request_max_tokens_scale: float = 1.0,
    ) -> str:
        """
        调用LLM的通用方法（带重试机制）

        Args:
            prompt: 用户提示（messages 为 None 时使用）
            system_prompt: 系统提示（可选）
            max_retries: 兼容保留；普通 API 错误固定为最多 5 轮退避重试（3^1…3^5 秒等待）。
            timeout: 超时时间（秒），默认使用 self.timeout_seconds（300秒）
            allow_mock_fallback: 失败时是否降级为模拟响应；启动握手等场景应传 False，避免误判为可用
            messages: 完整对话列表（可选）；传入时直接使用，忽略 prompt 和 system_prompt
            request_max_tokens_scale: 仅缩放本次请求的 max_tokens/num_predict（供 JSON 解析重试时临时放大上限）

        Returns:
            LLM的响应文本；allow_mock_fallback=False 且失败时返回空字符串
        """
        # Use configured timeout if not explicitly provided
        if timeout is None:
            timeout = self.timeout_seconds
        if not self._endpoint_available:
            if allow_mock_fallback:
                mock_prompt = (messages[-1]["content"] if messages else prompt) if messages else prompt
                return self._mock_llm_response(mock_prompt)
            return ""

        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        
        last_error = None
        _utf8_round = 0
        _normal_failures = 0
        _conn_failures = 0
        _tpm_round = 0
        _detailed_error_logged = False
        _priority_init = getattr(self._priority_local, "priority", LLM_PRIORITY_STEP7)
        _mt0 = self._effective_max_tokens_base(_priority_init)
        _effective_max_tokens = _mt0 if _mt0 is not None else 4096
        # Resolve LLM endpoint config once (doesn't change between retries)
        _eff_base = self._effective_base_url(_priority_init)
        _eff_key = self._effective_api_key(_priority_init)
        _eff_model = self._effective_model(_priority_init)
        _eff_think = self._effective_think_mode(_priority_init)
        _sem = self._select_llm_semaphore(_priority_init)
        while True:
            # 获取并发信号量（按优先级排队等待；上游/下游分池）
            _sem_held = False
            if _sem is not None:
                _sem.acquire(_priority_init)
                _sem_held = True
            try:
                _scale = max(0.25, float(request_max_tokens_scale or 1.0))
                _desired_max_tokens = max(1, int(_effective_max_tokens * _scale))
                _api_max_tokens = self._resolve_request_max_tokens(messages, _desired_max_tokens)

                if self._use_openai_compatible_url(_eff_base, _eff_key):
                    _bu = (_eff_base or "").rstrip("/")
                    resp = openai_compatible_chat(
                        messages,
                        model=_eff_model,
                        base_url=_bu,
                        api_key=_eff_key,
                        timeout=timeout,
                        max_tokens=_api_max_tokens,
                    )
                else:
                    resp = ollama_chat(
                        messages,
                        model=_eff_model,
                        base_url=self._ollama_root_from(_eff_base),
                        think=_eff_think,
                        timeout=timeout,
                        num_predict=_api_max_tokens,
                    )
                response_text = resp.content or ""
                # 已成功完成一次上游 HTTP 调用：清零各类失败计数（UTF-8 轮次单独计）
                _normal_failures = 0
                _conn_failures = 0
                _tpm_round = 0

                # 检测 LLM 输出被 max_tokens 截断（finish_reason/done_reason == "length"）
                _is_truncated = (
                    getattr(resp, "done_reason", None) == "length"
                    or (resp.raw and resp.raw.get("choices") and
                        resp.raw["choices"][0].get("finish_reason") == "length")
                )
                if _is_truncated:
                    _est_input = self._estimate_messages_token_count(messages)
                    wprint_info(
                        "[DeepDream] LLM 输出被截断（finish_reason=length）。"
                        f"当前请求输出上限为 {_api_max_tokens}，已不再自动扩容重试；"
                        "如需避免截断，请缩短输入上下文或减少输出体积。"
                    )
                    wprint_info(
                        f"[DeepDream] 截断摘要: 估算输入 tokens: {_est_input}, "
                        f"期望输出上限: {_desired_max_tokens}, "
                        f"实际输出上限: {_api_max_tokens}, "
                        f"输入上限: {self.context_window_tokens}, "
                        f"消息条数: {len(messages)}, "
                        f"输出长度: {len(response_text)} 字符"
                    )

                # 检测是否是有效的UTF-8编码
                if not self._is_valid_utf8(response_text):
                    _utf8_round += 1
                    if _utf8_round <= _LLM_MAX_FAILURE_ROUNDS:
                        wprint_info(f"检测到非UTF-8编码的文本，正在重新生成（第 {_utf8_round}/{_LLM_MAX_FAILURE_ROUNDS} 次尝试）...")
                        wprint_info(f"问题内容预览:\n{response_text}")
                        continue
                    else:
                        wprint_info(f"警告：检测到非UTF-8编码但已达到最大重试次数，返回原始响应")
                        wprint_info(f"问题内容预览:\n{response_text}")

                # 编码有效则返回响应（已取消乱码检测）
                # 蒸馏数据保存（步骤2/3走多轮手动保存，在此跳过）
                if (response_text and self._current_distill_step
                        and self._current_distill_step not in _DISTILL_SKIP_STEPS):
                    self._save_distill_conversation(
                        messages + [{"role": "assistant", "content": response_text}]
                    )
                # 清理弱模型可能回显的 XML 分隔符标签（<记忆缓存>、<输入文本> 等）
                return clean_separator_tags(response_text)

            except Exception as e:
                # 统一处理错误，包括连接错误、超时等
                error_str = str(e).lower()
                last_error = e
                is_timeout = "timeout" in error_str or "timed out" in error_str
                is_fd_error = (
                    isinstance(e, OSError) and getattr(e, "errno", None) == 24
                ) or "too many open files" in error_str or "errno 24" in error_str
                is_connection_error = any(
                    kw in error_str
                    for kw in _CONNECTION_ERROR_KEYWORDS
                ) or is_fd_error
                is_tpm_error = _is_rate_limit_tpm_error(e, _pre_lowered=error_str)

                if not _detailed_error_logged and not is_connection_error and not is_timeout and not is_tpm_error:
                    if self._error_suggests_context_overflow(e):
                        wprint_info(
                            "[DeepDream] 服务端报上下文/长度相关错误: "
                            f"估算输入 tokens: {self._estimate_messages_token_count(messages)}, "
                            f"期望输出上限: {_effective_max_tokens}, "
                            f"消息条数: {len(messages)}, "
                            f"错误: {e}"
                        )
                    _detailed_error_logged = True

                if "上下文预算超限" in error_str or "输入上下文超限" in error_str:
                    wprint_info(str(e))
                    raise

                # max_tokens 超限：自动降低重试（不计入退避轮次）
                if "max_tokens" in error_str or "max_completion_tokens" in error_str or "too large" in error_str:
                    if _effective_max_tokens and _effective_max_tokens > 1:
                        _effective_max_tokens = _effective_max_tokens // 2
                        wprint_info(f"[DeepDream] max_tokens 超限，自动降至 {_effective_max_tokens} 后重试")
                        if _sem is not None:
                            _sem.release()
                        _sem_held = False
                        time.sleep(0.5)
                        continue

                # 429 / TPM / 速率限制：视为可恢复，指数退避直至成功，不限制重试次数
                if is_tpm_error:
                    _tpm_round += 1
                    wait_seconds = min(
                        _LLM_BACKOFF_BASE ** min(_tpm_round, 12),
                        _LLM_TPM_SLEEP_CAP_SECONDS,
                    )
                    wprint_info(
                        f"LLM 速率限制（TPM/429），{wait_seconds}s 后重试（不限制次数，第 {_tpm_round} 次等待）: {e}"
                    )
                    if _sem is not None:
                        _sem.release()
                    _sem_held = False
                    time.sleep(wait_seconds)
                    continue

                # 连接错误：最多 5 轮，等待 3^n 秒
                if is_connection_error:
                    _conn_failures += 1
                    if _conn_failures <= _LLM_MAX_FAILURE_ROUNDS:
                        wait_seconds = _LLM_BACKOFF_BASE ** _conn_failures
                        wprint_info(f"LLM连接错误（第 {_conn_failures}/{_LLM_MAX_FAILURE_ROUNDS} 次失败）: {e}")
                        wprint_info(f"{wait_seconds} 秒后重试...")
                        if _sem is not None:
                            _sem.release()
                        _sem_held = False
                        time.sleep(wait_seconds)
                        continue
                    wprint_info(f"LLM连接错误已达 {_LLM_MAX_FAILURE_ROUNDS} 轮，放弃重试: {e}")
                    if _sem is not None:
                        _sem.release()
                    _sem_held = False
                    raise

                # 其它错误（含超时）：最多 5 轮，等待 3^n 秒
                _normal_failures += 1
                if _normal_failures <= _LLM_MAX_FAILURE_ROUNDS:
                    wait_seconds = _LLM_BACKOFF_BASE ** _normal_failures
                    if is_timeout:
                        wprint_info(f"LLM调用超时（第 {_normal_failures}/{_LLM_MAX_FAILURE_ROUNDS} 次失败，超时: {timeout}s）: {e}")
                    else:
                        wprint_info(f"LLM调用错误（第 {_normal_failures}/{_LLM_MAX_FAILURE_ROUNDS} 次失败）: {e}")
                    wprint_info(f"{wait_seconds} 秒后重试...")
                    if _sem is not None:
                        _sem.release()
                    _sem_held = False
                    time.sleep(wait_seconds)
                    continue

                if is_timeout:
                    wprint_info(f"LLM调用超时（已达 {_LLM_MAX_FAILURE_ROUNDS} 轮重试，超时时间: {timeout}秒）: {e}")
                else:
                    wprint_info(f"LLM调用错误（已达 {_LLM_MAX_FAILURE_ROUNDS} 轮重试）: {e}")
                if _sem is not None:
                    _sem.release()
                _sem_held = False
                if allow_mock_fallback:
                    return self._mock_llm_response(prompt)
                return ""
            finally:
                if _sem is not None and _sem_held:
                    _sem.release()

        # 理论上不会到达这里，但为了稳妥保留兜底
        if last_error:
            wprint_info("所有重试都失败，使用模拟响应")
        if allow_mock_fallback:
            return self._mock_llm_response(prompt)
        return ""
    
    def _clean_json_string(self, json_str: str) -> str:
        """
        清理JSON字符串，修复常见错误
        
        Args:
            json_str: 原始JSON字符串
            
        Returns:
            清理后的JSON字符串
        """
        # 移除BOM标记
        json_str = json_str.lstrip('\ufeff')
        # 移除首尾空白
        json_str = json_str.strip()
        # 修复中文标点符号（fast path: skip when no CJK punctuation present）
        if _CJK_PUNCT_RE.search(json_str):
            json_str = _CJK_PUNCT_RE.sub(lambda m: _CJK_PUNCT_MAP[m.group()], json_str)
        # 注意：中文弯引号 \u201c \u201d 经常出现在 JSON 字符串值内部（如 "研制"九章"…"）
        # 不能全局替换为 ASCII "，否则会破坏 JSON 结构。
        # 它们是合法 UTF-8 字符，可直接保留。
        # 移除可能的尾随逗号（在数组或对象的最后一个元素后）
        json_str = _TRAILING_COMMA_RE.sub(r'\1', json_str)
        return json_str
    
    def _fix_json_errors(self, json_str: str) -> str:
        """
        尝试修复JSON错误

        Args:
            json_str: 有错误的JSON字符串 (assumed already _clean_json_string'd by caller)

        Returns:
            修复后的JSON字符串
        """
        # Note: caller (_parse_json_response) already applied _clean_json_string,
        # so we skip it here to avoid redundant work.

        # Fix invalid Unicode escape sequences
        json_str = _fix_unicode_escapes(json_str)

        # Escape bare control characters inside JSON string values
        json_str = _escape_control_chars_in_json_strings(json_str)

        return json_str

    def _parse_json_response(self, response: str) -> Any:
        """从 LLM 响应中提取并解析 JSON。"""
        json_str = response or ""
        # Single-pass fence extraction using regex (replaces 4 find() calls)
        fence_match = _JSON_FENCE_RE.search(json_str)
        if fence_match:
            json_start = fence_match.end()
            json_end = json_str.find("```", json_start)
            if json_end == -1:
                wprint_info("[DeepDream] 警告: LLM 响应的 ```json 块未闭合，JSON 可能被截断")
                json_str = json_str[json_start:].strip()
            else:
                json_str = json_str[json_start:json_end].strip()

        json_str = self._clean_json_string(json_str)

        # 截断检测：检查 JSON 结构是否完整（cache stripped versions for reuse in except）
        _ls = json_str.lstrip()
        _first_char = _ls[0] if _ls else ''
        if _first_char in ('[', '{'):
            close_char = ']' if _first_char == '[' else '}'
            _rs = json_str.rstrip()
            _last_char = _rs[-1] if _rs else ''
            if _last_char != close_char:
                wprint_info(f"[DeepDream] 警告: LLM 响应 JSON 被截断，以 {_first_char} 开头但不以 {close_char} 结尾。"
                      f"请缩短输入上下文或输出内容。响应前200字符: {json_str[:200]}")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            fixed = self._fix_json_errors(json_str)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                # Reuse _ls/_rs from above (avoid recomputing strip)
                _rs_cached = _rs if _first_char in ('[', '{') else json_str.rstrip()
                _is_complete = (
                    (_first_char == '[' and _rs_cached.endswith(']')) or
                    (_first_char == '{' and _rs_cached.endswith('}'))
                )
                if not _is_complete:
                    # Try array truncation repair
                    repaired = self._try_repair_truncated_json_array(json_str)
                    if repaired is not None:
                        try:
                            parsed = json.loads(repaired)
                            wprint_info(
                                "[DeepDream] 警告: 检测到数组型 JSON 尾部截断；"
                                "已裁剪不完整尾部并补全 `]`，沿用可恢复部分。"
                            )
                            return parsed
                        except json.JSONDecodeError:
                            pass
                    # Try object truncation repair
                    repaired_obj = self._try_repair_truncated_json_object(json_str)
                    if repaired_obj is not None:
                        try:
                            parsed = json.loads(repaired_obj)
                            wprint_info(
                                "[DeepDream] 警告: 检测到对象型 JSON 尾部截断；"
                                "已裁剪不完整键值对并补全 `}`，沿用可恢复部分。"
                            )
                            return parsed
                        except json.JSONDecodeError:
                            pass
                wprint_info("[DeepDream] 警告: LLM 响应 JSON 解析失败（可能被截断）。"
                      f"响应: {json_str}")
                raise

    def _try_repair_truncated_json_array(self, json_str: str) -> Optional[str]:
        """修复尾部被截断的 JSON 数组：裁掉不完整尾巴并补上 `]`。"""
        stripped = (json_str or "").strip()
        if not stripped.startswith("[") or stripped.endswith("]"):
            return None

        in_string = False
        escaped = False
        stack: List[str] = []
        last_complete_value_end: Optional[int] = None

        for idx, ch in enumerate(stripped):
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch in "[{":
                stack.append(ch)
                continue

            if ch in "]}":
                if not stack:
                    break
                opener = stack[-1]
                if (opener == "[" and ch != "]") or (opener == "{" and ch != "}"):
                    break
                stack.pop()
                if stack == ["["]:
                    last_complete_value_end = idx + 1
                elif not stack and ch == "]":
                    last_complete_value_end = idx + 1
                    break

        if last_complete_value_end is None:
            return None

        candidate = stripped[:last_complete_value_end].rstrip()
        if not candidate.startswith("["):
            return None
        candidate = candidate.rstrip(", \n\r\t") + "]"
        return candidate if candidate != stripped else None

    def _try_repair_truncated_json_object(self, json_str: str) -> Optional[str]:
        """修复尾部被截断的 JSON 对象：裁掉不完整键值对并补上 `}`。

        例如 {"action": "match", "id": "rel_xxx", "content": "很长的内容被截断...
        修复为 {"action": "match", "id": "rel_xxx"}
        """
        stripped = (json_str or "").strip()
        if not stripped.startswith("{") or stripped.endswith("}"):
            return None

        in_string = False
        escaped = False
        stack: List[str] = []
        last_complete_value_end: Optional[int] = None

        for idx, ch in enumerate(stripped):
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch in "[{":
                stack.append(ch)
                continue

            if ch in "]}":
                if not stack:
                    break
                opener = stack[-1]
                if (opener == "[" and ch != "]") or (opener == "{" and ch != "}"):
                    break
                stack.pop()
                if not stack:
                    last_complete_value_end = idx + 1
                    break
                if stack == ["{"]:
                    last_complete_value_end = idx + 1
                continue

            # Comma at top-level object: the key-value pair before it is complete
            if ch == "," and stack == ["{"]:
                last_complete_value_end = idx

        if last_complete_value_end is None:
            return None

        candidate = stripped[:last_complete_value_end].rstrip()
        if not candidate.startswith("{"):
            return None
        candidate = candidate.rstrip(", \n\r\t") + "}"
        return candidate if candidate != stripped else None

    def _mock_llm_response(self, prompt: str) -> str:
        """模拟LLM响应（用于测试）"""
        prompt_lower = prompt.lower()

        def _extract_tag_block(tag: str) -> str:
            match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", prompt, re.DOTALL)
            return match.group(1).strip() if match else ""

        def _extract_bullet_names(*tags: str) -> list[str]:
            for tag in tags:
                block = _extract_tag_block(tag)
                if not block:
                    continue
                names = []
                for line in block.splitlines():
                    line = line.strip()
                    if not line.startswith("-"):
                        continue
                    item = line[1:].strip()
                    if "|" in item:
                        item = item.split("|", 1)[0].strip()
                    if "<->" in item:
                        continue
                    if item:
                        names.append(item)
                if names:
                    return names
            return []

        def _extract_candidate_pairs() -> list[dict]:
            block = _extract_tag_block("候选概念对")
            pairs = []
            for line in block.splitlines():
                line = line.strip()
                if not line.startswith("-"):
                    continue
                item = line[1:].strip()
                hint = ""
                if "|" in item:
                    pair_part, hint_part = item.split("|", 1)
                    item = pair_part.strip()
                    hint = hint_part.replace("线索:", "").strip()
                if "<->" not in item:
                    continue
                left, right = [part.strip() for part in item.split("<->", 1)]
                if not left or not right or left == right:
                    continue
                entity1_name, entity2_name = sorted((left, right))
                pairs.append({
                    "entity1_name": entity1_name,
                    "entity2_name": entity2_name,
                    "content": hint,
                })
            return pairs

        if ("更新记忆缓存" in prompt or "memory_cache" in prompt_lower
                or "创建初始记忆缓存" in prompt or "创建初始的记忆缓存" in prompt):
            return """当前摘要：正在处理文档内容。当前阅读的是文档的开头部分，介绍了故事的基本背景和主要人物。重要细节包括主要人物的基本信息和故事的初始情境。

自我思考：
- 应该关注：主要人物的身份、性格特点、故事发生的背景环境
- 预判重点：后续情节可能围绕这些主要人物展开，需要留意人物之间的关系和故事的发展方向
- 疑虑：暂无特别疑虑，需要继续阅读以了解故事的发展

系统状态：
- 已处理文本范围：处理到"文档开始"结束
- 当前文档名：示例文档.txt"""
        elif "候选实体列表" in prompt and "match_existing_id" in prompt:
            _candidate_block = prompt.split("</当前实体>")[1] if "</当前实体>" in prompt else ""
            _current_name_match = _CURRENT_ENTITY_NAME_RE.search(prompt)
            _current_name = _current_name_match.group(1) if _current_name_match else ""
            _candidate_entries = _candidate_block.split("候选")[1:] if _candidate_block else []
            _match_id = ""
            _update_mode = "create_new"
            for _entry in _candidate_entries:
                _cid_m = _FAMILY_ID_RE.search(_entry)
                _cname_m = _ENTRY_NAME_RE.search(_entry)
                if _cid_m and _cname_m and _cname_m.group(1) == _current_name:
                    _match_id = _cid_m.group(1)
                    _update_mode = "reuse_existing"
                    break
            return _mock_json_fence({
                "match_existing_id": _match_id,
                "update_mode": _update_mode,
                "merged_name": "",
                "merged_content": "",
                "relations_to_create": [],
                "confidence": 0.9 if _match_id else 0.3,
            })
        elif ("判断.*实体.*匹配" in prompt or "judge.*entity.*match" in prompt_lower or
              "判断新抽取的实体是否与已有实体" in prompt):
            return _mock_json_fence({
                "family_id": "ent_001",
                "need_update": False
            })
        elif "<指定实体名称>" in prompt:
            names = _extract_bullet_names("指定实体名称")
            if not names:
                names = ["示例实体1"]
            return _mock_json_fence([
                {
                    "name": name,
                    "content": f"{name}在当前文本中被提及，并有一段稳定的结构化描述。"
                }
                for name in names
            ])
        elif "请召回所有结构性文本锚点概念候选" in prompt or "结构性文本锚点概念候选" in prompt:
            return _mock_json_fence([
                {"name": "第一章", "content": "文本中的结构性章节标题。"},
                {"name": "需求分析阶段", "content": "文本中明确出现的阶段性锚点。"},
            ])
        elif "请召回所有具体/具名概念候选" in prompt or "具名概念候选" in prompt:
            return _mock_json_fence([
                {"name": "示例实体1", "content": "文本中明确出现的具体概念。"},
                {"name": "示例实体2", "content": "文本中明确出现的另一具体概念。"},
            ])
        elif "请召回所有具体/具名概念候选" in prompt or "具体/具名概念候选" in prompt:
            return _mock_json_fence([
                {"name": "示例实体1", "content": "文本中明确出现的具体概念。"},
                {"name": "示例实体2", "content": "文本中明确出现的另一具体概念。"},
            ])
        elif "请召回所有抽象/过程/时间/文本锚点类概念候选" in prompt or "抽象/过程/时间/文本锚点类概念候选" in prompt:
            return _mock_json_fence([
                {"name": "示例主题", "content": "文本中的抽象主题或过程概念。"}
            ])
        elif "<已召回概念列表>" in prompt or "请只补充上面列表中明显遗漏" in prompt:
            known_names = set(_extract_bullet_names("已召回概念列表"))
            candidate = "补充概念"
            if candidate in known_names:
                return _mock_json_fence([])
            return _mock_json_fence([
                {"name": candidate, "content": "对已召回概念的补充概念。"}
            ])
        elif "<候选概念对>" in prompt and "只为候选概念对写出具体关系内容" in prompt:
            pairs = _extract_candidate_pairs()
            return _mock_json_fence([
                {
                    "entity1_name": pair["entity1_name"],
                    "entity2_name": pair["entity2_name"],
                    "content": pair.get("content") or f"{pair['entity1_name']}与{pair['entity2_name']}在文本中存在明确关联。"
                }
                for pair in pairs
            ])
        elif "<稳定概念实体列表>" in prompt and "值得建立关系的概念对" in prompt:
            names = _extract_bullet_names("稳定概念实体列表")
            if len(names) < 2:
                return _mock_json_fence([])
            pairs = []
            for left, right in zip(names, names[1:]):
                entity1_name, entity2_name = sorted((left, right))
                pairs.append({
                    "entity1_name": entity1_name,
                    "entity2_name": entity2_name,
                    "content": f"{entity1_name}与{entity2_name}之间存在关系线索"
                })
                if len(pairs) >= 2:
                    break
            return _mock_json_fence(pairs)
        elif "继续生成" in prompt or "继续补充" in prompt:
            return _mock_json_fence([])
        elif "输出格式纠错" in prompt or "json 代码块" in prompt_lower:
            return _mock_json_fence([])
        elif ("抽取关系" in prompt or "抽取所有概念实体间的关系" in prompt or
              "relation" in prompt_lower or "从输入文本中抽取实体之间的关系" in prompt or
              "关系抽取" in prompt or "实体间的关系" in prompt):
            names = _extract_bullet_names("概念实体列表", "稳定概念实体列表")
            if not names and "已抽取的实体：" in prompt:
                entities_section = prompt.split("已抽取的实体：", 1)[1].split("</已抽取实体>", 1)[0].strip()
                if not entities_section:
                    return _mock_json_fence([])
            if len(names) >= 2:
                entity1_name, entity2_name = sorted((names[0], names[1]))
            else:
                entity1_name, entity2_name = "示例实体1", "示例实体2"
            return _mock_json_fence([
                {
                    "entity1_name": entity1_name,
                    "entity2_name": entity2_name,
                    "content": f"{entity1_name}与{entity2_name}之间存在稳定关系。"
                }
            ])
        elif ("实体后验增强" in prompt or "enhance.*entity.*content" in prompt_lower or
              "对该实体的content进行更细致的补全和挖掘" in prompt or "增强后的完整实体content" in prompt):
            if "当前content：" in prompt:
                original_content = prompt.split("当前content：", 1)[1].split("</已抽取实体>", 1)[0].strip()
                enhanced_content = f"{original_content}\n\n[增强信息]：基于记忆缓存和当前文本的补充细节和上下文信息。"
            else:
                enhanced_content = "这是一个示例实体的描述\n\n[增强信息]：基于记忆缓存和当前文本的补充细节和上下文信息。"
            return _mock_json_fence({"content": enhanced_content})
        elif ("抽取实体" in prompt or "抽取所有概念实体" in prompt or "entity" in prompt_lower or
              "从输入文本中抽取所有实体" in prompt or "实体抽取" in prompt or
              "概念实体" in prompt):
            return _mock_json_fence([
                {
                    "name": "示例实体1",
                    "content": "这是一个示例实体的描述，包含足够的结构化信息。"
                }
            ])
        elif ("判断" in prompt and "合并" in prompt and "实体" in prompt) or "merge_entity_name" in prompt_lower:
            return _mock_json_fence({"merged_name": "示例实体1", "merged_content": "合并后的描述"})
        elif ("判断" in prompt and "更新" in prompt and ("content" in prompt_lower or "内容" in prompt)):
            return _mock_json_fence({"need_update": False})
        elif ("关系" in prompt and "匹配" in prompt) or "relation_match" in prompt_lower:
            return _mock_json_fence({"family_id": None})
        elif ("生成关系" in prompt or "relation_content" in prompt_lower or "关系的content" in prompt):
            return _mock_json_fence({"content": "这是一个示例关系描述"})
        elif "知识图谱整理" in prompt or "consolidation" in prompt_lower:
            return "知识图谱整理完成，未发现需要处理的重复实体。"
        elif ("整体记忆" in prompt or "document_overall" in prompt_lower or "文档整体" in prompt):
            return "# 文档整体记忆\n\n这是一份示例文档的整体描述。"
        return "默认响应"
