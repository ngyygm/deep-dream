"""
LLM客户端：封装LLM调用，实现三个核心任务。

请求方式：统一通过 `processor/ollama_chat_api.py` 访问：
- Ollama 走原生 `POST /api/chat`；
- OpenAI/GLM/LM Studio 等走 OpenAI 兼容接口。

think 模式由初始化参数 think_mode 控制；只有 Ollama 原生协议支持通过 `think: true/false` 显式开关思考模式。
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import CancelledError
import json
import os
import random
import threading
import time

from ..utils import clean_separator_tags, wprint_info
from .chat_api import ollama_chat, openai_compatible_chat
from .errors import LLMContextBudgetExceeded
from .memory_ops import _MemoryOpsMixin
from .content_merger import _ContentMergerMixin
from .consolidation import _ConsolidationMixin
from .extraction import _LLMExtractionMixin
from .extraction_strong import _StrongExtractionMixin
from .json_repair import (
    clean_json_string,
    fix_json_errors,
    parse_json_response,
    _TRUNCATION_KEYWORDS,
    _JSON_RETRY_USER_MESSAGE,
    _JSON_RETRY_TRUNCATION_SUFFIX,
)
from .mock_response import mock_llm_response
from .prompts import (
    _LLM_BACKOFF_BASE,
    _LLM_BACKOFF_SCHEDULE,
    _LLM_MAX_FAILURE_ROUNDS,
    _XINFERENCE_500_BACKOFF,
    _XINFERENCE_500_MAX_RETRIES,
    _XINFERENCE_500_JITTER_MAX,
    _LLM_TPM_SLEEP_CAP_SECONDS,
    _CONNECTION_ERROR_KEYWORDS,
    LLM_PRIORITY_ALIGN,
    LLM_PRIORITY_EXTRACT,
    estimate_tokens,
    estimate_messages_token_count,
    error_suggests_context_overflow,
    ollama_root_from,
    is_valid_utf8,
)

try:
    from openai import RateLimitError
except ImportError:  # pragma: no cover
    RateLimitError = None  # type: ignore[misc,assignment]

# Static error keywords — computed once at import time, not per-call
_RATE_LIMIT_KEYWORDS = ("rate_limit", "rate limit", "tpm", "throttl", "capacity", "overloaded")


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


class SharedLLMSemaphore:
    """FIFO LLM 并发闸门 + active/max 监控计数。

    优先级已降级为纯路由标签（LLM_PRIORITY_*），只决定请求走哪个端点，
    不再参与排队顺序；acquire 的 priority 参数仅为调用点兼容而保留（忽略）。
    """

    def __init__(self, value: int):
        v = max(1, int(value))
        self._sem = threading.Semaphore(v)
        self._max_value = v
        self._active_lock = threading.Lock()
        self._active = 0

    @property
    def active_count(self) -> int:
        with self._active_lock:
            return self._active

    @property
    def max_value(self) -> int:
        return self._max_value

    def acquire(self, priority: int = 0):
        self._sem.acquire()
        with self._active_lock:
            self._active += 1

    def release(self):
        with self._active_lock:
            self._active = max(0, self._active - 1)
        self._sem.release()


class LLMCallStats:
    """线程安全的 LLM 调用/token 计数器；可被同一 processor 的多个 LLMClient 共享。

    优先记录上游 usage（prompt_eval_count/eval_count）；缺失时回退为本地估算并计 estimated。
    按调用发生时的 distill step 标签分桶（未打标的归 "unlabeled"），供管线各阶段对比。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counters = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_calls": 0,
        }
        self._by_step: Dict[str, Dict[str, int]] = {}
        self._by_model: Dict[str, int] = {}

    def record(self, *, model: str, step: str, prompt_tokens: int, completion_tokens: int,
               estimated: bool) -> None:
        with self._lock:
            self._counters["calls"] += 1
            self._counters["prompt_tokens"] += max(0, int(prompt_tokens))
            self._counters["completion_tokens"] += max(0, int(completion_tokens))
            if estimated:
                self._counters["estimated_calls"] += 1
            bucket = self._by_step.setdefault(
                step, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            )
            bucket["calls"] += 1
            bucket["prompt_tokens"] += max(0, int(prompt_tokens))
            bucket["completion_tokens"] += max(0, int(completion_tokens))
            self._by_model[str(model or "?")] = self._by_model.get(str(model or "?"), 0) + 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            out = dict(self._counters)
            out["by_step"] = {
                k: dict(v) for k, v in sorted(self._by_step.items())
            }
            out["by_model"] = dict(self._by_model)
            return out

    def reset(self) -> None:
        with self._lock:
            for k in self._counters:
                self._counters[k] = 0
            self._by_step.clear()
            self._by_model.clear()


class LLMClient(_MemoryOpsMixin, _ContentMergerMixin, _ConsolidationMixin,
                 _StrongExtractionMixin, _LLMExtractionMixin):

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
                 shared_llm_semaphore: Optional[SharedLLMSemaphore] = None,
                 shared_llm_slot_max: Optional[int] = None,
                 openai_extra_body: Optional[Dict[str, Any]] = None,
                 alignment_base_url: Optional[str] = None,
                 alignment_api_key: Optional[str] = None,
                 alignment_model: Optional[str] = None,
                 alignment_max_tokens: Optional[int] = None,
                 alignment_think_mode: Optional[bool] = None,
                 alignment_content_snippet_length: Optional[int] = None,
                 alignment_relation_content_snippet_length: Optional[int] = None,
                 alignment_enabled: bool = False,
                 alignment_openai_extra_body: Optional[Dict[str, Any]] = None,
                 call_stats: Optional[LLMCallStats] = None,
                 judge_service: Optional[Any] = None):
        """
        初始化LLM客户端

        Args:
            api_key / model_name / base_url / content_snippet_length / relation_content_snippet_length / think_mode / max_tokens / context_window_tokens:
                LLM 基础配置；max_llm_concurrency 是当前服务的全局 LLM 请求并发上限。
                context_window_tokens：请求输入 prompt 的 token 预算上限；本地仅预检输入，不再用它压缩输出 max_tokens。
            timeout_seconds / connect_timeout_seconds:
                LLM API 请求超时配置（秒）。timeout_seconds 控制总请求超时（默认 300），connect_timeout_seconds 控制连接超时（默认 30）。
            alignment_enabled:
                False 时忽略所有 alignment_*，步骤 6/7 与上游共用同一模型配置。
            shared_llm_semaphore / shared_llm_slot_max:
                多个 LLMClient 共享同一个全局并发闸门。Deep-Dream 的 remember 流水线用它把
                主模型、抽取模型、对齐模型统一限制在 llm.max_concurrency 内。
            alignment_*:
                步骤 6–7 可单独覆盖；未设置的项回退到上游对应项。
            call_stats:
                共享的调用计数器；同一 processor 的多个 LLMClient 传同一实例即可汇总。
        """
        self._call_stats = call_stats if call_stats is not None else LLMCallStats()
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
        self.openai_extra_body = dict(openai_extra_body or {})
        if context_window_tokens is None:
            raise ValueError(
                "context_window_tokens 未设置。请在 service_config.json 的 llm 中配置 context_window_tokens，"
                "并由 TemporalMemoryGraphProcessor 传入 LLMClient。"
            )
        self.context_window_tokens = max(256, int(context_window_tokens))
        # Timeout configuration for LLM requests
        self.timeout_seconds = max(10, int(timeout_seconds)) if timeout_seconds is not None else 300
        self.connect_timeout_seconds = max(5, int(connect_timeout_seconds)) if connect_timeout_seconds is not None else 30

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
        self.alignment_openai_extra_body = dict(alignment_openai_extra_body or {})

        # 统一使用 Python SDK（openai>=1.0）访问；任一端点有 api/base 则非模拟模式
        self._endpoint_available = bool(
            api_key or base_url or self.alignment_base_url or (self.alignment_api_key is not None)
        )
        if not self._endpoint_available:
            wprint_info("提示：未提供 API key 或任一 base_url，将使用模拟响应模式")

        # LLM 并发：一个全局闸门代表当前 LLM 服务的真实容量。
        # 旧版曾拆成 upstream/downstream 两池；这会让 extraction/alignment 双客户端在同一服务上超发。
        # 全局单一 FIFO 闸门（SharedLLMSemaphore）；upstream 命名仅为监控兼容。
        self._max_llm_concurrency: int = max_llm_concurrency or 0
        self._llm_upstream_slot_max: int = 0
        self._llm_sem_upstream: Optional[SharedLLMSemaphore] = None
        self._llm_semaphore: Optional[SharedLLMSemaphore] = None  # 兼容旧代码/测试：与上游相同或总池
        mc = max_llm_concurrency or 0
        if shared_llm_semaphore is not None:
            self._llm_sem_upstream = shared_llm_semaphore
            self._llm_semaphore = shared_llm_semaphore
            self._llm_upstream_slot_max = max(1, int(shared_llm_slot_max or shared_llm_semaphore.max_value))
            self._max_llm_concurrency = self._llm_upstream_slot_max
        elif mc >= 1:
            self._llm_upstream_slot_max = int(mc)
            self._llm_sem_upstream = SharedLLMSemaphore(self._llm_upstream_slot_max)
            self._llm_semaphore = self._llm_sem_upstream
        # 线程局部变量：当前 LLM 调用优先级
        self._priority_local = threading.local()

        # 取消检查：由 pipeline 设置，在 LLM 重试循环中调用
        self._cancel_check_fn = None

        # 蒸馏数据保存
        self._distill_data_dir = distill_data_dir
        self._distill_task_id = None  # task_id 由 step1 生成，全局共享
        self._distill_lock = threading.Lock()
        # 线程局部变量：distill step（step9/step10 并行线程各自独立）
        self._distill_local = threading.local()

    @property
    def _current_distill_step(self) -> Optional[str]:
        return getattr(self._distill_local, 'step', None)

    @_current_distill_step.setter
    def _current_distill_step(self, value: Optional[str]):
        self._distill_local.step = value

    @staticmethod
    def _ollama_root_from(base: Optional[str]) -> str:
        return ollama_root_from(base)

    def set_cancel_check(self, fn):
        """设置取消检查回调（返回 True 表示应取消）。"""
        self._cancel_check_fn = fn

    def clear_cancel_check(self):
        """清除取消检查回调。"""
        self._cancel_check_fn = None

    def _in_alignment_phase(self, priority: int) -> bool:
        return priority >= LLM_PRIORITY_ALIGN

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

    def _effective_openai_extra_body(self, priority: int) -> Dict[str, Any]:
        if self._use_alignment_llm_endpoint(priority) and self.alignment_openai_extra_body:
            return self.alignment_openai_extra_body
        return self.openai_extra_body

    def _effective_max_tokens_base(self, priority: int) -> Optional[int]:
        if self._use_alignment_llm_endpoint(priority) and self.alignment_max_tokens is not None:
            return int(self.alignment_max_tokens)
        if self.max_tokens is not None:
            return int(self.max_tokens)
        return None

    @staticmethod
    def _estimate_text_token_count(text: Any) -> int:
        return estimate_tokens(text)

    def _estimate_messages_token_count(self, messages: List[Dict[str, Any]]) -> int:
        return estimate_messages_token_count(messages)

    @staticmethod
    def _error_suggests_context_overflow(err: BaseException) -> bool:
        return error_suggests_context_overflow(err)

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
        """按当前线程优先级返回实体 content 截断长度（步骤9–10 可走 alignment 配置）。"""
        p = getattr(self._priority_local, "priority", LLM_PRIORITY_EXTRACT)
        if self._use_alignment_llm_endpoint(p) and self.alignment_content_snippet_length is not None:
            return int(self.alignment_content_snippet_length)
        return int(self.content_snippet_length or 50)

    def effective_relation_snippet_length(self) -> int:
        """按当前线程优先级返回关系 content 截断长度，防止关系匹配 prompt 随库增长。"""
        p = getattr(self._priority_local, "priority", LLM_PRIORITY_EXTRACT)
        if (
            self._use_alignment_llm_endpoint(p)
            and self.alignment_relation_content_snippet_length is not None
        ):
            return max(1, int(self.alignment_relation_content_snippet_length))
        return max(1, int(self.relation_content_snippet_length or 50))

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
        return is_valid_utf8(text)

    def _select_llm_semaphore(self) -> Optional[SharedLLMSemaphore]:
        """所有 LLM 调用共享一个全局 FIFO 闸门；priority 只是路由标签，不影响排队。"""
        return self._llm_sem_upstream

    def get_llm_semaphore_active_count(self) -> int:
        return self._llm_sem_upstream.active_count if self._llm_sem_upstream else 0

    def get_llm_semaphore_max(self) -> int:
        return self._llm_upstream_slot_max or self._max_llm_concurrency

    def get_llm_semaphore_detail(self) -> dict:
        u_active = self._llm_sem_upstream.active_count if self._llm_sem_upstream else 0
        return {
            "upstream_active": u_active, "upstream_max": self._llm_upstream_slot_max,
            # 旧版 upstream/downstream 双池已合并为单闸门；保留键位兼容监控输出
            "downstream_active": 0, "downstream_max": 0,
        }

    def _record_call_stats(self, *, model: str, messages: List[Dict[str, Any]],
                           response_text: str, prompt_tokens: Optional[int],
                           completion_tokens: Optional[int]) -> None:
        """记录一次成功的上游调用；usage 缺失时回退本地估算并标记 estimated。"""
        estimated = prompt_tokens is None or completion_tokens is None
        if prompt_tokens is None:
            prompt_tokens = self._estimate_messages_token_count(messages)
        if completion_tokens is None:
            completion_tokens = self._estimate_text_token_count(response_text)
        self._call_stats.record(
            model=model,
            step=self._current_distill_step or "unlabeled",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated=estimated,
        )

    def reset_call_stats(self) -> None:
        """清零本客户端的调用计数（共享实例则影响所有共享方）。"""
        self._call_stats.reset()

    def get_call_stats(self) -> Dict[str, Any]:
        """返回调用计数快照：calls/prompt_tokens/completion_tokens/by_step/by_model。"""
        return self._call_stats.snapshot()

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
        json_parse_retries: int = 1,
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
                json_mode=True,
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
        json_mode: bool = False,
    ) -> str:
        """
        调用LLM的通用方法（带重试机制）

        Args:
            prompt: 用户提示（messages 为 None 时使用）
            system_prompt: 系统提示（可选）
            max_retries: 兼容保留（已忽略）；普通 API 错误固定为最多 3 轮退避重试（等待 2/5/10 秒）。
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
                self._record_call_stats(
                    model="mock",
                    messages=messages or [],
                    response_text="",
                    prompt_tokens=None,
                    completion_tokens=None,
                )
                return mock_llm_response(mock_prompt)
            return ""

        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        _utf8_round = 0
        _normal_failures = 0
        _500_failures = 0
        _conn_failures = 0
        _tpm_round = 0
        _detailed_error_logged = False
        _priority_init = getattr(self._priority_local, "priority", LLM_PRIORITY_ALIGN)
        _mt0 = self._effective_max_tokens_base(_priority_init)
        _effective_max_tokens = _mt0 if _mt0 is not None else 4096
        _cancel_fn = self._cancel_check_fn
        # Resolve LLM endpoint config once (doesn't change between retries)
        _eff_base = self._effective_base_url(_priority_init)
        _eff_key = self._effective_api_key(_priority_init)
        _eff_model = self._effective_model(_priority_init)
        _eff_think = self._effective_think_mode(_priority_init)
        _sem = self._select_llm_semaphore()
        while True:
            # 获取并发信号量（按优先级排队等待；上游/下游分池）
            _sem_held = False
            if _sem is not None:
                _wait_started = time.monotonic()
                _sem.acquire(_priority_init)
                _sem_held = True
                _wait_elapsed = time.monotonic() - _wait_started
                if _wait_elapsed >= 1.0:
                    wprint_info(
                        f"[llm_wait] priority={_priority_init} model={_eff_model} "
                        f"wait={_wait_elapsed:.1f}s active={_sem.active_count}/{_sem.max_value}"
                    )
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
                        extra_body=self._effective_openai_extra_body(_priority_init),
                    )
                else:
                    resp = ollama_chat(
                        messages,
                        model=_eff_model,
                        base_url=self._ollama_root_from(_eff_base),
                        think=_eff_think,
                        timeout=timeout,
                        num_predict=_api_max_tokens,
                        json_format=json_mode,
                    )
                response_text = resp.content or ""
                _pe = getattr(resp, "prompt_eval_count", None)
                _ev = getattr(resp, "eval_count", None)
                self._record_call_stats(
                    model=_eff_model,
                    messages=messages,
                    response_text=response_text,
                    prompt_tokens=_pe,
                    completion_tokens=_ev,
                )
                if _pe is not None or _ev is not None:
                    wprint_info(
                        f"[llm_tokens] in={_pe or '?'} out={_ev or '?'} "
                        f"({len(messages)} msgs, {len(response_text)} chars)"
                    )
                # 已成功完成一次上游 HTTP 调用：清零各类失败计数（UTF-8 轮次单独计）
                _normal_failures = 0
                _500_failures = 0
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
                        wprint_info("警告：检测到非UTF-8编码但已达到最大重试次数，返回原始响应")
                        wprint_info(f"问题内容预览:\n{response_text}")

                # 编码有效则返回响应（已取消乱码检测）
                # 蒸馏数据保存（按当前 step 标签归档）
                if response_text and self._current_distill_step:
                    self._save_distill_conversation(
                        messages + [{"role": "assistant", "content": response_text}]
                    )
                # 清理弱模型可能回显的 XML 分隔符标签（<记忆缓存>、<输入文本> 等）
                return clean_separator_tags(response_text)

            except Exception as e:
                # 统一处理错误，包括连接错误、超时等
                error_str = str(e).lower()
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

                # Xinference 500 内部错误（如 'choices' KeyError）：快速重试
                # 这类错误是 Xinference/llama.cpp 的临时性 bug，快速重试通常即可恢复
                # 使用独立计数器（_500_failures），不受普通错误上限影响
                _sc = getattr(e, "status_code", None)
                if _sc == 500:
                    _500_failures += 1
                    # 构建调用上下文标签，方便定位是哪步出错
                    _caller_ctx = f"model={_eff_model}"
                    if _sem is not None:
                        _caller_ctx += f" active={_sem.active_count}/{_sem.max_value}"
                    _caller_ctx += f" msgs={len(messages)}"
                    if _500_failures <= _XINFERENCE_500_MAX_RETRIES:
                        _idx = min(_500_failures - 1, len(_XINFERENCE_500_BACKOFF) - 1)
                        wait_seconds = _XINFERENCE_500_BACKOFF[_idx]
                        # 添加随机抖动防止并发重试的惊群效应
                        jitter = random.uniform(0, _XINFERENCE_500_JITTER_MAX)
                        wait_seconds = round(wait_seconds + jitter, 1)
                        wprint_info(
                            f"LLM 服务端 500 错误（第 {_500_failures}/{_XINFERENCE_500_MAX_RETRIES} 次）"
                            f" [{_caller_ctx}]: {e}"
                        )
                        wprint_info(f"{wait_seconds}s 后快速重试（保持信号量槽位）...")
                        # 500 期间保持信号量：服务端正在恢复，释放槽位会让排队的请求
                        # 立即冲击尚未恢复的服务端，形成级联 500
                        if _cancel_fn and _cancel_fn():
                            if _sem is not None:
                                _sem.release()
                            _sem_held = False
                            raise CancelledError("LLM call cancelled by pipeline control")
                        time.sleep(wait_seconds)
                        continue
                    wprint_info(
                        f"LLM 服务端 500 错误已达 {_XINFERENCE_500_MAX_RETRIES} 轮"
                        f" [{_caller_ctx}]: {e}"
                    )
                    if _sem is not None:
                        _sem.release()
                    _sem_held = False
                    raise

                # max_tokens 超限：自动降低重试（不计入退避轮次）
                if "max_tokens" in error_str or "max_completion_tokens" in error_str or "too large" in error_str:
                    if _effective_max_tokens and _effective_max_tokens > 1:
                        _effective_max_tokens = _effective_max_tokens // 2
                        wprint_info(f"[DeepDream] max_tokens 超限，自动降至 {_effective_max_tokens} 后重试")
                        if _sem is not None:
                            _sem.release()
                        _sem_held = False
                        if _cancel_fn and _cancel_fn():
                            raise CancelledError("LLM call cancelled by pipeline control")
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
                    if _cancel_fn and _cancel_fn():
                        raise CancelledError("LLM call cancelled by pipeline control")
                    time.sleep(wait_seconds)
                    continue

                # 连接错误：最多 5 轮，等待固定退避
                if is_connection_error:
                    _conn_failures += 1
                    if _conn_failures <= _LLM_MAX_FAILURE_ROUNDS:
                        wait_seconds = _LLM_BACKOFF_SCHEDULE[min(_conn_failures - 1, len(_LLM_BACKOFF_SCHEDULE) - 1)]
                        wprint_info(f"LLM连接错误（第 {_conn_failures}/{_LLM_MAX_FAILURE_ROUNDS} 次失败）: {e}")
                        wprint_info(f"{wait_seconds} 秒后重试...")
                        if _sem is not None:
                            _sem.release()
                        _sem_held = False
                        if _cancel_fn and _cancel_fn():
                            raise CancelledError("LLM call cancelled by pipeline control")
                        time.sleep(wait_seconds)
                        continue
                    wprint_info(f"LLM连接错误已达 {_LLM_MAX_FAILURE_ROUNDS} 轮，放弃重试: {e}")
                    if _sem is not None:
                        _sem.release()
                    _sem_held = False
                    raise

                # 其它错误（含超时）：最多 5 轮，等待固定退避
                _normal_failures += 1
                if _normal_failures <= _LLM_MAX_FAILURE_ROUNDS:
                    wait_seconds = _LLM_BACKOFF_SCHEDULE[min(_normal_failures - 1, len(_LLM_BACKOFF_SCHEDULE) - 1)]
                    if is_timeout:
                        wprint_info(f"LLM调用超时（第 {_normal_failures}/{_LLM_MAX_FAILURE_ROUNDS} 次失败，超时: {timeout}s）: {e}")
                    else:
                        wprint_info(f"LLM调用错误（第 {_normal_failures}/{_LLM_MAX_FAILURE_ROUNDS} 次失败）: {e}")
                    wprint_info(f"{wait_seconds} 秒后重试...")
                    if _sem is not None:
                        _sem.release()
                    _sem_held = False
                    if _cancel_fn and _cancel_fn():
                        raise CancelledError("LLM call cancelled by pipeline control")
                    time.sleep(wait_seconds)
                    continue

                if is_timeout:
                    wprint_info(f"LLM调用超时（已达 {_LLM_MAX_FAILURE_ROUNDS} 轮重试，超时时间: {timeout}秒）: {e}")
                else:
                    wprint_info(f"LLM调用错误（已达 {_LLM_MAX_FAILURE_ROUNDS} 轮重试）: {e}")
                if _sem is not None:
                    _sem.release()
                _sem_held = False
                raise
            finally:
                if _sem is not None and _sem_held:
                    _sem.release()

    # Delegate to extracted module-level functions for backward compatibility
    def _clean_json_string(self, json_str: str) -> str:
        return clean_json_string(json_str)

    def _fix_json_errors(self, json_str: str) -> str:
        return fix_json_errors(json_str)

    def _parse_json_response(self, response: str) -> Any:
        return parse_json_response(response)

    def _try_repair_truncated_json_array(self, json_str: str) -> Optional[str]:
        from .json_repair import try_repair_truncated_json_array
        return try_repair_truncated_json_array(json_str)

    def _try_repair_truncated_json_object(self, json_str: str) -> Optional[str]:
        from .json_repair import try_repair_truncated_json_object
        return try_repair_truncated_json_object(json_str)

    def _mock_llm_response(self, prompt: str) -> str:
        return mock_llm_response(prompt)
