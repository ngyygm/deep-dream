"""
LLM客户端：封装LLM调用，实现三个核心任务。

请求方式：统一通过 `processor/ollama_chat_api.py` 访问：
- Ollama 走原生 `POST /api/chat`；
- OpenAI/GLM/LM Studio 等走 OpenAI 兼容接口。

think 模式由初始化参数 think_mode 控制；只有 Ollama 原生协议支持通过 `think: true/false` 显式开关思考模式。
"""
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import heapq
import json
import os
import threading
import uuid
import time

from ..models import MemoryCache, Entity
from ..utils import clean_markdown_code_blocks, clean_separator_tags, wprint
from .chat_api import ollama_chat, openai_compatible_chat
from .prompts import (
    # 记忆缓存
    UPDATE_MEMORY_CACHE_SYSTEM_PROMPT,
    CREATE_DOCUMENT_OVERALL_MEMORY_SYSTEM_PROMPT,
    GENERATE_CONSOLIDATION_SUMMARY_SYSTEM_PROMPT,
    GENERATE_RELATION_MEMORY_CACHE_SYSTEM_PROMPT,
    # 实体抽取
    EXTRACT_ENTITIES_AND_RELATIONS_SYSTEM_PROMPT,
    EXTRACT_ENTITIES_SINGLE_PASS_SYSTEM_PROMPT,
    EXTRACT_ENTITIES_BY_NAMES_SYSTEM_PROMPT,
    ENHANCE_ENTITY_CONTENT_SYSTEM_PROMPT,
    # 关系抽取
    EXTRACT_RELATIONS_SINGLE_PASS_SYSTEM_PROMPT,
    # 内容判断与合并
    JUDGE_CONTENT_NEED_UPDATE_SYSTEM_PROMPT,
    MERGE_ENTITY_NAME_SYSTEM_PROMPT,
    # 知识图谱整理
    ANALYZE_ENTITY_CANDIDATES_PRELIMINARY_SYSTEM_PROMPT,
    RESOLVE_ENTITY_CANDIDATES_BATCH_SYSTEM_PROMPT,
)
from .memory_ops import _MemoryOpsMixin
from .entity_extraction import _EntityExtractionMixin
from .relation_extraction import _RelationExtractionMixin
from .content_merger import _ContentMergerMixin
from .consolidation import _ConsolidationMixin


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


class LLMClient(_MemoryOpsMixin, _EntityExtractionMixin, _RelationExtractionMixin, _ContentMergerMixin, _ConsolidationMixin):

    @staticmethod
    def _normalize_entity_pair(entity1: str, entity2: str) -> tuple:
        """
        标准化实体对，将实体对按字母顺序排序，使关系无向化
        
        Args:
            entity1: 第一个实体名称
            entity2: 第二个实体名称
        
        Returns:
            标准化后的实体对元组（按字母顺序排序）
        """
        entity1 = entity1.strip()
        entity2 = entity2.strip()
        # 按字母顺序排序，确保(A,B)和(B,A)被视为同一个关系
        if entity1 <= entity2:
            return (entity1, entity2)
        else:
            return (entity2, entity1)
    
    @staticmethod
    def _normalize_entity_name_to_original(entity_name: str, valid_entity_names: set) -> str:
        """
        将LLM返回的实体名称规范化为原始的完整名称（仅精确匹配）
        
        Args:
            entity_name: LLM返回的实体名称
            valid_entity_names: 有效的原始实体名称集合
        
        Returns:
            如果找到精确匹配则返回规范化后的实体名称，否则返回原名称
        """
        entity_name = entity_name.strip()
        
        # 精确匹配：如果已经是完整名称，直接返回
        if entity_name in valid_entity_names:
            return entity_name
        
        # 没有找到精确匹配，返回原名称
        return entity_name
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4", base_url: Optional[str] = None,
                 content_snippet_length: int = 50, think_mode: bool = False,
                 distill_data_dir: Optional[str] = None, max_tokens: Optional[int] = None,
                 max_llm_concurrency: Optional[int] = None):
        """
        初始化LLM客户端

        Args:
            api_key: API密钥
            model_name: 模型名称
            base_url: API基础URL（可选，用于自定义API端点）
            content_snippet_length: 传入LLM prompt的实体content最大长度（默认50字符）
            think_mode: 是否开启思维链/think 模式（默认 False）。仅 Ollama 原生 `/api/chat` 下通过 API 参数 think 控制；其他后端忽略
            max_tokens: LLM 最大输出 token 数（可选）。Ollama 对应 num_predict，OpenAI 对应 max_tokens
            max_llm_concurrency: 最大并发 LLM 请求数（可选）。None 表示不限制
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.content_snippet_length = content_snippet_length
        self.think_mode = think_mode
        self.max_tokens = max_tokens
        # 统一使用 Python SDK（openai>=1.0）访问；无 api_key 且无 base_url 时为模拟模式
        self._endpoint_available = bool(api_key or base_url)
        if not self._endpoint_available:
            wprint("提示：未提供 API key 或 base_url，将使用模拟响应模式")

        # LLM 并发控制：带优先级的信号量
        self._llm_semaphore: Optional[PrioritySemaphore] = None
        if max_llm_concurrency is not None and max_llm_concurrency >= 1:
            self._llm_semaphore = PrioritySemaphore(max_llm_concurrency)
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

    def _get_ollama_base_url(self) -> str:
        """将配置的 base_url 规范化为 Ollama 根地址（不含 /v1），供 /api/chat 使用。"""
        base = (self.base_url or "http://localhost:11434").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return base

    def _use_openai_compatible(self) -> bool:
        """是否为 OpenAI 兼容接口（智谱 GLM、OpenAI 等）：需同时有 api_key 与 base_url，且 base_url 为 v4/v1 或已知域名。本地 Ollama 地址即使带 /v1 也按 Ollama 处理。"""
        if not self.api_key or not self.base_url:
            return False
        u = (self.base_url or "").rstrip("/").lower()
        # 约定：api_key=ollama 表示使用 Ollama（即使是远端 /v1）
        if (self.api_key or "").strip().lower() == "ollama":
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
        """
        检测文本是否是有效的UTF-8编码
        
        Args:
            text: 待检测的文本
        
        Returns:
            True表示是有效的UTF-8编码，False表示不是
        """
        if not text:
            return True  # 空文本视为有效
        
        try:
            # 尝试将字符串编码为UTF-8字节，然后再解码回来
            # 如果文本包含无效的Unicode字符，这个过程会失败或产生替换字符
            encoded = text.encode('utf-8')
            decoded = encoded.decode('utf-8')
            
            # 检查解码后的文本是否与原始文本相同
            # 如果不同，说明编码/解码过程中出现了问题
            if decoded != text:
                return False
            
            # 检查是否包含Unicode替换字符（\ufffd），这是编码错误的标志
            if '\ufffd' in text:
                return False
            
            return True
        except (UnicodeEncodeError, UnicodeDecodeError):
            # 编码或解码失败，说明不是有效的UTF-8
            return False
        except Exception:
            # 其他异常，保守起见返回True（避免误判）
            return True

    def _save_distill_conversation(self, messages: List[Dict[str, str]]):
        """保存一次 LLM 对话到 JSONL 文件（OpenAI fine-tuning 格式）。"""
        if not self._distill_data_dir or not self._current_distill_step or not self._distill_task_id:
            return
        step_dir = os.path.join(self._distill_data_dir, self._current_distill_step)
        os.makedirs(step_dir, exist_ok=True)
        filepath = os.path.join(step_dir, f"{self._distill_task_id}.jsonl")
        line = json.dumps({"messages": messages}, ensure_ascii=False)
        with self._distill_lock:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 300,
        allow_mock_fallback: bool = True,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        调用LLM的通用方法（带重试机制）

        Args:
            prompt: 用户提示（messages 为 None 时使用）
            system_prompt: 系统提示（可选）
            max_retries: 最大重试次数（默认3次）
            timeout: 超时时间（秒），默认300秒（5分钟），本地 Ollama 等可适当调大
            allow_mock_fallback: 失败时是否降级为模拟响应；启动握手等场景应传 False，避免误判为可用
            messages: 完整对话列表（可选）；传入时直接使用，忽略 prompt 和 system_prompt

        Returns:
            LLM的响应文本；allow_mock_fallback=False 且失败时返回空字符串
        """
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
        attempt = 0
        _priority = getattr(self._priority_local, 'priority', LLM_PRIORITY_STEP7)
        _sem = self._llm_semaphore
        last_error = None
        attempt = 0
        _effective_max_tokens = self.max_tokens  # 允许运行时动态降低
        _truncation_retries = 0  # 截断自动扩容重试计数（不计入 max_retries）
        while True:
            # 获取并发信号量（按优先级排队等待）
            _sem_held = False
            if _sem is not None:
                _sem.acquire(_priority)
                _sem_held = True
            try:
                if self._use_openai_compatible():
                    resp = openai_compatible_chat(
                        messages,
                        model=self.model_name,
                        base_url=self.base_url.rstrip("/"),
                        api_key=self.api_key,
                        timeout=timeout,
                        max_tokens=_effective_max_tokens,
                    )
                else:
                    resp = ollama_chat(
                        messages,
                        model=self.model_name,
                        base_url=self._get_ollama_base_url(),
                        think=self.think_mode,
                        timeout=timeout,
                        num_predict=_effective_max_tokens,
                    )
                response_text = resp.content or ""

                # 检测 LLM 输出被 max_tokens 截断（finish_reason/done_reason == "length"）
                _is_truncated = (
                    getattr(resp, "done_reason", None) == "length"
                    or (resp.raw and resp.raw.get("choices") and
                        resp.raw["choices"][0].get("finish_reason") == "length")
                )
                _max_allowed = self.max_tokens * 2  # 扩容上限为配置值的2倍
                if _is_truncated and _truncation_retries < 1 and _effective_max_tokens < _max_allowed:
                    _truncation_retries += 1
                    _effective_max_tokens = _max_allowed
                    wprint(f"[TMG] LLM 输出被截断（max_tokens 不足），自动扩容至 {_effective_max_tokens} 后重试")
                    if _sem is not None:
                        _sem.release()
                    _sem_held = False
                    time.sleep(0.5)
                    continue

                # 检测是否是有效的UTF-8编码
                if not self._is_valid_utf8(response_text):
                    if attempt < max_retries - 1:
                        wprint(f"检测到非UTF-8编码的文本，正在重新生成（第 {attempt + 1}/{max_retries} 次尝试）...")
                        wprint(f"问题内容预览:\n{response_text}")
                        attempt += 1
                        continue
                    else:
                        wprint(f"警告：检测到非UTF-8编码但已达到最大重试次数，返回原始响应")
                        wprint(f"问题内容预览:\n{response_text}")

                # 编码有效则返回响应（已取消乱码检测）
                # 蒸馏数据保存（步骤2/3走多轮手动保存，在此跳过）
                if (response_text and self._current_distill_step
                        and self._current_distill_step not in ("02_extract_entities", "03_extract_relations")):
                    self._save_distill_conversation(
                        messages + [{"role": "assistant", "content": response_text}]
                    )
                # 清理弱模型可能回显的 XML 分隔符标签（<记忆缓存>、<输入文本> 等）
                return clean_separator_tags(response_text)

            except Exception as e:
                # 统一处理错误，包括连接错误、超时等
                error_str = str(e).lower()
                last_error = e
                # 检查是否是超时错误
                is_timeout = "timeout" in error_str or "timed out" in error_str
                # 检查是否是连接类错误（如 connection refused）
                is_connection_error = any(
                    kw in error_str
                    for kw in [
                        "connection refused",
                        "connectionerror",
                        "failed to establish a new connection",
                        "newconnectionerror",
                        "temporarily unreachable",
                        "temporary failure in name resolution",
                        "name or service not known",
                        "connection aborted",
                        "connection reset",
                        "errno 111",
                    ]
                )

                # max_tokens 超限：自动降低重试（不计入 max_retries）
                if "max_tokens" in error_str or "max_completion_tokens" in error_str or "too large" in error_str:
                    if _effective_max_tokens and _effective_max_tokens > 2048:
                        _effective_max_tokens = _effective_max_tokens // 2
                        wprint(f"[TMG] max_tokens 超限，自动降至 {_effective_max_tokens} 后重试")
                        if _sem is not None:
                            _sem.release()
                        _sem_held = False
                        time.sleep(0.5)
                        continue
                    # 已经很低了仍失败，走正常重试逻辑

                # 对于连接错误：重试但设置上限（避免无限阻塞线程）
                if is_connection_error:
                    _conn_max = max_retries * 3  # 连接错误允许更多次重试
                    wait_seconds = min(5 * (attempt + 1), 60)  # 指数退避，上限60秒
                    wprint(f"LLM连接错误（第 {attempt + 1} 次尝试）: {e}")
                    if attempt >= _conn_max:
                        wprint(f"连接错误已达上限 {_conn_max} 次，放弃重试")
                        raise
                    wprint(f"{wait_seconds} 秒后重试...")
                    if _sem is not None:
                        _sem.release()
                    _sem_held = False
                    time.sleep(wait_seconds)
                    attempt += 1
                    continue

                # 非连接错误：按照原有 max_retries 策略处理
                wprint(f"LLM调用错误（第 {attempt + 1}/{max_retries} 次尝试）: {e}")
                if attempt < max_retries - 1:
                    if is_timeout:
                        wprint(f"LLM调用超时（第 {attempt + 1}/{max_retries} 次尝试，超时时间: {timeout}秒）: {e}")
                    else:
                        wprint(f"LLM调用错误（第 {attempt + 1}/{max_retries} 次尝试）: {e}")
                    wprint(f"正在重试...")
                    attempt += 1
                    if _sem is not None:
                        _sem.release()
                    _sem_held = False
                    continue
                else:
                    if is_timeout:
                        wprint(f"LLM调用超时（已达最大重试次数，超时时间: {timeout}秒）: {e}")
                    else:
                        wprint(f"LLM调用错误（已达最大重试次数）: {e}")
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
            wprint(f"所有重试都失败，使用模拟响应")
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
        import re
        # 移除BOM标记
        json_str = json_str.lstrip('\ufeff')
        # 移除首尾空白
        json_str = json_str.strip()
        # 修复中文标点符号
        json_str = json_str.replace('：', ':')  # 中文冒号 -> 英文冒号
        json_str = json_str.replace('，', ',')  # 中文逗号 -> 英文逗号
        json_str = json_str.replace('；', ';')  # 中文分号 -> 英文分号
        json_str = json_str.replace('"', '"')  # 中文左引号 -> 英文引号
        json_str = json_str.replace('"', '"')  # 中文右引号 -> 英文引号
        # 移除可能的尾随逗号（在数组或对象的最后一个元素后）
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        return json_str
    
    def _fix_json_errors(self, json_str: str) -> str:
        """
        尝试修复JSON错误
        
        Args:
            json_str: 有错误的JSON字符串
            
        Returns:
            修复后的JSON字符串
        """
        import re
        # 先进行基本清理
        json_str = self._clean_json_string(json_str)
        
        # 尝试修复常见的JSON错误
        # 1. 修复未转义的换行符（但只在字符串值内部，不在已经转义的地方）
        # 注意：这里需要小心，不要破坏已经正确转义的字符
        
        # 2. 修复无效的Unicode转义序列
        # JSON标准要求 \u 后面必须跟着恰好4个十六进制字符
        # 使用正则表达式更健壮地修复无效的Unicode转义序列
        def fix_unicode_escapes_regex(text):
            """使用正则表达式修复无效的Unicode转义序列"""
            import re
            # 匹配 \u 后面跟着0-3个十六进制字符，然后跟着非十六进制字符或字符串结尾的情况
            # 这包括：
            # - \u 后面直接跟着非十六进制字符（如 \uX, \uXX, \uXXX）
            # - \u 后面跟着0-3个十六进制字符，然后跟着非十六进制字符
            pattern = r'\\u([0-9a-fA-F]{0,3})(?![0-9a-fA-F])'
            
            def replace_invalid_escape(match):
                hex_part = match.group(1)
                if len(hex_part) == 0:
                    # 没有十六进制字符（如 \uX），替换为空格 \u0020
                    return '\\u0020'
                elif len(hex_part) < 4:
                    # 不足4位，用0补齐到4位
                    return '\\u' + hex_part.ljust(4, '0')
                else:
                    # 已经有4位，不应该匹配这个模式，但为了安全起见保持不变
                    return match.group(0)
            
            return re.sub(pattern, replace_invalid_escape, text)
        
        # 使用正则表达式方法修复无效的Unicode转义序列
        json_str = fix_unicode_escapes_regex(json_str)
        
        # 3. 修复未转义的换行符、回车符、制表符（在字符串值中）
        # 注意：这需要在字符串值内部进行，但要避免破坏已经转义的字符
        # 由于这比较复杂，我们只在明显需要的地方进行修复
        
        return json_str

    def _parse_json_response(self, response: str) -> Any:
        """从 LLM 响应中提取并解析 JSON。"""
        json_str = response or ""
        if "```json" in json_str:
            json_start = json_str.find("```json") + 7
            json_end = json_str.find("```", json_start)
            if json_end == -1:
                wprint("[TMG] 警告: LLM 响应的 ```json 块未闭合，JSON 可能被截断")
            json_str = json_str[json_start:json_end].strip() if json_end != -1 else json_str[json_start:].strip()
        elif "```" in json_str:
            json_start = json_str.find("```") + 3
            json_end = json_str.find("```", json_start)
            if json_end == -1:
                wprint("[TMG] 警告: LLM 响应的 ``` 块未闭合，JSON 可能被截断")
            json_str = json_str[json_start:json_end].strip() if json_end != -1 else json_str[json_start:].strip()

        json_str = self._clean_json_string(json_str)

        # 截断检测：检查 JSON 结构是否完整
        stripped = json_str.strip()
        if stripped and stripped[0] in ('[', '{'):
            open_char = stripped[0]
            close_char = ']' if open_char == '[' else '}'
            if not stripped.endswith(close_char):
                wprint(f"[TMG] 警告: LLM 响应 JSON 被截断，以 {open_char} 开头但不以 {close_char} 结尾。"
                      f"建议在配置中增大 llm.max_tokens 值。响应前200字符: {stripped[:200]}")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            fixed = self._fix_json_errors(json_str)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                wprint(f"[TMG] 警告: LLM 响应 JSON 解析失败（可能被截断）。"
                      f"建议在配置中增大 llm.max_tokens 值。响应前300字符: {json_str[:300]}")
                raise

    def _mock_llm_response(self, prompt: str) -> str:
        """模拟LLM响应（用于测试）"""
        prompt_lower = prompt.lower()
        if "更新记忆缓存" in prompt or "memory_cache" in prompt_lower or "创建初始的记忆缓存" in prompt:
            return """当前摘要：正在处理文档内容。当前阅读的是文档的开头部分，介绍了故事的基本背景和主要人物。重要细节包括主要人物的基本信息和故事的初始情境。

自我思考：
- 应该关注：主要人物的身份、性格特点、故事发生的背景环境
- 预判重点：后续情节可能围绕这些主要人物展开，需要留意人物之间的关系和故事的发展方向
- 疑虑：暂无特别疑虑，需要继续阅读以了解故事的发展

系统状态：
- 已处理文本范围：处理到"文档开始"结束
- 当前文档名：示例文档.txt"""
        elif ("抽取实体" in prompt or "抽取所有概念实体" in prompt or "entity" in prompt_lower or
              "从输入文本中抽取所有实体" in prompt or "实体抽取" in prompt or
              "概念实体" in prompt):
            return json.dumps([
                {
                    "name": "示例实体1",
                    "content": "这是一个示例实体的描述"
                }
            ], ensure_ascii=False)
        elif ("抽取关系" in prompt or "抽取所有概念实体间的关系" in prompt or
              "relation" in prompt_lower or "从输入文本中抽取实体之间的关系" in prompt or
              "关系抽取" in prompt or "实体间的关系" in prompt):
            # 检查实体列表是否为空
            if "已抽取的实体：" in prompt:
                entities_section = prompt.split("已抽取的实体：")[1].split("</已抽取实体>")[0].strip()
                # 如果实体部分为空或只有换行符，返回空关系列表
                if not entities_section or entities_section == "\n" or entities_section == "":
                    return json.dumps([], ensure_ascii=False)
            # 如果有实体，返回示例关系（使用与实体抽取一致的实体名称）
            return json.dumps([
                {
                    "entity1_name": "示例实体1",
                    "entity2_name": "示例实体2",
                    "content": "示例实体1与示例实体2之间的关系描述"
                }
            ], ensure_ascii=False)
        elif ("判断.*实体.*匹配" in prompt or "judge.*entity.*match" in prompt_lower or
              "判断新抽取的实体是否与已有实体" in prompt):
            # 模拟实体匹配响应
            return json.dumps({
                "entity_id": "ent_001",
                "need_update": False
            }, ensure_ascii=False)
        elif ("实体后验增强" in prompt or "enhance.*entity.*content" in prompt_lower or
              "对该实体的content进行更细致的补全和挖掘" in prompt or "增强后的完整实体content" in prompt):
            # 模拟实体后验增强响应（JSON格式）
            # 从prompt中提取原始content，然后返回增强后的版本
            if "当前content：" in prompt:
                original_content = prompt.split("当前content：")[1].split("</已抽取实体>")[0].strip()
                enhanced_content = f"{original_content}\n\n[增强信息]：基于记忆缓存和当前文本的补充细节和上下文信息。"
            else:
                enhanced_content = "这是一个示例实体的描述\n\n[增强信息]：基于记忆缓存和当前文本的补充细节和上下文信息。"
            # 返回JSON格式
            return json.dumps({"content": enhanced_content}, ensure_ascii=False)
        elif ("判断" in prompt and "合并" in prompt and "实体" in prompt) or "merge_entity_name" in prompt_lower:
            return json.dumps({"merged_name": "示例实体1", "merged_content": "合并后的描述"}, ensure_ascii=False)
        elif ("判断" in prompt and "更新" in prompt and ("content" in prompt_lower or "内容" in prompt)):
            return json.dumps({"need_update": False}, ensure_ascii=False)
        elif ("关系" in prompt and "匹配" in prompt) or "relation_match" in prompt_lower:
            return json.dumps({"relation_id": None}, ensure_ascii=False)
        elif ("生成关系" in prompt or "relation_content" in prompt_lower or "关系的content" in prompt):
            return "这是一个示例关系描述"
        elif "知识图谱整理" in prompt or "consolidation" in prompt_lower:
            return "知识图谱整理完成，未发现需要处理的重复实体。"
        elif ("整体记忆" in prompt or "document_overall" in prompt_lower or "文档整体" in prompt):
            return "# 文档整体记忆\n\n这是一份示例文档的整体描述。"
        return "默认响应"
