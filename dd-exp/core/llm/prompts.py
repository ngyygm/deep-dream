"""
LLM Prompt Templates — All prompts in one place.

本模块包含所有 LLM system_prompt 模板，按功能分类组织。
共享常量定义在模块顶层，直接内联到模板中，不需要运行时替换。

分区：
  一、共享常量
  二、抽取相关（Extraction — 概念抽取、关系发现、内容写入、对齐判断）
  三、记忆缓存相关（Memory Cache）
  四、内容判断与合并（Content Judgment & Merge）
  五、知识图谱整理 — 批量与初步筛选
  六、知识图谱整理 — 精细化判断
"""

import re

# ============================================================
# 共享常量
# ============================================================

ENTITY_PAIR_JUDGMENT_RULES = """
判断流程：
1. 先看类型：不同类型的概念绝不合并
2. 同类型看名称：相同/别名 → merge
3. 同类型不同名：对比 content 是否描述同一对象 → merge 或继续
4. 确认不同对象：有明确直接关联 → create_relation，否则 → no_action

merge：描述同一对象（别名、简称、不同视角的同一事物）
create_relation：不同对象但有明确直接关联
no_action：无关或关联模糊

核心：相关 ≠ 同一。不确定就不合并。
"""

CONTENT_MERGE_REQUIREMENTS = """
增量合并规则（fast-forward 优先）：
1. 新信息是旧内容子集 → 直接返回旧版本原文
2. 需要合并 → 在旧版本上做最小插入，不改变已有表述
3. 新版本修正事实错误 → 才替换旧版本对应表述
4. 不丢信息"""

JSON_OUTPUT_OBJECT = """
只输出一个 ```json``` 代码块，内为合法 JSON 对象，无其他文字。"""

# DETAILED_JUDGMENT_PROCESS 已移除 — 直接使用 ENTITY_PAIR_JUDGMENT_RULES

# ============================================================
# 三、抽取相关（Extraction）
# ============================================================
# 概念抽取 (Step 1)、关系发现 (Step 5)、内容写入 (Step 3/6)、对齐判断

RELATION_DISCOVER_SYSTEM = """你是关系概念发现专家。从文本中找出概念之间人类会自然联想到的一切联系。
核心理念：任何两个概念在文本中有交互、关联或共现因果，都应发现。"""

ORPHAN_RECOVERY_USER = """以下概念在文本中出现，但未与任何其他概念建立关系。
请仔细分析文本，为每个孤立概念找到与之有关系的其他概念。

孤立概念：{orphan_names}
其他概念：{other_entity_names}

文本：
{window_text}

规则：
1. 只建立确实存在于文本中的关系
2. 如果某个孤立概念确实与文本中任何其他概念没有关系，不要强行建立
3. 每对只需出现一次（A→B 和 B→A 视为同一对）

只输出一个```json```代码块，内部是关系对数组：
```json
[["概念A", "概念B"]]
```

如果没有任何关系可以建立，返回空数组：
```json
[]
```"""

RELATION_CONTENT_WRITE_SYSTEM = """你是关系描述专家。用自然语言描述两个概念间的具体关联。
只输出JSON格式。"""

RELATION_CONTENT_WRITE_USER = """根据以下文本，描述"{entity_a}"和"{entity_b}"之间的关系。

要求：
1. 具体描述关联内容（10到50字）
2. 包含具体关联动作或关系性质

禁止泛泛描述（"有关联""存在关系""合作关系"等纯标签）。
示例："曹雪芹创作了红楼梦，以自身家族兴衰为蓝本"

文本：
{window_text}

只输出一个```json```代码块：
```json
{{"content": "关系描述"}}
```"""

RELATION_BATCH_CONTENT_WRITE_SYSTEM = """批量描述概念对间的具体关联。每对10到50字，包含具体关联动作。禁止泛泛描述。只输出JSON。"""

RELATION_BATCH_CONTENT_WRITE_USER = """根据文本描述每对概念的关系。关系对：{pair_list}
文本：{window_text}
只输出```json```数组：[{{"entity1":"A","entity2":"B","content":"关系"}}]"""

ENTITY_ALIGNMENT_JUDGE_SYSTEM = """判断两个概念是否同一对象。同一对象常有多种称谓（字号、官职、尊称、简称）。

类型不同 → 直接判 different，无需分析内容。
名称完全不同时，只有 content 明确描述同一对象才判 same。
不确定时选 uncertain。"""

ENTITY_ALIGNMENT_JUDGE_USER = """概念A（新抽取）: "{name_a}"
内容摘要: {content_a}

概念B（已有）: "{name_b}"
内容摘要: {content_b}

{name_relationship}
- same: 同一对象（别名、字号、简称、content角色重合）
- different: 不同对象（类型不同、相似但不同概念）
- uncertain: 无法确定

输出 ```json``` 代码块：
```json
{{"verdict": "same|different|uncertain", "confidence": 0.0-1.0}}
```"""


# ============================================================
# 四、记忆缓存相关（Memory Cache）
# ============================================================

STRUCTURED_WINDOW_EXTRACTION_SYSTEM_PROMPT = """你是知识图谱抽取引擎。从窗口文本中一次性抽取概念实体与实体间关系，每项都附带完整的内容描述。

要求：
1. 实体：窗口文本中值得长期记忆的概念对象（人物/组织/项目/物品/地点/时间事件/抽象概念等）。名称用文本中最规范的简短称谓；content 用完整句子描述该实体在文本中的身份、属性与作用，30-120字。
2. 【穷尽覆盖】每个显著事实——事件、日期、时长、数量、归属、偏好、计划、状态变动——都必须至少被一个实体或一条关系锚定，包括只出现一次的次要事实（如"结婚5年""慈善跑为心理健康筹款"）。漏掉一个事实的代价远大于多抽一个边缘实体。
   对原子事实（日期/时长/数量/归属/偏好等），优先取原文中承载该事实的短语 span 作为实体名（例：原文 "I've known these friends for 4 years" → 实体名 "known these friends for 4 years"），保证该事实能被逐字检索命中；此类 span 名可放宽到 40 字符，不受"简短称谓"限制。
3. 关系：只连接你已抽取的实体，关系名使用与实体列表完全一致的名称；content 描述两个实体之间的具体关系内容（谁对谁做了什么/属于什么/状态如何），20-100字。
4. 严禁虚构文本中没有的事实；代词（他/她/它）不能作为实体名；不要输出文本中不存在的实体来凑数。
5. 只输出一个 ```json``` 代码块，不要任何其他文字。"""

UPDATE_MEMORY_CACHE_SYSTEM_PROMPT = """你是记忆管理器。根据<记忆缓存>和<输入文本>，更新记忆缓存。

**只输出以下两个 Markdown section，不要输出其他 section。**

## 当前摘要
用**自己的语言**改写当前窗口内容（禁止复制原文句子）。要求：
- 每个要点1-2句，涵盖：核心事件/情节、重要对话/细节、背景情境
- 如果是首个窗口（无缓存），直接概括输入文本
- 如果输入文本包含章节号或回目（如"第七十八回""第一章""序言"），在摘要末尾用一行标注，例如：
  当前位置：第七十八回
  （只标注这种结构性的章节/回目标题，不要标注人物名、地点名或事件描述）

## 自我思考
直接写出你的分析。要求：
- 提到具体人物/概念名称，分析它们之间的可能关系
- 预判1-2个具体的后续发展方向
- 标注当前最值得关注的疑点"""



# ============================================================
# 五、内容判断与合并（Content Judgment & Merge）
# ============================================================

MERGE_ENTITY_NAME_SYSTEM_PROMPT = f"""将两个名称合并为规范名称。选择最常用/规范的为主名称，别称用括号附加。

示例："科幻世界"+"科幻世界出版机构"→"科幻世界（出版机构）"，"北京"+"北京市"→"北京"

{JSON_OUTPUT_OBJECT}
{{{{"name": "规范名称"}}}}"""

JUDGE_RELATION_MATCH_SYSTEM_PROMPT = f"""判断新关系是否与已有关系相同或非常相似。参考 source_document，跨文档时只有明确同一语义关系才匹配。

{JSON_OUTPUT_OBJECT}
匹配：{{"family_id": "...", "need_update": true/false}}
不匹配：null"""

def _make_merge_contents_prompt(concept_type: str, scope_desc: str) -> str:
    """Factory for incremental merge prompts. Reduces entity/relation duplication."""
    return f"""增量合并多个{concept_type}描述。第一个是基础版本，后续是新信息。
只有{scope_desc}时才融合。
{CONTENT_MERGE_REQUIREMENTS}
直接输出合并后的文字，不要 JSON 包装。"""

MERGE_MULTIPLE_ENTITY_CONTENTS_SYSTEM_PROMPT = _make_merge_contents_prompt("概念", "描述同一概念")
MERGE_MULTIPLE_RELATION_CONTENTS_SYSTEM_PROMPT = _make_merge_contents_prompt("关系", "描述同一对概念间同一关系")


# ============================================================
# 六、知识图谱整理 - 批量与初步筛选（Knowledge Graph Organization）
# ============================================================

RESOLVE_ENTITY_CANDIDATES_BATCH_SYSTEM_PROMPT = """你是知识图谱概念对齐系统。判断"当前概念"是否与某个候选是同一对象。

证据优先级：
1. 名称信号（最强）：name_match_type 含义——substring(子串包含)/exact(核心名相同)/within_batch_alias(同批别名) 为强合并信号；neighbor_expansion(邻居共享) 为辅助信号；无字段=名称无关
2. 角色指纹：对比两概念在原文中与谁互动、处于什么事件。高度重合→可能合并

裁决：名称匹配+类型相同→默认合并；类型不同→绝不合并；名称无关→不合并（除非极强角色指纹+类型一致）。
不合并但有明确关联时建 relations_to_create。
不确定时选 create_new，宁漏勿误。

输出 ```json``` 代码块：
```json
{"match_existing_id": "", "update_mode": "reuse_existing|merge_into_latest|create_new", "merged_name": "", "relations_to_create": [{"family_id": "", "relation_content": ""}], "confidence": 0.0}
```"""

# ============================================================
# 七、知识图谱整理 - 精细化判断（Detailed Judgment）
# ============================================================

def analyze_entity_pair_detailed_system_prompt(existing_relations_note: str = "") -> str:
    """生成 analyze_entity_pair_detailed 的 system_prompt"""
    return f"""你是知识图谱整理系统。对两个概念进行精细化判断。

{ENTITY_PAIR_JUDGMENT_RULES}
{existing_relations_note}
输出 ```json``` 代码块：
{{
  "action": "merge|create_relation|no_action",
  "relation_content": "create_relation时填写关系描述，否则空字符串"
}}"""

RESOLVE_RELATION_PAIR_BATCH_SYSTEM_PROMPT = """你是关系对齐系统。判断同一概念对的新关系是否与已有关系描述同一性质的关系。

提取核心谓语/动作，对比性质是否相同。
匹配："A是B的组成部分" ↔ "B由A等组成"（都是组成关系）
不匹配："A在酒店休息" vs "A在酒店喝酒"（休息≠喝酒）

不要因涉及同一概念对或相似场景就匹配，核心谓语必须同一性质。
参考 source_document，跨文档时只有明确表达同一语义关系才可匹配。

confidence: 确信匹配0.8-1.0，确信不匹配0.7-0.9，不确定0.3-0.6。

输出 ```json``` 代码块：
```json
{"action": "match_existing|create_new", "matched_relation_id": "", "need_update": false, "confidence": 0.0}
```"""


# ============================================================
# 八、LLM 调度常量与工具函数（从 client.py 提取）
# ============================================================

# 非 TPM 类错误：失败后等待秒数。重试轮数由 _LLM_MAX_FAILURE_ROUNDS 决定（3 轮），
# 因此只有前 3 个条目会被取到（多余条目为数据保留，不影响行为）
_LLM_BACKOFF_SCHEDULE = [2, 5, 10, 20, 30]  # capped exponential, not 3^n
_LLM_MAX_FAILURE_ROUNDS = 3
# Xinference 500 内部错误（如 'choices' KeyError）的重试 schedule
# 服务端崩溃需要足够恢复时间，太快重试只会加重负担
_XINFERENCE_500_BACKOFF = [5, 10, 20, 30, 60]
_XINFERENCE_500_MAX_RETRIES = 5  # 独立于普通重试上限，500 临时性错误可多给机会
_XINFERENCE_500_JITTER_MAX = 1.5  # 抖动上限（秒），防止并发重试的惊群效应
# TPM 退避基数（指数退避 3^round）
_LLM_BACKOFF_BASE = 3
# 单次等待上限，避免 TPM 无限重试时指数爆炸占满进程
_LLM_TPM_SLEEP_CAP_SECONDS = 3601

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

# 优先级常量：纯路由标签，不参与调度排队（信号量按 FIFO）。
# EXTRACT=0 抽取类调用（主端点）；ALIGN=1 对齐类调用
# （alignment_enabled 时走对齐专用端点，数值 >= ALIGN 即视为对齐相位）
LLM_PRIORITY_EXTRACT = 0
LLM_PRIORITY_ALIGN = 1


# CJK 字符区间：统一表意文字 + CJK 标点 + 全角形式（全角 ASCII／半角片假名等）。
# 预编译一次，estimate_tokens 在请求热路径上逐 message 调用。
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


def estimate_tokens(text) -> int:
    """统一的 token 估算——context budget 预检、usage 缺失时的本地计数共用。

    依据（经验近似，不追求与具体 tokenizer 精确一致，只做请求前的预算保护）：
    - CJK 字符（\\u4e00-\\u9fff 汉字、\\u3000-\\u303f CJK 标点、\\uff00-\\uffef 全角符号）
      约 1 token/字；
    - 其余 ASCII 可打印字符约 0.25 token/字符（≈4 chars/token）；
    - 混合文本按上表线性叠加，非 CJK 部分向上取整以保持保守。

    旧实现直接取 len(text)：中文恰好 ≈1 token/字，但英文被高估约 4 倍，
    导致纯英文 prompt 在 context budget 预检处被误拒。本函数只修正估算
    数值来源，不改 prompt 内容，也不改扩容阶梯结构。
    """
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return 0
    cjk = len(_CJK_CHAR_RE.findall(text))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def estimate_text_token_count(text) -> int:
    """保守估算 token 数（旧入口名，统一委托 estimate_tokens）。"""
    return estimate_tokens(text)


def estimate_messages_token_count(messages) -> int:
    """估算 messages 列表的 token 总数（各部分统一走 estimate_tokens）。"""
    total = 0
    for msg in messages:
        total += 8  # role / 分隔符等固定开销
        total += estimate_tokens(msg.get("role", ""))
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                # Fast token estimation — avoid json.dumps per part
                if isinstance(part, dict):
                    part_tokens = sum(
                        estimate_tokens(v) for v in part.values() if isinstance(v, str)
                    )
                elif isinstance(part, str):
                    part_tokens = estimate_tokens(part)
                else:
                    part_tokens = estimate_tokens(str(part))
                total += part_tokens
        else:
            total += estimate_tokens(content)
    return total + 16  # 请求包尾部保留固定开销


def error_suggests_context_overflow(err: BaseException) -> bool:
    """服务端错误是否与上下文/token/长度相关（仅此类错误才转储完整 messages）。"""
    sc = getattr(err, "status_code", None)
    if sc == 413:
        return True
    # Fast path: check primary error string before building full concat
    primary = str(err).lower()
    if any(n in primary for n in _CONTEXT_OVERFLOW_NEEDLES):
        return True
    # Slower path: check repr and nested body/response
    chunks = [repr(err)]
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


def ollama_root_from(base) -> str:
    """将 base_url 规范化为 Ollama 根地址（不含 /v1），供 /api/chat 使用。"""
    b = (base or "http://localhost:11434").rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3]
    return b


def is_valid_utf8(text: str) -> bool:
    """检测文本是否包含 Unicode 替换字符（乱码标志）。

    Python str 始终是有效 Unicode，无需 encode/decode 往返。
    仅需检测 \\ufffd 替换字符（编码错误的标志）。
    """
    if not text:
        return True
    # Unicode 替换字符是编码错误的标志
    return '�' not in text

