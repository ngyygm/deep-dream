"""抽取流水线 mixin：从 orchestrator.py 提取。"""
from __future__ import annotations

import hashlib
import math
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
import threading

from ..models import Episode, Entity
from ..debug_log import log as dbg, log_section as dbg_section
from ..utils import compute_doc_hash, wprint
from ..llm.client import (
    LLM_PRIORITY_STEP1, LLM_PRIORITY_STEP2, LLM_PRIORITY_STEP3,
    LLM_PRIORITY_STEP4, LLM_PRIORITY_STEP5, LLM_PRIORITY_STEP6, LLM_PRIORITY_STEP7,
)


def _normalize_pair_for_relation(e1: str, e2: str) -> Tuple[str, str]:
    """无向边端点按字典序固定。委托给 processor.utils.normalize_entity_pair。"""
    from processor.utils import normalize_entity_pair
    return normalize_entity_pair(e1, e2)


# 非实体文本片段黑名单：常见动词/助词/描述片段，这些不应该作为概念实体
_NON_ENTITY_FRAGMENTS: set = {
    # 常见动词/动词片段
    "遂得", "险恶", "处理进度", "突阵而出", "诸营已失", "弃关",
    "相拒", "火来", "营已失", "失了", "遂", "险", "恶",
    # 过于宽泛的描述性概念
    "忠心耿耿的群体", "官職", "居住地",
}

# 实体名称最大长度（超过此长度的大概率是场景描述、对话片段等）
_MAX_ENTITY_NAME_LENGTH = 30

# 中文名称最小长度（小于此长度几乎不可能是有意义的实体）
_MIN_ENTITY_NAME_LENGTH = 2


def _is_valid_entity_name(name: str) -> bool:
    """检查实体名称是否是有效的概念实体名称（而非文本片段或无意义词）。

    过滤规则：
    1. 长度检查：<2字符 无意义；>30字符 大概率是场景描述/对话片段
    2. 黑名单检查：已知的非实体文本片段
    3. 系统状态泄露检查：包含"处理进度"等系统提示词
    4. 重复片段检查：同一2字片段重复3次以上（如"解读解读解读..."）
    5. 对话标记检查：包含"曰：""道："等对话格式
    6. 括号内容检查：括号内不应包含场景/事件/系统信息
    """
    import re

    if not name or len(name) < _MIN_ENTITY_NAME_LENGTH:
        return False

    # 2字纯中文名称：只允许专有名词（人名、地名、特定概念）
    # 动词/动作词/状态词/形容词不应作为实体
    if len(name) == 2 and re.match(r'^[\u4e00-\u9fff]+$', name):
        _verb_2char = {
            "主持", "负责", "建议", "赞同", "大悦", "修缮",
            "广纳", "满意", "感慨", "内斗", "恢复", "统一",
            "开发", "设计", "建立", "实现", "管理", "处理",
            "分析", "评估", "研究", "探讨", "优化", "改进",
            "迁移", "发现", "阅读", "合作", "超越",
        }
        if name in _verb_2char:
            return False

    # 最大长度检查：超过此长度的大概率是场景描述、对话片段等
    # 英文名称（空格分隔的多词短语）允许更长，因为英文表达同一概念需要更多字符
    # 含英文括号标注的双语名称也允许更长（如"1903年诺贝尔物理学奖（Nobel Prize in Physics, 1903）"）
    _has_en_paren = bool(re.search(r'[(（][A-Za-z]', name))
    _max_len = 50 if (name.isascii() and ' ' in name) or _has_en_paren else _MAX_ENTITY_NAME_LENGTH
    if len(name) > _max_len:
        return False

    # 黑名单精确匹配
    if name in _NON_ENTITY_FRAGMENTS:
        return False

    # 系统状态泄露检查：不应出现系统/技术术语
    # 中文模式用子串匹配（不会误伤正常实体名）
    _system_patterns_cn = [
        "处理进度", "系统状态", "步骤", "缓存", "抽取", "对齐",
        "窗口", "流水线", "阈值", "配置", "待处理", "已完成",
    ]
    for pat in _system_patterns_cn:
        if pat in name:
            return False
    # 英文技术词用单词边界匹配（避免 "FastAPI" 被误杀）
    _system_patterns_en = [
        (r'\bToken\b', re.IGNORECASE),
        (r'\bAPI\b', re.IGNORECASE),
        (r'\bLLM\b', re.IGNORECASE),
        (r'\bJSON\b', re.IGNORECASE),
    ]
    for pat, flags in _system_patterns_en:
        if re.search(pat, name, flags):
            # 如果整个名称是纯英文技术名（如 "FastAPI"、"OpenAPI"），不应被误杀
            # 只拒绝那些把 API/Token/LLM 当作描述性片段使用的名称
            # 规则：如果名称中英文部分是一个知名的技术名词（camelCase/PascalCase），放行
            _stripped = re.sub(r'[（(][^）)]+[)）]', '', name).strip()
            if _stripped and re.match(r'^[A-Za-z][A-Za-z0-9_.+\-/]+$', _stripped):
                continue  # 纯英文技术名（如 FastAPI、OpenAPI），放行
            return False

    # 重复片段检查：同一2字片段重复3次以上（如"解读解读解读..."）
    for i in range(len(name) - 5):
        fragment = name[i:i+2]
        if len(fragment) == 2 and fragment == name[i+2:i+4] == name[i+4:i+6]:
            return False

    # 对话标记检查：包含古代/小说对话格式或感叹句
    _dialogue_markers = ["曰：", "道：", "笑曰", "怒曰", "叹曰", "言曰", "喝曰",
                         "答曰", "问曰", "叫曰", "说道", "说道：",
                         "哀哉", "痛哉", "惜哉", "壮哉"]
    for marker in _dialogue_markers:
        if marker in name:
            return False

    # 包含中文感叹号的引用/感叹句不是实体名称
    if "！" in name and len(name) > 4:
        return False

    # 包含中文顿号"、"的通常是枚举列表/句子片段，不是实体名
    if "、" in name:
        return False

    # 包含中文逗号"，"且长度>8的通常是句子片段/场景描述，不是实体名
    # 短名称含逗号可能是地名（如"长安，洛阳"），但>8字几乎一定是句子
    if "，" in name and len(name) > 8:
        return False

    # 包含"的"的结构（名词+的+名词）是描述性短语，不是实体名
    # 如"网络服务器、分布式系统和云计算基础设施的开发"
    # 但括号内的"的"是消歧信息的一部分，不应触发此规则
    # 如 "Marie Curie（波兰出生的物理学家和化学家）" 中"的"在括号内，应保留
    _core_for_de = re.sub(r'[（(][^）)]+[）)]', '', name).strip()
    if "的" in _core_for_de and len(_core_for_de) > 6:
        return False

    # 下划线分隔的模式通常是文件名/变量名/文档名，不是概念实体（如"test_go_v2"）
    # 匹配：含下划线，且以下划线分隔的某段是纯小写英文（技术标识符）
    if '_' in name and re.match(r'^[a-z][a-z0-9_]+$', name):
        return False

    # 检查括号内容是否是场景信息（仅限全角括号）
    paren_match = re.findall(r'（([^）]+)）', name)
    for paren_content in paren_match:
        # 如果括号内容是一个完整句子（含标点），说明不是消歧信息
        if any(c in paren_content for c in "，。！？、；："):
            return False
        # 系统信息泄露到括号中
        for pat in _system_patterns_cn:
            if pat in paren_content:
                return False

    # 纯年份/日期格式（如"2019年"、"2024年3月"）不是实体概念
    # 也覆盖 "1903年(1903)" 等双语年份格式
    # 但 "1903年诺贝尔物理学奖" 这种带实质内容的年份前缀应保留
    _year_match = re.match(r'^(\d{4}年?(?:\d{1,2}月?(?:\d{1,2}日?)?)?)', name)
    if _year_match:
        _year_part = _year_match.group(1)
        _after_year = name[len(_year_part):].strip()
        # 如果年份后面没有实质内容（纯数字/括号标注），则拒绝
        _after_clean = re.sub(r'[（(][^）)]+[)）]', '', _after_year).strip()
        if not _after_clean:
            return False

    # 提取纯中文核心名称（去掉英文括号标注后），用于通用词检测
    core_chinese = re.sub(r'\([A-Za-z\s]+\)$', '', name).strip()
    core_chinese = re.sub(r'（[A-Za-z\s]+）$', '', core_chinese).strip()

    # 过于宽泛的通用词汇（2-4字），不应作为独立实体
    _too_generic = {
        # 动词/动作
        "投资", "研究", "开发", "分析", "比较", "测试", "设计",
        "管理", "处理", "调整", "优化", "改进", "建立", "实现",
        "投资方", "研发", "探讨", "评估",
        "迁移", "合作研究", "开创性研究", "元素发现",
        # 泛化名词（无具体指代）
        "计算任务", "特定计算任务", "叠加态", "计算能力",
        "超越", "投资行为", "重要成就",
        "开源", "文档名", "文件名",
        "知名项目", "开源方式", "开源方式发布",
        # 技术/设计类泛化概念（4字以内）
        "设计理念", "设计目标", "设计发布",
        "编程效率", "提高编程效率",
        "静态强类型", "编译型语言",
        "网络服务器开发",
        "放射性研究", "科学历史", "科学遗产",
        # 动词/动作/状态（不是实体）
        "主持", "负责", "建议", "赞同", "遂令", "大悦",
        "修缮", "广纳", "修缮城防", "广纳贤才",
        # 过于宽泛的概念词
        "大业", "贤才", "百姓", "民心", "民生",
        "内政", "军事防务", "国事", "城防",
        "满意", "农业", "开源语言",
        # 科学/学术类泛化概念
        "科学成就", "科学贡献", "科学传承", "科学史", "科学合作",
        "科学影响", "科学教育", "科学界", "研究方法",
        "学术生涯", "科学界地位",
        "科学成就传播", "科学成就历史意义", "科学成就国际影响",
        "科学成就影响", "科学成就社会影响", "科学成就认可",
        "科学贡献评价",
        # 泛化动作/角色/关系词（中英文混合场景）
        "发现", "阅读", "合作", "丈夫", "妻子",
        "唯一", "开创性", "第一位", "元素",
        # 泛化情感/状态/过程
        "悲痛", "战败", "病逝", "病逝时间",
        "社会秩序", "继承权争夺", "统一北方",
        "军事和政治策略", "谋士",
        "军事行动", "悲痛之情", "谋士角色",
        "屯田制度实施",
        # 技术领域泛化概念/过程/状态
        "可移植", "可移植性", "大规模", "自动化", "自动部署",
        "扩展", "伸缩", "可伸缩", "可伸缩性",
        "资源使用", "资源管理", "资源优化",
        "容器化", "容器化应用", "容器集群",
        "多容器", "多容器应用",
    }
    if name in _too_generic or core_chinese in _too_generic:
        return False

    # 文档元数据模式（如"文件名:XXX""文档名:XXX"）
    if re.match(r'^(文件名|文档名|文档|文件|标题)[：:]', name):
        return False

    # 角色后缀模式（如"谋士角色""XX角色"）
    if name.endswith("角色") and len(name) <= 6:
        return False

    # 情感后缀模式（如"悲痛之情""XX之情"）
    if name.endswith("之情") and len(name) <= 6:
        return False

    # 英文通用词黑名单：直接匹配或作为双语名称的英文部分
    _english_generic = {
        "research", "discovery", "study", "analysis", "method",
        "partnership", "collaboration", "husband", "wife", "read",
        "influence", "impact", "education", "contribution",
        "achievement", "legacy", "history", "status", "role",
        "first", "pioneering", "only", "one and only", "elements",
        "new elements", "pioneering research", "scientific contribution",
        "academic studies", "discovery of new elements",
        "scientific history", "scientific heritage", "scientific legacy",
        "radioactive research", "medical assistance", "women scientists",
    }
    # 提取括号内的英文部分
    _en_paren = re.findall(r'[(（]([A-Za-z][A-Za-z\s]+)[)）]', name)
    for en in _en_paren:
        if en.strip().lower() in _english_generic:
            return False
    # 纯英文名称也检查
    if name.isascii() and name.lower().strip() in _english_generic:
        return False

    # 英文泛化模式：以动作/过程词结尾的短语（如 "Pioneering research"、"Discovery of ..."）
    if name.isascii():
        _en_generic_start = ("pioneering ", "discovery of ", "study of ",
                             "analysis of ", "history of ", "impact of ")
        _en_generic_end = (" research", " discovery", " contribution",
                           " achievement", " method", " study",
                           " analysis", " collaboration", " partnership",
                           " education", " influence", " impact",
                           " legacy", " history", " studies")
        name_lower = name.lower().strip()
        for start in _en_generic_start:
            if name_lower.startswith(start):
                return False
        for end in _en_generic_end:
            if name_lower.endswith(end):
                return False

    # 模式匹配：泛化概念组合（如"科学成就XX""科学贡献XX"等）
    _generic_prefixes = [
        "科学成就", "科学贡献", "科学界", "学术", "研究",
        "设计",
    ]
    for prefix in _generic_prefixes:
        if name.startswith(prefix) and len(name) > len(prefix) and len(name) <= 8:
            # 检查后缀是否是常见的泛化修饰词
            suffix = name[len(prefix):]
            _generic_suffixes = {
                "传播", "历史意义", "国际影响", "影响", "社会影响",
                "认可", "评价", "地位", "发展", "意义", "价值",
                "概述", "总结", "背景", "特点", "方法",
                "理念", "目标", "发布", "思想", "方向", "趋势",
            }
            if suffix in _generic_suffixes:
                return False

    # 过于泛化的动作/过程描述（如"XX发现""XX研究"）
    # 同时检查原始名称和去掉英文括号后的核心名称
    _action_suffixes = ["研究", "发现", "探索", "分析", "评价", "讨论", "传播"]
    _check_name = core_chinese or name
    if len(_check_name) >= 4 and len(_check_name) <= 6:
        for suffix in _action_suffixes:
            if _check_name.endswith(suffix):
                # 如果前缀也是泛化词（如"元素""开创性"），则拒绝
                prefix = _check_name[:-len(suffix)]
                _generic_action_prefixes = {
                    "元素", "开创性", "合作", "系统", "深入", "初步",
                    "全面", "综合", "整体", "长期", "持续", "新",
                }
                if prefix in _generic_action_prefixes:
                    return False

    # 动词短语/事件描述模式检测（如"恢复农业生产""统一北方""郭嘉病逝"）
    # 这些是动作+宾语或主语+动词的结构，不是实体概念
    _verb_phrase_patterns = [
        # 动词+名词模式（如"恢复XX""稳定XX""统一XX""继承权争斗"）
        r'^(恢复|稳定|统一|建立|推行|实施|开展|加强|推进|促进|巩固|维护|保障|完成|实现|争取|争夺|继承权)',
        # 提高XX/提升XX（如"提高编程效率""提升系统性能"）
        r'^(提高|提升|增强|改善|优化|加速)',
        # 自动XX（如"自动部署""自动扩展""自动化测试"）
        r'^自动(化|部署|扩展|伸缩|管理|配置|监控)',
        # 名词+动词/事件模式（如"郭嘉病逝""XX去世"）
        r'^(感慨|内斗)',
        r'病逝$',
        r'去世$',
        r'争斗$',
        # "感慨的话语"类型的描述性短语
        r'的话语$',
    ]
    for pat in _verb_phrase_patterns:
        if re.search(pat, core_chinese or name):
            return False

    return True


def _clean_entity_name(name: str) -> str:
    """清理实体名称中的场景标注括号。

    策略：默认保留括号内容（作为消歧信息），仅在明确判定为场景标注时移除。
    场景标注的特征：包含其他实体名、地名+事件、动作片段等临时上下文信息。

    例如：
    - "曹操（汉中张鲁）" → "曹操"  （括号内容是场景/事件信息，应移除）
    - "曹操（魏王）" → "曹操（魏王）" （括号内容是身份/类型信息，应保留）
    - "杨昂（诸营）" → "杨昂" （括号内容是场景上下文，应移除）
    - "许褚（曹操亲卫）" → "许褚（曹操亲卫）" （括号内容是持久关系，应保留）
    """
    import re

    def _is_scene_annotation(content: str) -> bool:
        """判断括号内容是否是场景/事件标注（而非消歧信息）。

        场景标注特征（应移除）：
        - 包含句号、逗号等标点 → 事件描述
        - 以动词开头 → 动作片段（如"众官告免""忽寨后一把火起"）
        - 纯地名/纯人名（2-4字的专有名词，且不含关系词）
        - 军事场景描述（地名+营/寨/关/城/阵 等）

        消歧信息特征（应保留）：
        - 含关系词（的、之、里）→ 持久关系描述
        - 身份后缀（王、侯、将、师等）→ 角色标识
        - 领域/类型后缀 → 类型归属
        - 章节标识 → 结构标记
        """
        content = content.strip()
        if not content:
            return True  # 空括号移除

        # 包含句号、逗号等长标点 → 事件描述，移除
        if any(c in content for c in "，。！？、；："):
            return True

        # 明确的身份/角色/关系模式 → 保留
        _identity_patterns = [
            r'[\w]+[之的里亲][\w]+',    # "XX的XX"、"XX之XX"、"XX亲XX" — 关系描述
            r'[\w]*[者师家官将相帝侯公卿]$',  # 以身份后缀结尾
            r'[\w]*[王]$',             # 以"王"结尾（但不是"张鲁"之类的名字）
            r'[\w]+[语言地点物书作品学]$',  # 以类型/领域后缀结尾（含"学"）
            r'[\d]+[-~—][\d]+',       # 时间范围 "1368-1644"
            r'第[一二三四五六七八九十百千万零\d]+[卷回章节]',  # 章节标识
            r'[\w]*[卫兵护卫从随]$',    # 以关系/从属后缀结尾（"亲卫""护卫""侍从"）
            r'[\w]*[丫头仆婢妾妃]$',   # 以家庭角色后缀结尾
        ]
        for pattern in _identity_patterns:
            if re.search(pattern, content):
                return False  # 匹配身份模式，不是场景标注

        # 短内容（≤3字）：军事/场景关键词 → 移除
        if len(content) <= 3:
            _scene_short_words = {"诸营", "中军", "前军", "后军", "左军", "右军",
                                  "大营", "前寨", "后寨", "上关", "下关"}
            if content in _scene_short_words:
                return True

        # 含军事场景关键词 → 移除
        _scene_context_keywords = [
            "营", "寨", "关", "阵", "兵", "军", "战", "守", "攻", "围",
            "破", "败", "退", "降", "逃", "追", "击", "杀", "死", "伤",
            "火起", "告免", "出马", "迎敌", "交锋",
        ]
        for kw in _scene_context_keywords:
            if kw in content:
                # 但如果同时也含关系词或身份后缀，保留
                if any(c in content for c in "的之里") or \
                   re.search(r'[者师家官将相帝侯公卿王语言地点物书作品]$', content):
                    return False
                return True

        # 以动词/动作词开头的短片段（≤6字）→ 可能是场景片段
        if len(content) <= 6:
            _verb_starts = ("众", "忽", "遂", "相", "火", "弃", "突", "失")
            if any(content.startswith(v) for v in _verb_starts):
                return True

        # 最终兜底：纯中文专有名词（2-6字），不含关系/身份标记 → 很可能是场景标注
        # 场景标注的常见形式：括号内是另一个实体名（人名/地名）或场景上下文
        # 身份消歧的常见形式：括号内含关系词、角色后缀、领域后缀等
        if re.match(r'^[\u4e00-\u9fff]{2,6}$', content):
            # 已排除了含关系词（的之里亲）、身份后缀、领域后缀的情况
            # 如果走到这里，说明是纯专有名词（人名/地名），作为场景上下文移除
            return True

        return False  # 默认保留

    # 处理全角括号
    def _replace_fullwidth(m):
        inner = m.group(1)
        return f"（{inner}）" if not _is_scene_annotation(inner) else ""

    cleaned = re.sub(r'（([^）]+)）', _replace_fullwidth, name)

    # 处理半角括号
    def _replace_halfwidth(m):
        inner = m.group(1)
        return f"({inner})" if not _is_scene_annotation(inner) else ""

    cleaned = re.sub(r'\(([^)]+)\)', _replace_halfwidth, cleaned)

    return cleaned.strip() or name.strip()


def _core_entity_name(name: str) -> str:
    """提取实体名称的核心部分（去掉所有括号），用于去重比较。

    例如："曹操（魏王）" → "曹操"，"许褚（曹操亲卫）" → "许褚"
    """
    import re
    return re.sub(r'[（(][^）)]+[）)]', '', name).strip()


def dedupe_extracted_entities(entities: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """按实体 name（strip 后）去重；同名时保留 content 更长的条目。同时过滤非实体文本片段。

    去重策略：先按清洗后名称精确匹配，再按核心名称（去掉所有括号）匹配。
    核心名称匹配时，保留带消歧括号的名称（如"曹操（魏王）"优先于"曹操"）。
    """
    name_to_index: Dict[str, int] = {}
    core_name_to_index: Dict[str, int] = {}
    out: List[Dict[str, str]] = []
    for e in entities or []:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name") or "").strip()
        if not name:
            continue

        # 清理场景标注括号
        cleaned_name = _clean_entity_name(name)
        if cleaned_name != name:
            # 如果清理后名称改变，使用清理后的名称
            name = cleaned_name

        # 检查名称是否是有效的实体名称
        if not _is_valid_entity_name(name):
            continue

        content = str(e.get("content") or "").strip()
        # 检查content是否过短（<8字的content通常是无效描述）
        if len(content) < 8:
            continue

        # 第一轮：按清洗后名称精确匹配
        existing_idx = name_to_index.get(name)
        if existing_idx is not None:
            if len(content) > len(out[existing_idx]["content"]):
                out[existing_idx] = {"name": name, "content": content}
            continue

        # 第二轮：按核心名称匹配（去掉所有括号后比较）
        core_name = _core_entity_name(name)
        core_idx = core_name_to_index.get(core_name)
        if core_idx is not None:
            # 核心名称匹配到已有实体，合并
            existing = out[core_idx]
            # 保留带消歧括号的名称（更具体的名称优先）
            if len(name) > len(existing["name"]):
                existing["name"] = name
            # 更新名称映射
            name_to_index[name] = core_idx
            if len(content) > len(existing["content"]):
                existing["content"] = content
            continue

        # 新实体
        idx = len(out)
        name_to_index[name] = idx
        if core_name != name:
            core_name_to_index[core_name] = idx
        out.append({"name": name, "content": content})
    return out


def _is_valid_relation_content(content: str, entity1_name: str = "", entity2_name: str = "") -> bool:
    """检查关系内容是否是有效的关系描述。

    过滤规则：
    1. 最小长度：关系内容至少8个字符
    2. 禁止纯标签式描述（如"官職"、"居住地"、"主从关系"）
    3. 禁止模板化空洞描述（如"XX和YY的关联关系"、"XX与YY有关"）
    """
    if not content or len(content) < 8:
        return False

    # 纯标签式内容黑名单（只有类型标签，没有具体内容）
    _label_only_patterns = {
        "官職", "居住地", "主从关系", "同僚关系", "敌对关系",
        "合作关系", "亲属关系", "上下级关系", "竞争关系",
        "朋友关系", "师徒关系", "同事关系", "邻居关系",
        "所属关系", "从属关系", "包含关系", "依赖关系",
    }
    if content.strip() in _label_only_patterns:
        return False

    # 检查是否是空洞模板（只说"有关联"而不描述具体内容）
    import re
    _empty_patterns = [
        r'^.{0,4}与.{0,4}的?关联关系?$',
        r'^.{0,4}与.{0,4}有关$',
        r'^.{0,4}和.{0,4}的关系?$',
        r'^.{0,4}与.{0,4}存在关联$',
        r'^.{0,4}和.{0,4}相关$',
    ]
    for pattern in _empty_patterns:
        if re.match(pattern, content.strip()):
            return False

    return True


def dedupe_extracted_relations(relations: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """关系去重：无向 (entity1, entity2) 字典序 + content（忽略大小写）。同时过滤低质量关系。"""
    seen: set[Tuple[str, str, int]] = set()
    out: List[Dict[str, str]] = []
    for r in relations or []:
        if not isinstance(r, dict):
            continue
        e1 = str(r.get("entity1_name") or "").strip()
        e2 = str(r.get("entity2_name") or "").strip()
        content = str(r.get("content") or "").strip()
        if not e1 or not e2 or not content:
            continue
        # 过滤自关系
        if e1 == e2:
            continue
        # 质量检查：过滤低质量关系内容
        if not _is_valid_relation_content(content, e1, e2):
            continue
        n1, n2 = _normalize_pair_for_relation(e1, e2)
        key = (n1, n2, hash(content.lower()))
        if key in seen:
            continue
        seen.add(key)
        out.append({"entity1_name": n1, "entity2_name": n2, "content": content})
    return out


def dedupe_extraction_lists(
    entities: Optional[List[Dict[str, Any]]],
    relations: Optional[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """供缓存加载等场景：实体、关系各做一次列表级去重。"""
    return dedupe_extracted_entities(entities), dedupe_extracted_relations(relations)


def _dedupe_by_content_similarity(
    entities: List[Dict[str, str]],
    relations: List[Dict[str, str]],
    jaccard_threshold: float = 0.65,
) -> List[Dict[str, str]]:
    """按内容相似度去重：如果多个实体的content高度相似，说明它们都在描述同一主体而非自身。

    策略：
    1. 对每对实体计算 content 的词级 Jaccard 相似度
    2. 如果相似度 > threshold，将它们分组
    3. 每组只保留最"有价值"的实体：有关系连接的优先，然后选 content 最长的
    """
    if len(entities) <= 1:
        return entities

    # 收集关系端点，用于判断实体是否"有连接"
    related_names: set = set()
    for rel in relations:
        for key in ('entity1_name', 'entity2_name'):
            name = rel.get(key, '').strip()
            if name:
                related_names.add(name)

    def _word_set(text: str) -> set:
        """简单的词级切分（中文按字，英文按空格）。"""
        words = set()
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                words.add(char)
            elif char.isalnum():
                words.add(char.lower())
        return words

    # 计算所有 content 的词集合
    content_words = [_word_set(e.get('content', '')) for e in entities]

    # Union-Find 分组
    parent = list(range(len(entities)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # 对每对实体比较 content
    for i in range(len(entities)):
        if not content_words[i]:
            continue
        for j in range(i + 1, len(entities)):
            if not content_words[j]:
                continue
            intersection = content_words[i] & content_words[j]
            union_set = content_words[i] | content_words[j]
            if union_set:
                jaccard = len(intersection) / len(union_set)
                if jaccard > jaccard_threshold:
                    union(i, j)

    # 按组合并
    groups: Dict[int, List[int]] = {}
    for i in range(len(entities)):
        root = find(i)
        groups.setdefault(root, []).append(i)

    out: List[Dict[str, str]] = []
    for indices in groups.values():
        if len(indices) == 1:
            out.append(entities[indices[0]])
            continue
        # 多个实体内容高度相似：保留最有价值的一个
        def _entity_score(idx: int) -> tuple:
            """评分：(是否关系端点, content长度负值=越长越好)"""
            name = entities[idx].get('name', '')
            is_related = name in related_names
            content_len = len(entities[idx].get('content', ''))
            return (is_related, content_len)

        best_idx = max(indices, key=_entity_score)
        out.append(entities[best_idx])

    return out


# 过于泛化的名称模式：这些名称几乎不可能是有价值的具体实体
# 它们通常是 LLM 把文本中的某个词/短语错误地当作实体名
_OVER_GENERIC_NAMES: set = {
    # 泛化名词（非具体实体）
    "共同", "发展", "分离", "技术", "理论", "方法", "过程",
    "贡献", "成就", "影响", "挑战", "困难", "机会", "问题",
    "研究", "分析", "评价", "讨论", "探索", "传播", "概述",
    "总结", "背景", "特点", "意义", "价值", "地位", "方向",
    "趋势", "理念", "目标", "思想",
    # 描述性短语（不是实体名）
    "两次", "唯一", "第一位", "开创性贡献", "杰出成就",
    "国际性的奖项", "科学领域", "物理现象", "研究过程",
    "后世影响", "挑战和困难",
    # 单字泛化词
    "唯一", "首位",
    # 过于泛化的身份描述（不是具体人名/地名/作品名）
    "女性", "丈夫", "妻子", "科学家", "物理学家", "化学家",
    "物理学家和化学家",
}

# 泛化名称模式：2-3字的描述性短语
_OVER_GENERIC_PATTERNS = [
    r'^[\u4e00-\u9fff]{1,2}$',  # 1-2字纯中文（太短，几乎不可能是具体实体）
]


def _is_self_consistent_entity(name: str, content: str, all_entities: List[Dict[str, str]],
                                source_text: str = "") -> bool:
    """检查实体的 name 和 content 是否自洽。

    自洽性规则：
    1. 泛化名称黑名单检查（"女性"、"技术"、"发展"等）
    2. 内容主题检查：如果 content 主要描述的是另一个实体（而非 name 本身），则不自洽
    3. 名称长度检查：单字名称几乎不可能是有效实体
    4. 源文本锚定检查：名称的核心部分必须出现在源文本中（防止LLM幻觉生成泛化概念）

    Args:
        name: 实体名称
        content: 实体描述
        all_entities: 所有已抽取实体列表（用于检测内容是否描述了其他实体）
        source_text: 源文本（用于锚定检查）

    Returns:
        True 如果实体自洽（name 和 content 描述同一概念）
    """
    import re

    if not name or not content:
        return False

    # 规则1：泛化名称黑名单
    # 提取核心名称（去掉英文括号标注后）
    core_name = re.sub(r'\([A-Za-z\s]+\)$', '', name).strip()
    core_name = re.sub(r'（[A-Za-z\s]+）$', '', core_name).strip()
    if name in _OVER_GENERIC_NAMES or core_name in _OVER_GENERIC_NAMES:
        return False

    # 规则2：单字纯中文名称（几乎不可能是有效实体）
    if len(name) == 1 and re.match(r'^[\u4e00-\u9fff]+$', name):
        return False

    # 规则3：2字纯中文名称检查
    if len(name) == 2 and re.match(r'^[\u4e00-\u9fff]+$', name):
        # 2字名词：只保留人名、地名、作品名等专有名词
        _generic_2char = {
            "共同", "发展", "分离", "技术", "理论", "方法", "过程",
            "贡献", "成就", "影响", "挑战", "研究", "分析", "评价",
            "讨论", "探索", "传播", "概述", "总结", "背景", "特点",
            "意义", "价值", "地位", "方向", "趋势", "理念", "目标",
            "思想", "两次", "唯一", "首位", "女性", "丈夫", "妻子",
            "科学", "物理", "化学", "数学", "生物", "医学", "文学",
            "哲学", "历史", "社会", "政治", "经济", "文化", "教育",
            "技术", "工程", "艺术", "音乐", "体育", "法律", "军事",
            "商业", "工业", "农业", "环境", "能源", "材料", "信息",
            "管理", "设计", "创新", "质量", "效率", "安全", "稳定",
            "增长", "下降", "变化", "改进", "优化", "提升", "改善",
        }
        if name in _generic_2char:
            return False

    # 规则4：内容主题不一致检查
    # 如果 content 的主体（最频繁出现的具体实体名）不是当前 name，
    # 说明这个 content 描述的是另一个实体，当前实体是错误抽取
    # 放宽条件：
    # - 如果 content 足够长（>80字），即使提及其他实体也是正常的详细描述
    # - 如果 name 在源文本中直接出现，说明是真实实体，不应被过滤
    _name_in_source = False
    if source_text and len(name) >= 2:
        _name_core = re.sub(r'[(（][^)）]+[)）]', '', name).strip()
        if _name_core:
            if _name_core in source_text:
                _name_in_source = True
            # 模糊匹配：中文名称的子串可能被拆分出现在源文本中
            # 如"量子纠缠"：源文本中有"量子"和"纠缠"（分别在"量子比特"和"纠缠态"中）
            # 检查名称中>=2字的连续片段是否出现在源文本中
            elif len(_name_core) >= 3:
                _frag_size = max(2, len(_name_core) // 2)
                for i in range(len(_name_core) - _frag_size + 1):
                    _frag = _name_core[i:i+_frag_size]
                    if _frag in source_text:
                        _name_in_source = True
                        break
    if all_entities and len(content) > 20 and len(content) <= 80 and not _name_in_source:
        # 收集其他实体的名称，检测 content 是否主要描述了其他实体
        other_entity_names = []
        for e in all_entities:
            ename = e.get('name', '')
            if ename and ename != name:
                other_entity_names.append(ename)

        if other_entity_names:
            # 检查 content 中是否包含了其他实体的完整名称
            _mention_count = 0
            _mentioned_other = None
            for other_name in other_entity_names:
                # 只检查名称>=2个字符的（避免误匹配）
                if len(other_name) >= 2 and other_name in content:
                    _mention_count += 1
                    _mentioned_other = other_name

            # 如果 content 提到了多个其他实体名，且 name 本身是泛化词，
            # 很可能这个 content 是关于其他实体的描述
            if _mention_count >= 2 and len(name) <= 4:
                # 额外检查：name 本身不出现在 content 中（或出现次数少）
                name_mentions = content.count(name)
                if name_mentions == 0:
                    return False

    # 规则5：源文本锚定检查
    # 如果提供了源文本，验证实体的核心名称是否出现在源文本中。
    # 有效实体名（去除消歧括号后）必须是源文本中的子串（直接出现或作为翻译名）。
    # 这个检查能捕获LLM幻觉生成的泛化概念，如"理解"、"个人生活"、"性别偏见"等
    # 这些概念在源文本中根本不存在，是LLM凭空编造的。
    if source_text:
        # 提取核心名称（去掉括号标注后）
        _core = re.sub(r'\([A-Za-z\s]+\)$', '', name).strip()
        _core = re.sub(r'（[A-Za-z\s]+）$', '', _core).strip() or name
        # 去掉括号内的英文标注
        _core_clean = re.sub(r'[(（][^)）]+[)）]', '', _core).strip()
        if _core_clean:
            # 直接检查核心名称是否在源文本中出现
            if _core_clean in source_text:
                return True  # 名称在源文本中，通过
            # 子串包含检查：如果名称是源文本中某个词的子串，也通过
            # 如"纠缠"是"纠缠态"的子串，"九章"是""九章""中的子串
            source_lower = source_text.lower() if _core_clean.isascii() else source_text
            check_val = _core_clean.lower() if _core_clean.isascii() else _core_clean
            # 检查名称是否作为子串出现在源文本中
            if len(check_val) >= 2 and check_val in source_lower:
                return True
            # 中文复合名称子串匹配：检查名称中>=2字的连续片段是否在源文本中
            # 如"量子纠缠"→"量子"在"量子比特"中，"纠缠"在"纠缠态"中
            if not _core_clean.isascii() and len(_core_clean) >= 3:
                _frag_size = max(2, len(_core_clean) // 2)
                for i in range(len(_core_clean) - _frag_size + 1):
                    _frag = _core_clean[i:i+_frag_size]
                    if _frag in source_text:
                        return True
            # 英文名称检查：源文本中是否有对应的英文单词
            if _core_clean.isascii():
                # 英文名：不区分大小写检查
                if _core_clean.lower() in source_text.lower():
                    return True
                # 多词英文名：检查每个词是否都在源文本中（处理复合概念）
                words = _core_clean.lower().split()
                if len(words) > 1 and all(w in source_text.lower() for w in words):
                    return True
            # 中英别名检查：常见的中英对照名称
            # 如 "Google" 对应 "谷歌"，"IBM" 对应 "国际商业机器"
            _cn_en_aliases = {
                "Google": ["谷歌"], "IBM": ["国际商业机器"], "Microsoft": ["微软"],
                "Apple": ["苹果"], "Amazon": ["亚马逊"], "Meta": ["脸书", "元宇宙"],
                "Tesla": ["特斯拉"], "NVIDIA": ["英伟达"], "Intel": ["英特尔"],
                "OpenAI": ["开放AI"], "Quantum": ["量子"],
            }
            _aliases = _cn_en_aliases.get(_core_clean, [])
            if _aliases:
                for alias in _aliases:
                    if alias in source_text:
                        return True
            # 对于非直接出现的中文名称，检查是否有括号内的英文标注
            # 如 "放射性（radioactivity）" — 即使中文不在文本中，英文在也可接受
            _en_in_paren = re.findall(r'[(（]([A-Za-z][A-Za-z\s]+)[)）]', name)
            for en in _en_in_paren:
                if en.strip().lower() in source_text.lower():
                    return True
            # content 锚定：如果名称出现在其他已抽取实体的 content 中，说明它是文本中的隐含概念
            if all_entities:
                for e in all_entities:
                    if e.get('name', '') != name and _core_clean in e.get('content', ''):
                        return True
            # 名称不在源文本中，且名称足够短（<=4字），很可能是LLM幻觉的泛化概念
            # 放宽阈值到4字（原8字过于激进，过滤了有效实体如"潘建伟"、"九章"等）
            if len(_core_clean) <= 4:
                return False
            # 较长名称（>4字）可能是描述性实体，不做锚定检查

    return True


def _filter_self_consistent_entities(
    entities: List[Dict[str, str]],
    relations: List[Dict[str, str]],
    source_text: str = "",
) -> List[Dict[str, str]]:
    """过滤不自洽的实体：name 和 content 不匹配的实体。

    这是对 LLM 抽取质量的后验校验，纯代码实现，零 LLM 调用。
    """
    if len(entities) <= 1:
        return entities

    # 收集关系端点名（有关系连接的实体优先级更高）
    related_names: set = set()
    for rel in relations:
        for key in ('entity1_name', 'entity2_name'):
            n = rel.get(key, '').strip()
            if n:
                related_names.add(n)

    out: List[Dict[str, str]] = []
    _filtered = 0
    for e in entities:
        name = e.get('name', '')
        content = e.get('content', '')

        if not _is_self_consistent_entity(name, content, entities, source_text):
            # 有关系连接的实体放宽标准（即使名称泛化，如果有关系连接则保留）
            if name in related_names:
                out.append(e)
            else:
                _filtered += 1
            continue

        out.append(e)

    return out


@dataclass
class _AlignResult:
    """步骤6（实体对齐）的输出，供步骤7使用。"""
    entity_name_to_id: Dict[str, str] = field(default_factory=dict)
    pending_relations: List[Dict] = field(default_factory=list)
    unique_entities: List[Entity] = field(default_factory=list)
    unique_pending_relations: List[Dict] = field(default_factory=list)


class _ExtractionMixin:
    """抽取相关流水线步骤（mixin，通过 TemporalMemoryGraphProcessor 多继承使用）。"""

    def _update_cache(self, input_text: str, document_name: str,
                      text_start_pos: int = 0, text_end_pos: int = 0,
                      total_text_length: int = 0, verbose: bool = True,
                      verbose_steps: bool = True,
                      document_path: str = "",
                      event_time: Optional[datetime] = None,
                      window_index: int = 0, total_windows: int = 0) -> Episode:
        """步骤1：更新记忆缓存。必须在 _cache_lock 下调用，保证 cache 链串行。"""
        self.llm_client._priority_local.priority = LLM_PRIORITY_STEP1
        if verbose:
            wprint("【步骤1】缓存｜开始｜")
        elif verbose_steps:
            wprint("【步骤1】缓存｜开始｜")

        # 蒸馏数据准备：确保 task_id 在步骤1前生成
        if self.llm_client._distill_data_dir:
            if not self.llm_client._distill_task_id:
                self.llm_client._distill_task_id = f"{document_name}_{uuid.uuid4().hex[:8]}_{int(time.time() * 1000)}"
            self.llm_client._current_distill_step = "01_update_cache"

        new_episode = self.llm_client.update_episode(
            self.current_episode,
            input_text,
            document_name=document_name,
            text_start_pos=text_start_pos,
            text_end_pos=text_end_pos,
            total_text_length=total_text_length,
            event_time=event_time,
            window_index=window_index,
            total_windows=total_windows,
        )

        self.llm_client._current_distill_step = None

        doc_hash = compute_doc_hash(input_text) if input_text else ""
        self.storage.save_episode(new_episode, text=input_text, document_path=document_path, doc_hash=doc_hash)
        self.current_episode = new_episode

        if verbose:
            wprint(f"【步骤1】缓存｜写入｜ID {new_episode.absolute_id}")
        elif verbose_steps:
            wprint("【步骤1】缓存｜完成｜已更新")

        return new_episode

    # =========================================================================
    # 步骤2-5：抽取实体/关系/补全/增强（只读，不写存储，可并行）
    # =========================================================================

    def _extract_only(self, new_episode: Episode, input_text: str,
                      document_name: str, verbose: bool = True,
                      verbose_steps: bool = True,
                      event_time: Optional[datetime] = None,
                      progress_callback=None,
                      progress_range: tuple = (0.1, 0.5),
                      window_index: int = 0,
                      total_windows: int = 1) -> Tuple[List[Dict], List[Dict]]:
        """步骤2-3合并+3.5+5：联合抽取实体/关系，反哺，增强。不写存储，可在线程池中并行执行。

        Returns:
            (extracted_entities, extracted_relations) — 纯字典列表，不含 family_id。
        """

        p_lo, p_hi = progress_range
        _win_label = f"窗口 {window_index + 1}/{total_windows}"
        _steps = 3  # 合并抽取(2+3) + 反哺(3.5) + 增强(5) 共3步
        _step_size = (p_hi - p_lo) / _steps

        def _report_step(step_idx: int, label: str, message: str):
            if progress_callback:
                p = p_lo + _step_size * (step_idx + 1)
                _step_labels = ["合并抽取实体/关系", "反哺", "实体增强"]
                _step_name = _step_labels[step_idx] if step_idx < len(_step_labels) else label
                progress_callback(p,
                    f"{_win_label} · {_step_name}",
                    message)

        def _report_intermediate(step_idx: int, frac: float, label: str, message: str):
            if progress_callback:
                p = p_lo + _step_size * (step_idx + frac)
                progress_callback(p,
                    f"{_win_label} · {label}",
                    message)

        # ========== 步骤2+3合并：联合抽取实体和关系 ==========
        self.llm_client._priority_local.priority = LLM_PRIORITY_STEP2
        if verbose:
            wprint("【步骤2】实体｜开始｜")
        elif verbose_steps:
            wprint("【步骤2】实体｜开始｜")

        self.llm_client._current_distill_step = "02_extract_entities"

        # 自适应轮次：根据文本长度动态调整抽取轮次
        # 短文本（<500字符）→ 1轮，中文（500-2000）→ 2轮，长文本（>2000）→ 配置值
        _text_len = len(input_text) if input_text else 0
        _configured_rounds = max(self.entity_extraction_rounds, getattr(self, 'relation_extraction_rounds', 1))
        if _text_len < 500:
            _combined_rounds = 1
        elif _text_len < 2000:
            _combined_rounds = min(2, _configured_rounds)
        else:
            _combined_rounds = _configured_rounds
        if verbose and _combined_rounds != _configured_rounds:
            wprint(f"【步骤2】轮次｜自适应｜文本{_text_len}字→{_combined_rounds}轮（配置{_configured_rounds}）")

        extracted_entities = []
        extracted_relations = []

        def _on_combined_round(round_done, total_rounds, entity_count, rel_count):
            frac = round_done / max(1, total_rounds)
            _report_intermediate(0, frac,
                    f"合并抽取 ({round_done}/{total_rounds})",
                    f"第 {round_done}/{total_rounds} 轮，累计 {entity_count} 实体 {rel_count} 关系")

        extracted_entities, extracted_relations = self.llm_client.extract_entities_and_relations(
            new_episode, input_text,
            rounds=_combined_rounds,
            verbose=verbose,
            strict=True,
            on_round_done=_on_combined_round,
        )

        # 关系为空时重试1次：可能是 LLM 首轮只输出实体忘记输出关系
        if not extracted_relations and extracted_entities:
            wprint(f"【步骤2】警告｜关系空｜重试1次")
            _retry_ents, extracted_relations = self.llm_client.extract_entities_and_relations(
                new_episode, input_text,
                rounds=1,
                verbose=verbose,
                strict=True,
                on_round_done=_on_combined_round,
            )
            # 仅采纳重试的关系结果，实体保留首次结果（首次已经过验证）
            if not extracted_relations:
                extracted_relations = []
        elif not extracted_entities and not extracted_relations:
            wprint(f"【步骤2】警告｜空结果｜重试1次")
            extracted_entities, extracted_relations = self.llm_client.extract_entities_and_relations(
                new_episode, input_text,
                rounds=1,
                verbose=verbose,
                strict=True,
                on_round_done=_on_combined_round,
            )

        self.llm_client._current_distill_step = None

        _ent_count = len(extracted_entities)
        _rel_count = len(extracted_relations)
        if verbose:
            wprint(f"【步骤2】实体｜完成｜{_ent_count}个")
        elif verbose_steps:
            wprint(f"【步骤2】实体｜完成｜{_ent_count}个")
        dbg(f"步骤2+3合并完成: 抽取到 {_ent_count} 个实体, {_rel_count} 个关系")
        for _ei, _e in enumerate(extracted_entities):
            dbg(f"  实体[{_ei}]: name='{_e.get('name', '')}'  content='{_e.get('content', '')[:80]}'")
        for _ri, _r in enumerate(extracted_relations):
            dbg(f"  关系[{_ri}]: '{_r.get('entity1_name', '')}' <-> '{_r.get('entity2_name', '')}'  content='{_r.get('content', '')[:100]}'")
        if verbose:
            if _ent_count == 0:
                wprint(f"【步骤2】警告｜跳过｜重试{_extraction_empty_retry}次仍空")
            else:
                wprint(f"【步骤2】实体｜小结｜共{_ent_count}个")
        _report_step(0, "合并抽取实体/关系", f"抽取到 {_ent_count} 个实体, {_rel_count} 个关系")

        _ents_before_dedup = len(extracted_entities)
        extracted_entities = dedupe_extracted_entities(extracted_entities)
        if verbose and len(extracted_entities) != _ents_before_dedup:
            wprint(
                f"【步骤2】去重｜实体｜{_ents_before_dedup}→{len(extracted_entities)}"
            )
        dbg(
            f"步骤2后实体去重: {_ents_before_dedup} → {len(extracted_entities)}"
        )

        _rels_before_dedup = len(extracted_relations)
        extracted_relations = dedupe_extracted_relations(extracted_relations)
        if verbose and len(extracted_relations) != _rels_before_dedup:
            wprint(
                f"【步骤3】去重｜关系｜{_rels_before_dedup}→{len(extracted_relations)}"
            )
        dbg(
            f"步骤3后关系去重: {_rels_before_dedup} → {len(extracted_relations)}"
        )

        # ========== 步骤2.5：内容相似度去重 ==========
        # 如果多个实体的content高度相似（Jaccard > 0.7），说明它们都在描述同一件事
        # 而不是各自描述自己，保留最有关系的那个
        if len(extracted_entities) > 1:
            _before_content_dedup = len(extracted_entities)
            extracted_entities = _dedupe_by_content_similarity(extracted_entities, extracted_relations)
            if verbose and len(extracted_entities) != _before_content_dedup:
                wprint(
                    f"【步骤2.5】内容去重｜{_before_content_dedup}→{len(extracted_entities)}"
                )
            dbg(
                f"步骤2.5内容相似度去重: {_before_content_dedup} → {len(extracted_entities)}"
            )

        # ========== 步骤2.7：实体自洽性校验 ==========
        # 纯代码检查：name 和 content 是否描述同一概念
        # 过滤掉"女性"（content 全是 Marie Curie 描述）这类不自洽实体
        if len(extracted_entities) > 1:
            _before_consistency = len(extracted_entities)
            extracted_entities = _filter_self_consistent_entities(
                extracted_entities, extracted_relations, input_text
            )
            if verbose and len(extracted_entities) != _before_consistency:
                wprint(
                    f"【步骤2.7】自洽性｜{_before_consistency}→{len(extracted_entities)}"
                )
            dbg(
                f"步骤2.7自洽性校验: {_before_consistency} → {len(extracted_entities)}"
            )

        # ========== 步骤2.8：关系端点清洗 ==========
        # 移除引用了被过滤实体名称的关系（实体已不存在，关系端点无法解析）
        if extracted_relations:
            _valid_names = {e['name'] for e in extracted_entities}
            # 同时构建核心名称集合，处理括号不一致的情况
            _valid_core_names = {_core_entity_name(n) for n in _valid_names}
            _before_rel_cleanup = len(extracted_relations)
            _cleaned_relations = []
            for rel in extracted_relations:
                e1 = rel.get('entity1_name', '').strip()
                e2 = rel.get('entity2_name', '').strip()
                # 精确匹配或核心名称匹配
                e1_ok = e1 in _valid_names or _core_entity_name(e1) in _valid_core_names
                e2_ok = e2 in _valid_names or _core_entity_name(e2) in _valid_core_names
                if e1_ok and e2_ok:
                    # 标准化端点名称：如果关系端点不在精确集合中但在核心集合中，替换为精确名
                    if e1 not in _valid_names:
                        core1 = _core_entity_name(e1)
                        for vn in _valid_names:
                            if _core_entity_name(vn) == core1:
                                rel = dict(rel)
                                rel['entity1_name'] = vn
                                break
                    if e2 not in _valid_names:
                        core2 = _core_entity_name(e2)
                        for vn in _valid_names:
                            if _core_entity_name(vn) == core2:
                                rel = dict(rel)
                                rel['entity2_name'] = vn
                                break
                    _cleaned_relations.append(rel)
            extracted_relations = _cleaned_relations
            if len(extracted_relations) != _before_rel_cleanup:
                wprint(
                    f"【步骤2.8】关系清洗｜{_before_rel_cleanup}→{len(extracted_relations)}"
                )
            dbg(
                f"步骤2.8关系端点清洗: {_before_rel_cleanup} → {len(extracted_relations)}"
            )

        # 步骤3 后：记录统计信息
        if extracted_relations:
            _relation_endpoint_names: set[str] = set()
            for _rel in extracted_relations:
                _e1 = _rel.get('entity1_name', '').strip()
                _e2 = _rel.get('entity2_name', '').strip()
                if _e1:
                    _relation_endpoint_names.add(_e1)
                if _e2:
                    _relation_endpoint_names.add(_e2)
            _unconnected = sum(
                1 for e in extracted_entities
                if e.get('name', '').strip() not in _relation_endpoint_names
            )
            if _unconnected and (verbose or verbose_steps):
                wprint(
                    f"【步骤3】实体｜保留｜{_unconnected}个独立概念无关系端点，仍保留"
                )
            dbg(
                f"步骤3后实体保留: {len(extracted_entities)} 个 "
                f"（其中 {_unconnected} 个无关系端点，端点种类 {len(_relation_endpoint_names)}）"
            )

        # ========== 步骤3.5：关系反哺实体内容 ==========
        # 对内容过短的实体，利用其参与的关系描述补充上下文
        # 仅当实体增强（步骤5）关闭时才启用反哺；如果增强开启，增强步骤会提供更好的内容
        if extracted_relations and not self.entity_post_enhancement:
            _ENTITY_CONTENT_MIN_LENGTH = 30
            _entity_name_to_idx = {e['name']: i for i, e in enumerate(extracted_entities)}
            _rel_context_by_entity: Dict[str, List[str]] = {}
            for _rel in extracted_relations:
                _re1 = _rel.get('entity1_name', '').strip()
                _re2 = _rel.get('entity2_name', '').strip()
                _rc = _rel.get('content', '').strip()
                if _re1 and _rc:
                    _rel_context_by_entity.setdefault(_re1, []).append(
                        f"与{_re2}的关系: {_rc}"
                    )
                if _re2 and _rc:
                    _rel_context_by_entity.setdefault(_re2, []).append(
                        f"与{_re1}的关系: {_rc}"
                    )
            _enriched_count = 0
            for _name, _rel_contexts in _rel_context_by_entity.items():
                _idx = _entity_name_to_idx.get(_name)
                if _idx is None:
                    continue
                _ent = extracted_entities[_idx]
                _cur_content = _ent.get('content', '').strip()
                if len(_cur_content) >= _ENTITY_CONTENT_MIN_LENGTH:
                    continue
                # 附加关系上下文，不超过3条
                _snippets = _rel_contexts[:3]
                _suffix = "；".join(_snippets)
                if _cur_content:
                    _new_content = f"{_cur_content}（{_suffix}）"
                else:
                    _new_content = _suffix
                extracted_entities[_idx] = {
                    'name': _ent['name'],
                    'content': _new_content,
                }
                _enriched_count += 1
            if _enriched_count and verbose:
                wprint(
                    f"【步骤3.5】反哺｜完成｜{_enriched_count}个实体内容由关系上下文补充"
                )
            dbg(
                f"步骤3.5 关系反哺: {_enriched_count} 个实体内容被丰富 "
                f"(共 {len(_rel_context_by_entity)} 个实体参与关系)"
            )

        # ========== 步骤4已移除：联合抽取保证关系端点与实体列表一致，无需补全 ==========

        # ========== 步骤5：实体增强（仅对内容过短的实体） ==========
        if self.entity_post_enhancement:
            # 结构优化：只增强内容过短（<50字）的实体，跳过已有充分描述的实体
            _ENT_ENHANCE_MIN_LEN = 50
            _entities_to_enhance = [
                e for e in extracted_entities
                if len(e.get('content', '')) < _ENT_ENHANCE_MIN_LEN
            ]
            _entities_enough = [
                e for e in extracted_entities
                if len(e.get('content', '')) >= _ENT_ENHANCE_MIN_LEN
            ]

            if not _entities_to_enhance:
                if verbose:
                    wprint(f"【步骤5】增强｜跳过｜全部{_ENT_ENHANCE_MIN_LEN}字以上，无需增强")
                elif verbose_steps:
                    wprint("【步骤5】增强｜跳过｜")
            else:
                self.llm_client._priority_local.priority = LLM_PRIORITY_STEP5
                if verbose:
                    wprint(f"【步骤5】增强｜开始｜{len(_entities_to_enhance)}/{len(extracted_entities)}个需增强")
                elif verbose_steps:
                    wprint(f"【步骤5】增强｜开始｜{len(_entities_to_enhance)}个")

                self.llm_client._current_distill_step = "05_entity_enhancement"

                enhanced_entities = []
                if self.llm_threads > 1 and len(_entities_to_enhance) > 1:
                    _enhance_priority = self.llm_client._priority_local.priority
                    _enhance_distill_step = self.llm_client._current_distill_step
                    with ThreadPoolExecutor(max_workers=self.llm_threads, thread_name_prefix="tmg-llm") as executor:
                        def _enhance_task(entity):
                            # 将主线程的 distill step 和优先级传播到工作线程（threading.local）
                            self.llm_client._priority_local.priority = _enhance_priority
                            self.llm_client._current_distill_step = _enhance_distill_step
                            return self.llm_client.enhance_entity_content(
                                new_episode, input_text, entity
                            )
                        future_entity2 = {
                            executor.submit(_enhance_task, entity): entity
                            for entity in _entities_to_enhance
                        }

                        entity_results = {}
                        for future in as_completed(future_entity2):
                            entity = future_entity2[future]
                            try:
                                enhanced_content = future.result()
                                entity_results[entity['name']] = {
                                    'name': entity['name'],
                                    'content': enhanced_content
                                }
                            except Exception as e:
                                if verbose:
                                    wprint(f"【步骤5】警告｜增强｜{entity['name']}: {e}")
                                entity_results[entity['name']] = {
                                    'name': entity['name'],
                                    'content': entity['content']
                                }

                        for entity in _entities_to_enhance:
                            if entity['name'] in entity_results:
                                enhanced_entities.append(entity_results[entity['name']])
                            else:
                                enhanced_entities.append({
                                    'name': entity['name'],
                                    'content': entity['content']
                                })
                else:
                    enhanced_entities = []
                    for entity in _entities_to_enhance:
                        enhanced_content = self.llm_client.enhance_entity_content(
                            new_episode,
                            input_text,
                            entity
                        )
                        enhanced_entities.append({
                            'name': entity['name'],
                            'content': enhanced_content
                        })

                # 合并：未增强的实体 + 已增强的实体
                extracted_entities = _entities_enough + enhanced_entities

            self.llm_client._current_distill_step = None

            if verbose:
                wprint(f"【步骤5】增强｜完成｜共{len(extracted_entities)}个")
            elif verbose_steps:
                wprint("【步骤5】增强｜完成｜")
        else:
            if verbose:
                wprint("【步骤5】增强｜跳过｜已禁用")
            elif verbose_steps:
                wprint("【步骤5】增强｜跳过｜已禁用")
        _report_step(2, "实体增强", f"增强完成，共 {len(extracted_entities)} 个实体")

        return extracted_entities, extracted_relations

    # =========================================================================
    # 步骤6：实体对齐（写存储，必须串行跨窗口）
    # =========================================================================

    def _build_step7_relation_inputs_from_align_result(
        self, align_result: _AlignResult
    ) -> Tuple[List[Dict[str, str]], Dict[str, str], List[Dict], List[Dict]]:
        """从步骤6输出构造步骤7批处理输入；与 _align_relations 内逻辑一致，供预取与步骤7共用。"""
        entity_name_to_id = dict(align_result.entity_name_to_id)
        pending_relations_from_entities = align_result.pending_relations
        updated_pending_relations = align_result.unique_pending_relations

        # 某些并行实体对齐分支可能留下只存在于内存中的临时 family_id；
        # Step7 开始前按名称刷新一次，避免关系写入时再命中”family_id 不存在”。
        for name, eid in list(entity_name_to_id.items()):
            if eid:
                entity_name_to_id[name] = self.storage.resolve_family_id(eid)

        invalid_names = [
            name for name, eid in entity_name_to_id.items()
            if eid and self.storage.get_entity_by_family_id(eid) is None
        ]
        if invalid_names:
            refreshed_map = self.storage.get_family_ids_by_names(invalid_names)
            for name, refreshed_id in refreshed_map.items():
                if refreshed_id:
                    entity_name_to_id[name] = refreshed_id

        all_pending_relations = updated_pending_relations.copy()

        for rel_info in pending_relations_from_entities:
            entity1_name = rel_info.get("entity1_name", "")
            entity2_name = rel_info.get("entity2_name", "")
            content = rel_info.get("content", "")
            relation_type = rel_info.get("relation_type", "normal")

            entity1_id = entity_name_to_id.get(entity1_name)
            entity2_id = entity_name_to_id.get(entity2_name)

            if entity1_id and entity2_id:
                if entity1_id == entity2_id:
                    continue
                all_pending_relations.append({
                    "entity1_id": entity1_id,
                    "entity2_id": entity2_id,
                    "entity1_name": entity1_name,
                    "entity2_name": entity2_name,
                    "content": content,
                    "relation_type": relation_type
                })

        seen_relations = set()
        unique_pending_relations = []
        for rel in all_pending_relations:
            entity1_id = rel.get("entity1_id")
            entity2_id = rel.get("entity2_id")
            content = rel.get("content", "")
            if entity1_id and entity2_id:
                pair_key = tuple(sorted([entity1_id, entity2_id]))
                content_hash = hash(content.strip().lower())
                relation_key = (pair_key, content_hash)
                if relation_key not in seen_relations:
                    seen_relations.add(relation_key)
                    unique_pending_relations.append(rel)

        relation_inputs = [
            {
                "entity1_name": rel_info.get("entity1_name", ""),
                "entity2_name": rel_info.get("entity2_name", ""),
                "content": rel_info.get("content", ""),
            }
            for rel_info in unique_pending_relations
        ]

        return relation_inputs, entity_name_to_id, unique_pending_relations, all_pending_relations

    def _align_entities(self, extracted_entities: List[Dict], extracted_relations: List[Dict],
                        new_episode: Episode, input_text: str,
                        document_name: str, verbose: bool = True,
                        verbose_steps: bool = True,
                        event_time: Optional[datetime] = None,
                        progress_callback=None,
                        progress_range: tuple = (0.5, 0.75),
                        window_index: int = 0,
                        total_windows: int = 1,
                        entity_embedding_prefetch: Optional[Future] = None) -> _AlignResult:
        """步骤6：实体对齐（搜索、合并、写入存储）。必须串行跨窗口。

        Returns:
            _AlignResult 包含 entity_name_to_id、pending_relations 等，供步骤7使用。
        """

        p_lo, p_hi = progress_range
        _win_label = f"窗口 {window_index + 1}/{total_windows}"

        self.llm_client._priority_local.priority = LLM_PRIORITY_STEP6
        if verbose:
            wprint("【步骤6】实体｜开始｜对齐写入")
        elif verbose_steps:
            wprint("【步骤6】实体｜开始｜")

        self.llm_client._current_distill_step = "06_entity_alignment"

        # 记录原始实体名称列表（用于后续建立映射）
        original_entity_names = [e['name'] for e in extracted_entities]

        # 用于存储待处理的关系（使用实体名称）
        all_pending_relations_by_name = []
        if extracted_relations:
            for rel in extracted_relations:
                entity1_name = rel.get('entity1_name') or rel.get('from_entity_name', '').strip()
                entity2_name = rel.get('entity2_name') or rel.get('to_entity_name', '').strip()
                content = rel.get('content', '').strip()
                if entity1_name and entity2_name:
                    all_pending_relations_by_name.append({
                        "entity1_name": entity1_name,
                        "entity2_name": entity2_name,
                        "content": content,
                        "relation_type": "normal"
                    })

        entity_name_to_id_from_entities = {}
        _entity_total = len(extracted_entities)
        _entity_done = 0
        _step_size = p_hi - p_lo

        def on_entity_processed_callback(entity, current_entity_name_to_id, current_pending_relations):
            nonlocal all_pending_relations_by_name, entity_name_to_id_from_entities, _entity_done
            _entity_done += 1
            entity_name_to_id_from_entities.update(current_entity_name_to_id)
            all_pending_relations_by_name.extend(current_pending_relations)
            if progress_callback:
                frac = _entity_done / max(1, _entity_total)
                progress_callback(p_lo + _step_size * frac,
                    f"{_win_label} · 步骤6/7: 实体对齐 ({_entity_done}/{_entity_total})",
                    f"实体对齐 {_entity_done}/{_entity_total}")

        processed_entities, pending_relations_from_entities, entity_name_to_id_from_entities_final = self.entity_processor.process_entities(
            extracted_entities,
            new_episode.absolute_id,
            self.similarity_threshold,
            episode=new_episode,
            source_document=document_name,
            context_text=input_text,
            extracted_relations=extracted_relations,
            jaccard_search_threshold=self.jaccard_search_threshold,
            embedding_name_search_threshold=self.embedding_name_search_threshold,
            embedding_full_search_threshold=self.embedding_full_search_threshold,
            on_entity_processed=on_entity_processed_callback,
            base_time=new_episode.event_time,
            max_workers=self.llm_threads,
            verbose=verbose,
            entity_embedding_prefetch=entity_embedding_prefetch,
        )

        entity_name_to_id_from_entities.update(entity_name_to_id_from_entities_final)
        pending_relations_from_entities = all_pending_relations_by_name

        # 按family_id去重，只保留最新版本
        unique_entities_dict = {}
        for entity in processed_entities:
            if entity.family_id not in unique_entities_dict:
                unique_entities_dict[entity.family_id] = entity
            else:
                if entity.processed_time > unique_entities_dict[entity.family_id].processed_time:
                    unique_entities_dict[entity.family_id] = entity

        unique_entities = list(unique_entities_dict.values())

        # 构建完整的实体名称到family_id的映射
        entity_name_to_ids = {}
        for entity in unique_entities:
            if entity.name not in entity_name_to_ids:
                entity_name_to_ids[entity.name] = []
            if entity.family_id not in entity_name_to_ids[entity.name]:
                entity_name_to_ids[entity.name].append(entity.family_id)

        for name, family_id in entity_name_to_id_from_entities.items():
            if name not in entity_name_to_ids:
                entity_name_to_ids[name] = []
            if family_id not in entity_name_to_ids[name]:
                entity_name_to_ids[name].append(family_id)

        for i, entity in enumerate(processed_entities):
            if i < len(original_entity_names):
                original_name = original_entity_names[i]
                if original_name not in entity_name_to_ids:
                    entity_name_to_ids[original_name] = []
                if entity.family_id not in entity_name_to_ids[original_name]:
                    entity_name_to_ids[original_name].append(entity.family_id)

        # 检测和处理同名实体冲突
        duplicate_names = {name: ids for name, ids in entity_name_to_ids.items() if len(ids) > 1}

        if duplicate_names:
            if verbose:
                wprint(f"【步骤6】警告｜同名｜{len(duplicate_names)}处")
                for name, ids in duplicate_names.items():
                    wprint(
                        f"【步骤6】冲突｜详情｜{name} {len(ids)}id {ids[:3]}{'...' if len(ids) > 3 else ''}"
                    )

            entity_name_to_id = {}
            for name, ids in entity_name_to_ids.items():
                if len(ids) > 1:
                    # 先检查同名实体的内容是否语义相关
                    versions_map = {}
                    contents_map = {}
                    for fid in ids:
                        vs = self.storage.get_entity_versions(fid)
                        versions_map[fid] = len(vs)
                        contents_map[fid] = (vs[0].content or "")[:300] if vs else ""

                    # 计算内容重叠度，避免不同含义的同名实体被错误合并
                    contents = [c for c in contents_map.values() if c.strip()]
                    should_merge = True
                    if len(contents) >= 2:
                        # 简单的词级 Jaccard 相似度
                        all_words = [set(c.split()) for c in contents]
                        if all_words:
                            intersection = all_words[0]
                            union = all_words[0]
                            for ws in all_words[1:]:
                                intersection = intersection & ws
                                union = union | ws
                            jaccard = len(intersection) / max(len(union), 1)
                            if jaccard < 0.2:
                                # 内容几乎无重叠，不合并
                                should_merge = False
                                if verbose:
                                    wprint(f"【步骤6】冲突｜跳过｜同名实体 '{name}' 内容差异过大 (Jaccard={jaccard:.2f})，不合并")

                    if should_merge:
                        primary_id = max(ids, key=lambda fid: versions_map.get(fid, 0))
                        entity_name_to_id[name] = primary_id
                        for fid in ids:
                            if fid and fid != primary_id:
                                self.storage.register_entity_redirect(fid, primary_id)
                        if verbose:
                            wprint(
                                f"【步骤6】冲突｜主实体｜{name}->{primary_id} v{versions_map.get(primary_id, 0)}"
                            )
                    else:
                        # 不合并：同名但不同实体，不添加到映射
                        # 让关系端点保留原始名称，避免指向错误实体
                        if verbose:
                            wprint(f"【步骤6】冲突｜跳过｜同名实体 '{name}' 存在多个不同含义，不自动映射")
                        continue
                else:
                    entity_name_to_id[name] = ids[0]
        else:
            entity_name_to_id = {name: ids[0] for name, ids in entity_name_to_ids.items()}

        merged_mappings = []
        for i, entity in enumerate(processed_entities):
            if i < len(original_entity_names):
                original_name = original_entity_names[i]
                if original_name != entity.name:
                    merged_mappings.append((original_name, entity.name, entity.family_id))

        if verbose:
            if len(unique_entities) == 0:
                wprint(
                    f"【步骤6】小结｜实体｜无新·抽{len(original_entity_names)}个已存在"
                )
            else:
                wprint(
                    f"【步骤6】小结｜实体｜唯一{len(unique_entities)}·原{len(original_entity_names)}"
                )
            if merged_mappings:
                wprint(f"【步骤6】映射｜合并｜{len(merged_mappings)}个")

        # 步骤6.3：构建完整的实体名称→ID映射表，防止关系丢失
        # 收集关系中引用的所有实体名称
        _rel_entity_names = set()
        for rel_info in pending_relations_from_entities:
            n1 = rel_info.get("entity1_name", "")
            n2 = rel_info.get("entity2_name", "")
            if n1:
                _rel_entity_names.add(n1)
            if n2:
                _rel_entity_names.add(n2)
        # 补全：关系中引用但映射表里缺失的名称
        _missing_names = [n for n in _rel_entity_names if n not in entity_name_to_id]
        _db_matched = 0
        _fuzzy_matched = 0

        # 第一轮：精确匹配数据库
        if _missing_names:
            _db_map = self.storage.get_family_ids_by_names(_missing_names)
            for name, eid in _db_map.items():
                if name not in entity_name_to_id:
                    entity_name_to_id[name] = eid
                    _db_matched += 1

        # 第二轮：核心名称模糊匹配（去掉括号后比较）
        # 处理 LLM 在关系中使用带括号别名但实体列表中用简名的情况
        # 例如：关系端点"Docker（开源容器引擎）"匹配到实体"Docker"
        _still_missing = [n for n in _rel_entity_names if n not in entity_name_to_id]
        if _still_missing:
            # 构建核心名称→family_id 反查表
            _core_name_map: Dict[str, str] = {}
            for name, eid in entity_name_to_id.items():
                core = _core_entity_name(name)
                if core and core not in _core_name_map:
                    _core_name_map[core] = eid

            for missing_name in _still_missing:
                core_missing = _core_entity_name(missing_name)
                if not core_missing:
                    continue
                # 先在已有映射中找核心名称匹配
                if core_missing in _core_name_map:
                    entity_name_to_id[missing_name] = _core_name_map[core_missing]
                    _fuzzy_matched += 1
                    continue
                # 再在数据库中按核心名称搜索
                _db_core_map = self.storage.get_family_ids_by_names([core_missing])
                if _db_core_map:
                    for _, eid in _db_core_map.items():
                        entity_name_to_id[missing_name] = eid
                        _core_name_map[core_missing] = eid
                        _fuzzy_matched += 1
                        break

        # 名称→ID转换
        updated_pending_relations = []
        _skipped_relations = []
        _self_relations = 0
        for rel_info in pending_relations_from_entities:
            entity1_name = rel_info.get("entity1_name", "")
            entity2_name = rel_info.get("entity2_name", "")
            content = rel_info.get("content", "")
            relation_type = rel_info.get("relation_type", "normal")

            entity1_id = entity_name_to_id.get(entity1_name)
            entity2_id = entity_name_to_id.get(entity2_name)

            if entity1_id and entity2_id:
                if entity1_id == entity2_id:
                    _self_relations += 1
                    continue
                updated_pending_relations.append({
                    "entity1_id": entity1_id,
                    "entity2_id": entity2_id,
                    "entity1_name": entity1_name,
                    "entity2_name": entity2_name,
                    "content": content,
                    "relation_type": relation_type
                })
            else:
                _reason = []
                if not entity1_id:
                    _reason.append(f"entity1='{entity1_name}'")
                if not entity2_id:
                    _reason.append(f"entity2='{entity2_name}'")
                _skipped_relations.append(f"  {entity1_name} <-> {entity2_name} (无法解析: {', '.join(_reason)})")

        if _skipped_relations or _self_relations > 0:
            _parts = [f"成功解析 {len(updated_pending_relations)} 个"]
            if _db_matched > 0:
                _parts.append(f"数据库补全 {_db_matched} 个")
            if _fuzzy_matched > 0:
                _parts.append(f"模糊匹配 {_fuzzy_matched} 个")
            if _self_relations > 0:
                _parts.append(f"自关系 {_self_relations} 个")
            if _skipped_relations:
                _parts.append(f"无法解析 {len(_skipped_relations)} 个")
            if verbose:
                wprint(
                    f"【步骤6】关系｜待处理｜{len(pending_relations_from_entities)}→{', '.join(_parts)}"
                )
                if _skipped_relations:
                    wprint(
                        f"【步骤6】映射｜表｜{len(entity_name_to_id)}名 "
                        f"{', '.join(list(entity_name_to_id.keys())[:15])}{'...' if len(entity_name_to_id) > 15 else ''}"
                    )
                    for _sr in _skipped_relations[:10]:
                        wprint(f"【步骤6】关系｜跳过｜{_sr}")
                    if len(_skipped_relations) > 10:
                        wprint(f"【步骤6】关系｜跳过｜余{len(_skipped_relations) - 10}条")
        else:
            if verbose:
                wprint(
                    f"【步骤6】关系｜待处理｜{len(pending_relations_from_entities)}→全解析"
                    + (f"·库补{_db_matched}" if _db_matched > 0 else "")
                )

        if verbose_steps and not verbose:
            wprint("【步骤6】实体｜完成｜映射")

        dbg_section("步骤6.3: 实体名称→family_id映射")
        dbg(f"entity_name_to_id 映射 ({len(entity_name_to_id)} 个):")
        for _mn, _mid in entity_name_to_id.items():
            dbg(f"  '{_mn}' -> {_mid}")
        dbg(f"待处理关系 {len(pending_relations_from_entities)} 个 → 成功 {len(updated_pending_relations)}, 自关系 {_self_relations}, 跳过 {len(_skipped_relations)}")
        for _sr in _skipped_relations:
            dbg(f"  跳过: {_sr}")

        self.llm_client._current_distill_step = None

        if progress_callback:
            progress_callback(p_hi,
                f"{_win_label} · 步骤6/7: 实体对齐",
                f"实体对齐完成，共 {len(unique_entities)} 个实体")

        # Phase C: 记录 Episode → Entity MENTIONS
        if unique_entities and hasattr(self.storage, 'save_episode_mentions'):
            try:
                abs_ids = [e.absolute_id for e in unique_entities]
                self.storage.save_episode_mentions(new_episode.absolute_id, abs_ids)
            except Exception as e:
                if verbose:
                    wprint(f"【步骤6】MENTIONS｜失败｜{e}")

        return _AlignResult(
            entity_name_to_id=entity_name_to_id,
            pending_relations=pending_relations_from_entities,
            unique_entities=unique_entities,
            unique_pending_relations=updated_pending_relations,
        )

    # =========================================================================
    # 步骤7：关系对齐（写存储，串行跨窗口）
    # =========================================================================

    def _align_relations(self, align_result: _AlignResult,
                         new_episode: Episode, input_text: str,
                         document_name: str, verbose: bool = True,
                         verbose_steps: bool = True,
                         event_time: Optional[datetime] = None,
                         progress_callback=None,
                         progress_range: tuple = (0.75, 1.0),
                         window_index: int = 0,
                         total_windows: int = 1,
                         prepared_relations_by_pair: Optional[Dict[Tuple[str, str], List[Dict[str, str]]]] = None,
                         step7_inputs_cache: Optional[Tuple[List[Dict[str, str]], Dict[str, str], List[Dict], List[Dict]]] = None,
                         ) -> List:
        """步骤7：关系对齐（搜索、合并、写入存储）。串行跨窗口。

        Args:
            align_result: 步骤6的输出，包含 entity_name_to_id 和 pending_relations。
            prepared_relations_by_pair: 可选，跨窗预取的按实体对分组结果（须在上一窗 step7 完成后读库）。
            step7_inputs_cache: 可选，与 _build_step7_relation_inputs_from_align_result 返回值一致，避免重复计算。
        """

        p_lo, p_hi = progress_range
        _win_label = f"窗口 {window_index + 1}/{total_windows}"
        _step_size = p_hi - p_lo

        self.llm_client._priority_local.priority = LLM_PRIORITY_STEP7
        if verbose:
            wprint("【步骤7】关系｜开始｜对齐写入")
        elif verbose_steps:
            wprint("【步骤7】关系｜开始｜")

        self.llm_client._current_distill_step = "07_relation_alignment"

        unique_entities = align_result.unique_entities

        if step7_inputs_cache is not None:
            relation_inputs, entity_name_to_id, unique_pending_relations, all_pending_relations = step7_inputs_cache
        else:
            relation_inputs, entity_name_to_id, unique_pending_relations, all_pending_relations = (
                self._build_step7_relation_inputs_from_align_result(align_result)
            )

        if verbose:
            duplicate_count = len(all_pending_relations) - len(unique_pending_relations)
            if duplicate_count > 0:
                wprint(
                    f"【步骤7】关系｜待处理｜{len(all_pending_relations)}→去重{len(unique_pending_relations)}"
                )
            else:
                wprint(f"【步骤7】关系｜待处理｜{len(unique_pending_relations)}个")

        _upr_count = len(unique_pending_relations)
        if _upr_count == 0:
            if verbose:
                wprint("【步骤7】关系｜跳过｜无待处理")
        else:
            if verbose:
                wprint(
                    f"【步骤7】关系｜待处理｜去重{_upr_count}·原{len(all_pending_relations)}"
                )
        dbg(f"步骤7: 去重后待处理关系 {len(unique_pending_relations)} 个 (去重前 {len(all_pending_relations)} 个)")
        for _upr in unique_pending_relations:
            dbg(f"  待处理: '{_upr.get('entity1_name', '')}' <-> '{_upr.get('entity2_name', '')}' (e1_id={_upr.get('entity1_id', '?')}, e2_id={_upr.get('entity2_id', '?')})  content='{_upr.get('content', '')[:100]}'")

        _rel_done = [0]

        def _on_relation_pair_done(done, total):
            _rel_done[0] = done
            if progress_callback:
                frac = done / max(1, total)
                progress_callback(p_lo + _step_size * frac,
                    f"{_win_label} · 步骤7/7: 关系对齐 ({done}/{total})",
                    f"关系对齐 {done}/{total}")

        all_processed_relations = self.relation_processor.process_relations_batch(
            relation_inputs,
            entity_name_to_id,
            new_episode.absolute_id,
            source_document=document_name,
            base_time=new_episode.event_time,
            max_workers=self.llm_threads,
            on_relation_done=_on_relation_pair_done,
            # detail 模式常开 verbose、关 verbose_steps：避免逐条 [关系操作] 刷屏
            verbose_relation=bool(verbose and verbose_steps),
            prepared_relations_by_pair=prepared_relations_by_pair,
        )

        if verbose:
            if len(all_processed_relations) == 0:
                wprint("【步骤7】关系｜小结｜无新")
            else:
                wprint(f"【步骤7】关系｜小结｜{len(all_processed_relations)}个")
        elif verbose_steps:
            wprint("【步骤7】关系｜完成｜")

        if verbose:
            wprint("【窗口】流水｜结束｜")
        _final_ents = len(unique_entities)
        _final_rels = len(all_processed_relations)
        if verbose:
            if _final_ents == 0 and _final_rels == 0:
                wprint("【窗口】汇总｜空｜无新实体关系")
            else:
                wprint(
                    f"【窗口】汇总｜得｜实体{_final_ents} 关系{_final_rels}·待{len(unique_pending_relations)}"
                )
        elif verbose_steps:
            wprint(f"【窗口】汇总｜得｜实体{_final_ents} 关系{_final_rels}")
        dbg(f"窗口处理完成: {len(unique_entities)} 个实体, {len(all_processed_relations)} 个关系 (从 {len(unique_pending_relations)} 个待处理)")

        if progress_callback:
            progress_callback(p_hi,
                f"{_win_label} · 步骤7/7: 窗口完成",
                f"{len(unique_entities)} 个实体, {len(all_processed_relations)} 个关系")

        self.llm_client._current_distill_step = None
        self.llm_client._distill_task_id = None

        return all_processed_relations

    # =========================================================================
    # 兼容入口：串行执行步骤2-7（_process_window 旧路径使用）
    # =========================================================================

    def _verify_window_results(
        self,
        entities: list,
        relations: list,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """步骤8: 纯代码校验（零LLM调用）。返回校验报告。"""
        import re as _re

        report = {
            "entity_count": len(entities),
            "relation_count": len(relations),
            "issues": [],
            "warnings": [],
        }

        # Check 1: 孤立实体（无关系连接）
        entity_ids_in_relations = set()
        for rel in relations:
            entity_ids_in_relations.add(getattr(rel, 'entity1_id', None))
            entity_ids_in_relations.add(getattr(rel, 'entity2_id', None))
        entity_ids_in_relations.discard(None)
        isolated = [e for e in entities if getattr(e, 'family_id', None) not in entity_ids_in_relations]
        if isolated:
            report["warnings"].append({
                "type": "isolated_entities",
                "count": len(isolated),
                "names": [getattr(e, 'name', '?') for e in isolated[:5]],
            })

        # Check 2: 实体内容质量
        for e in entities:
            content = getattr(e, 'content', '') or ''
            name = getattr(e, 'name', '?')
            fid = getattr(e, 'family_id', '?')
            if len(content) < 10:
                report["issues"].append({
                    "type": "entity_content_too_short",
                    "entity_name": name,
                    "family_id": fid,
                })
            content_lower = content.lower()
            for pattern in ["处理进度", "步骤", "缓存", "抽取", "token", "api"]:
                if pattern in content_lower:
                    report["issues"].append({
                        "type": "entity_content_system_leak",
                        "entity_name": name,
                        "pattern": pattern,
                    })
                    break

        # Check 3: 关系内容质量
        for rel in relations:
            content = getattr(rel, 'content', '') or ''
            rid = getattr(rel, 'family_id', '?')
            if len(content) < 8:
                report["issues"].append({
                    "type": "relation_content_too_short",
                    "relation_id": rid,
                })

        # Check 4: 核心名称重复（去括号后同名）
        core_name_map: Dict[str, list] = {}
        for e in entities:
            name = getattr(e, 'name', '')
            fid = getattr(e, 'family_id', '')
            core = _re.sub(r'[（(][^）)]+[）)]', '', name).strip()
            if core not in core_name_map:
                core_name_map[core] = []
            core_name_map[core].append(fid)
        for core, fids in core_name_map.items():
            if len(set(fids)) > 1:
                report["warnings"].append({
                    "type": "duplicate_core_names",
                    "core_name": core,
                    "family_ids": list(set(fids)),
                })

        # Check 5: 实体名称有效性
        for e in entities:
            name = getattr(e, 'name', '')
            fid = getattr(e, 'family_id', '')
            if name and not _is_valid_entity_name(name):
                report["issues"].append({
                    "type": "invalid_entity_name",
                    "entity_name": name,
                    "family_id": fid,
                })

        if verbose and (report["issues"] or report["warnings"]):
            wprint(f"【步骤8】校验｜问题{len(report['issues'])} 警告{len(report['warnings'])}")
            for issue in report["issues"][:5]:
                wprint(f"  ⚠ 问题: {issue['type']} — {issue.get('entity_name', '') or issue.get('relation_id', '')}")
            for warn in report["warnings"][:5]:
                wprint(f"  ⚡ 警告: {warn['type']} — {warn.get('names', warn.get('core_name', ''))}")

        return report

    def _process_extraction(self, new_episode: Episode, input_text: str,
                            document_name: str, verbose: bool = True,
                            verbose_steps: bool = True,
                            event_time: Optional[datetime] = None,
                            progress_callback=None,
                            progress_range: tuple = (0.1, 1.0),
                            window_index: int = 0,
                            total_windows: int = 1):
        """兼容入口：串行执行步骤2-7（_process_window 等旧路径使用）。"""

        # 步骤2-5 占 progress_range 的 5/7，步骤6 占 1/7，步骤7 占 1/7
        total_size = progress_range[1] - progress_range[0]
        p1_end = progress_range[0] + total_size * 5 / 7
        p2_end = progress_range[0] + total_size * 6 / 7

        extracted_entities, extracted_relations = self._extract_only(
            new_episode, input_text, document_name,
            verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
            progress_callback=progress_callback,
            progress_range=(progress_range[0], p1_end),
            window_index=window_index, total_windows=total_windows,
        )

        align_result = self._align_entities(
            extracted_entities, extracted_relations,
            new_episode, input_text, document_name,
            verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
            progress_callback=progress_callback,
            progress_range=(p1_end, p2_end),
            window_index=window_index, total_windows=total_windows,
        )

        processed_relations = self._align_relations(
            align_result,
            new_episode, input_text, document_name,
            verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
            progress_callback=progress_callback,
            progress_range=(p2_end, progress_range[1]),
            window_index=window_index, total_windows=total_windows,
        )

        # 步骤8: 纯代码校验
        self._verify_window_results(
            align_result.unique_entities,
            processed_relations or [],
            verbose=verbose,
        )

    def _process_window(self, input_text: str, document_name: str,
                       is_new_document: bool, text_start_pos: int = 0,
                       text_end_pos: int = 0, total_text_length: int = 0,
                       verbose: bool = True, verbose_steps: bool = True,
                       document_path: str = "",
                       event_time: Optional[datetime] = None,
                       window_index: int = 0, total_windows: int = 1):
        """兼容入口：串行执行 cache 更新 + 抽取处理（process_documents 等旧路径使用）。"""
        if verbose:
            wprint(f"\n{'='*60}")
            wprint(f"处理窗口 (文档: {document_name}, 位置: {text_start_pos}-{text_end_pos}/{total_text_length})")
            wprint(f"输入文本长度: {len(input_text)} 字符")
            wprint(f"{'='*60}\n")
        elif verbose_steps:
            wprint(f"窗口开始 · {document_name}  [{text_start_pos}-{text_end_pos}/{total_text_length}]")

        with self._cache_lock:
            new_mc = self._update_cache(
                input_text, document_name,
                text_start_pos=text_start_pos, text_end_pos=text_end_pos,
                total_text_length=total_text_length, verbose=verbose,
                verbose_steps=verbose_steps,
                document_path=document_path, event_time=event_time,
                window_index=window_index, total_windows=total_windows,
            )
        self._process_extraction(new_mc, input_text, document_name,
                                 verbose=verbose, verbose_steps=verbose_steps, event_time=event_time,
                                 window_index=window_index, total_windows=total_windows)
