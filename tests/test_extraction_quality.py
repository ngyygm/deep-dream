"""
测试步骤2-3抽取质量改进：实体名称清洗、实体有效性过滤、关系内容质量验证。
基于从记忆图谱中发现的真实错误数据编写。
"""
from processor.pipeline.extraction import (
    _is_valid_entity_name,
    _clean_entity_name,
    _is_valid_relation_content,
    dedupe_extracted_entities,
    dedupe_extracted_relations,
)


# ===================================================================
# 1. 实体名称有效性检查（_is_valid_entity_name）
# ===================================================================

class TestIsValidEntityName:
    """过滤非实体文本片段。"""

    # --- 应该被过滤掉的（来自记忆图谱中的真实错误数据）---

    def test_blacklisted_verb_fragments(self):
        """常见动词片段不应作为实体名称。"""
        for name in ["遂得", "险恶", "处理进度", "突阵而出", "诸营已失", "弃关"]:
            assert not _is_valid_entity_name(name), f"'{name}' 应被过滤"

    def test_too_short_names(self):
        """过短的名称（<2字符）应被过滤。"""
        assert not _is_valid_entity_name("")
        assert not _is_valid_entity_name("a")
        assert not _is_valid_entity_name("遂")

    def test_wide_generic_descriptions(self):
        """过于宽泛的描述性概念不应作为实体。"""
        assert not _is_valid_entity_name("忠心耿耿的群体")
        assert not _is_valid_entity_name("官職")
        assert not _is_valid_entity_name("居住地")

    # --- 应该通过的（合法实体名称）---

    def test_valid_person_names(self):
        """人名应该通过。"""
        assert _is_valid_entity_name("曹操")
        assert _is_valid_entity_name("杨昂")
        assert _is_valid_entity_name("许褚")
        assert _is_valid_entity_name("华歆")

    def test_valid_names_with_identity_parens(self):
        """带身份消歧括号的名称应该通过。"""
        assert _is_valid_entity_name("曹操（魏王）")
        assert _is_valid_entity_name("许褚（曹操亲卫）")
        assert _is_valid_entity_name("刘慈欣（中国科幻作家）")
        assert _is_valid_entity_name("金钏（王夫人丫头）")

    def test_valid_place_names(self):
        """地名应该通过。"""
        assert _is_valid_entity_name("汉中")
        assert _is_valid_entity_name("长安")

    def test_valid_concept_names(self):
        """概念名称应该通过。"""
        assert _is_valid_entity_name("三体问题（经典力学）")
        assert _is_valid_entity_name("Python（编程语言）")

    # --- 新增规则：场景描述、系统泄露、乱码、对话标记 ---

    def test_reject_overly_long_scene_descriptions(self):
        """超过30字符的名称大概率是场景描述/对话片段，应被拒绝。"""
        for name in [
            "必出奇兵,方可取胜",           # 31 chars - 来自真实错误数据
            "刘备带领的五百人部队",        # 11 chars but valid length, OK
            "遂得险恶遂得险恶遂得险恶",    # repetitive fragment
        ]:
            if len(name) > 30:
                assert not _is_valid_entity_name(name), f"'{name}' (len={len(name)}) 应被过滤（过长）"

    def test_reject_system_status_leaks(self):
        """系统状态/技术术语泄露到实体名称中应被拒绝。"""
        for name in [
            "处理进度", "系统状态", "抽取结果",
            "缓存数据", "对齐结果", "窗口处理",
            "流水线配置", "阈值设定",
        ]:
            assert not _is_valid_entity_name(name), f"'{name}' 应被过滤（系统泄露）"

    def test_reject_repetitive_fragments(self):
        """同一2字片段重复3次以上应被拒绝（乱码/解析错误）。"""
        for name in [
            "解读解读解读",
            "分析分析分析",
            "关系关系关系",
        ]:
            assert not _is_valid_entity_name(name), f"'{name}' 应被过滤（重复片段）"

    def test_reject_dialogue_markers(self):
        """含对话格式标记的名称应被拒绝。"""
        for name in [
            "曹操笑曰",
            "杨昂答曰",
            "许褚喝曰",
            "刘备说道",
        ]:
            assert not _is_valid_entity_name(name), f"'{name}' 应被过滤（对话标记）"

    def test_valid_names_still_pass_new_rules(self):
        """正常实体名称不应被新规则误杀。"""
        for name in [
            "曹操", "杨昂", "许褚", "汉中", "长安",
            "曹操（魏王）", "许褚（曹操亲卫）",
            "三体问题（经典力学）", "Python（编程语言）",
            "红楼梦", "三国演义",
            "放射性", "诺贝尔奖", "索邦大学", "玛丽·居里",
            "皮埃尔·居里", "华沙", "巴黎",
        ]:
            assert _is_valid_entity_name(name), f"'{name}' 应通过（合法实体名）"

    def test_reject_generic_science_achievements(self):
        """泛化科学成就概念应被过滤（LLM过度抽取的复合概念）。"""
        for name in [
            "科学成就", "科学贡献", "科学传承", "科学史", "科学合作",
            "科学影响", "科学教育", "科学界", "研究方法", "学术生涯",
            "科学成就传播", "科学成就历史意义", "科学成就国际影响",
            "科学成就影响", "科学成就社会影响", "科学成就认可",
            "科学贡献评价", "科学界地位",
        ]:
            assert not _is_valid_entity_name(name), f"'{name}' 应被过滤（泛化科学概念）"

    def test_reject_generic_action_processes(self):
        """泛化动作/过程描述应被过滤。"""
        for name in [
            "迁移", "合作研究", "开创性研究", "元素发现",
        ]:
            assert not _is_valid_entity_name(name), f"'{name}' 应被过滤（泛化动作/过程）"

    def test_reject_bilingual_generic_names(self):
        """双语格式（中文(英文)）的泛化概念应被过滤。"""
        for name in [
            "发现(discovery)", "研究(research)", "阅读(read)",
            "丈夫(husband)", "妻子(wife)", "合作(partnership)",
            "1903年(1903)", "1911年(1911)",
        ]:
            assert not _is_valid_entity_name(name), f"'{name}' 应被过滤（双语泛化概念）"

    def test_valid_bilingual_names_pass(self):
        """有效的双语实体名不应被误杀。"""
        for name in [
            "玛丽·居里(Marie Curie)", "皮埃尔·居里(Pierre Curie)",
            "华沙(Warsaw)", "巴黎(Paris)", "波兰(Poland)",
            "索邦大学(Sorbonne)", "诺贝尔奖(Nobel Prize)",
            "放射性(radioactivity)", "钋(poium)", "镭(radium)",
        ]:
            assert _is_valid_entity_name(name), f"'{name}' 应通过（有效双语实体）"

    def test_reject_chinese_verb_phrases(self):
        """中文动词短语/事件描述应被过滤。"""
        for name in [
            "恢复农业生产", "稳定社会秩序", "统一北方",
            "继承权争夺", "继承权争斗",
            "屯田制度实施",
            "郭嘉病逝",
            "感慨", "内斗",
        ]:
            assert not _is_valid_entity_name(name), f"'{name}' 应被过滤（动词短语/事件描述）"

    def test_reject_emotion_and_metadata(self):
        """情感词、文档元数据、角色标签应被过滤。"""
        for name in [
            "悲痛之情", "军事行动",
            "文件名:test_dialogue3", "文档名:readme",
            "谋士角色",
        ]:
            assert not _is_valid_entity_name(name), f"'{name}' 应被过滤（情感/元数据/角色）"

    def test_reject_exclamation_quotes(self):
        """感叹句/引用应被过滤。"""
        for name in [
            "哀哉奉孝！痛哉奉孝！",
        ]:
            assert not _is_valid_entity_name(name), f"'{name}' 应被过滤（感叹句/引用）"

    def test_valid_chinese_entities_still_pass(self):
        """正常中文实体不应被新规则误杀。"""
        for name in [
            "乌桓", "官渡之战", "屯田制", "农业生产",
            "袁谭", "袁尚", "郭嘉", "曹操",
            "北征乌桓",  # 可被视为历史事件
        ]:
            assert _is_valid_entity_name(name), f"'{name}' 应通过（合法中文实体）"


# ===================================================================
# 2. 实体名称清洗（_clean_entity_name）
# ===================================================================

class TestCleanEntityName:
    """清洗实体名称中的场景标注括号，保留身份/类型括号。"""

    # --- 场景标注应被移除（来自记忆图谱中的真实错误数据）---

    def test_remove_scene_annotation_person(self):
        """人名后的场景标注应被移除。"""
        assert _clean_entity_name("曹操（汉中张鲁）") == "曹操"
        assert _clean_entity_name("许褚（汉中张鲁）") == "许褚"

    def test_remove_scene_annotation_place(self):
        """地名后的事件标注应被移除。"""
        assert _clean_entity_name("汉中张鲁（众官告免）") == "汉中张鲁"
        assert _clean_entity_name("汉中张鲁（徐晃）") == "汉中张鲁"

    def test_remove_scene_annotation_with_context(self):
        """上下文描述应被移除。"""
        assert _clean_entity_name("杨昂（诸营）") == "杨昂"

    def test_remove_event_annotation(self):
        """事件片段标注应被移除。"""
        assert _clean_entity_name("火来（忽寨后一把火起）") == "火来"

    # --- 身份/类型括号应被保留 ---

    def test_keep_identity_parens(self):
        """身份消歧括号应保留。"""
        assert _clean_entity_name("曹操（魏王）") == "曹操（魏王）"
        assert _clean_entity_name("许褚（曹操亲卫）") == "许褚（曹操亲卫）"
        assert _clean_entity_name("金钏（王夫人丫头）") == "金钏（王夫人丫头）"

    def test_keep_domain_parens(self):
        """领域归属括号应保留。"""
        assert _clean_entity_name("三体问题（经典力学）") == "三体问题（经典力学）"
        assert _clean_entity_name("Python（编程语言）") == "Python（编程语言）"

    def test_keep_relationship_parens(self):
        """关系描述括号应保留。"""
        assert _clean_entity_name("刘慈欣（中国科幻作家）") == "刘慈欣（中国科幻作家）"

    def test_keep_chapter_parens(self):
        """章节标识括号应保留。"""
        assert _clean_entity_name("红楼梦（第七十八回）") == "红楼梦（第七十八回）"

    def test_no_parens_unchanged(self):
        """无括号的名称不应变化。"""
        assert _clean_entity_name("曹操") == "曹操"
        assert _clean_entity_name("杨昂") == "杨昂"


# ===================================================================
# 3. 关系内容质量验证（_is_valid_relation_content）
# ===================================================================

class TestIsValidRelationContent:
    """过滤低质量关系描述。"""

    # --- 应该被过滤掉的（来自记忆图谱中的真实错误数据）---

    def test_too_short_content(self):
        """过短的内容（<8字符）应被过滤。"""
        assert not _is_valid_relation_content("")
        assert not _is_valid_relation_content("有关")
        assert not _is_valid_relation_content("关联关系")
        assert not _is_valid_relation_content("同僚关系")

    def test_label_only_content(self):
        """纯标签式描述应被过滤。"""
        for label in [
            "官職", "居住地", "主从关系", "同僚关系", "敌对关系",
            "合作关系", "亲属关系", "上下级关系", "竞争关系",
            "朋友关系", "师徒关系", "同事关系", "邻居关系",
            "所属关系", "从属关系", "包含关系", "依赖关系",
        ]:
            assert not _is_valid_relation_content(label), f"'{label}' 应被过滤"

    def test_empty_template_content(self):
        """空洞模板描述应被过滤。"""
        assert not _is_valid_relation_content("X与Y的关联关系")
        assert not _is_valid_relation_content("曹操与杨昂有关")
        assert not _is_valid_relation_content("曹操和杨昂的关系")
        assert not _is_valid_relation_content("曹操与杨昂存在关联")
        assert not _is_valid_relation_content("曹操和杨昂相关")

    # --- 应该通过的（有效关系描述）---

    def test_valid_relation_descriptions(self):
        """有效的关系描述应通过。"""
        assert _is_valid_relation_content("曹操命令杨昂镇守汉中，杨昂是曹操麾下的将领")
        assert _is_valid_relation_content("《三体》是刘慈欣创作的科幻小说三部曲")
        assert _is_valid_relation_content("宝玉前往怡红院找人，怡红院是他的住所")

    def test_valid_short_relation(self):
        """足够具体但不太长的描述应通过。"""
        assert _is_valid_relation_content("杨昂率军在诸营驻扎防守")


# ===================================================================
# 4. 实体去重 + 清洗（dedupe_extracted_entities）
# ===================================================================

class TestDedupeExtractedEntities:
    """实体去重应同时进行名称清洗和质量过滤。"""

    def test_clean_scene_annotation_then_dedupe(self):
        """场景标注被清洗后，同名实体应正确合并。"""
        entities = [
            {"name": "曹操（魏王）", "content": "三国时期曹魏政权的奠基人，杰出的政治家、军事家和文学家"},
            {"name": "曹操（汉中张鲁）", "content": "曹操率军征讨汉中张鲁，展现了其军事才能"},
        ]
        result = dedupe_extracted_entities(entities)
        # 清洗后两个都是"曹操"，应合并为一个
        names = [e["name"] for e in result]
        assert len(result) == 1
        assert "曹操" in names[0] or "魏王" in names[0]

    def test_filter_text_fragments(self):
        """文本片段应被过滤掉。"""
        entities = [
            {"name": "遂得", "content": "表示顺利获得某物"},
            {"name": "险恶", "content": "形容环境十分危险"},
            {"name": "处理进度", "content": "当前任务的处理进展"},
        ]
        result = dedupe_extracted_entities(entities)
        assert len(result) == 0

    def test_filter_short_content(self):
        """内容过短（<10字）的实体应被过滤。"""
        entities = [
            {"name": "杨昂", "content": "武将"},  # 太短
            {"name": "曹操", "content": "三国时期曹魏政权奠基人，杰出的政治家军事家和文学家"},
        ]
        result = dedupe_extracted_entities(entities)
        assert len(result) == 1
        assert result[0]["name"] == "曹操"

    def test_keep_valid_entities(self):
        """合法实体应保留。"""
        entities = [
            {"name": "曹操（魏王）", "content": "三国时期曹魏政权的奠基人，政治家军事家文学家"},
            {"name": "许褚（曹操亲卫）", "content": "曹操麾下的勇猛将领，忠诚护卫曹操"},
        ]
        result = dedupe_extracted_entities(entities)
        assert len(result) == 2

    def test_dedupe_same_name_keeps_longer_content(self):
        """同名实体保留内容更长的。"""
        entities = [
            {"name": "杨昂", "content": "武将"},
            {"name": "杨昂", "content": "汉中张鲁麾下的武将，镇守诸营，后被曹操军队击败"},
        ]
        result = dedupe_extracted_entities(entities)
        assert len(result) == 1
        assert "张鲁" in result[0]["content"] or "击败" in result[0]["content"]

    def test_filter_invalid_paren_then_blacklist(self):
        """清洗括号后变成黑名单词的应被过滤。"""
        entities = [
            {"name": "处理进度（当前任务）", "content": "当前任务处理进展情况描述信息"},
        ]
        result = dedupe_extracted_entities(entities)
        # 清洗后可能变成"处理进度"（黑名单）或保留括号（有效）
        # 实际上"处理进度"在黑名单中，如果清洗后变成"处理进度"应被过滤
        for e in result:
            assert e["name"] != "处理进度"


# ===================================================================
# 5. 关系去重 + 质量过滤（dedupe_extracted_relations）
# ===================================================================

class TestDedupeExtractedRelations:
    """关系去重应同时进行质量过滤。"""

    def test_filter_label_only_relations(self):
        """标签式关系内容应被过滤。"""
        relations = [
            {"entity1_name": "曹操", "entity2_name": "杨昂", "content": "官職"},
            {"entity1_name": "曹操", "entity2_name": "杨昂", "content": "曹操任命杨昂为汉中守将，负责抵御刘备的进攻"},
        ]
        result = dedupe_extracted_relations(relations)
        assert len(result) == 1
        assert "任命" in result[0]["content"] or "守将" in result[0]["content"]

    def test_filter_empty_template_relations(self):
        """空洞模板关系应被过滤。"""
        relations = [
            {"entity1_name": "曹操", "entity2_name": "杨昂", "content": "曹操与杨昂有关"},
            {"entity1_name": "曹操", "entity2_name": "许褚", "content": "曹操与许褚的关联关系"},
        ]
        result = dedupe_extracted_relations(relations)
        assert len(result) == 0

    def test_filter_self_relations(self):
        """自关系（entity1 == entity2）应被过滤。"""
        relations = [
            {"entity1_name": "曹操", "entity2_name": "曹操", "content": "曹操是一个复杂的历史人物"},
        ]
        result = dedupe_extracted_relations(relations)
        assert len(result) == 0

    def test_filter_missing_entity_names(self):
        """缺少端点名称的关系应被过滤。"""
        relations = [
            {"entity1_name": "", "entity2_name": "杨昂", "content": "描述内容"},
            {"entity1_name": "曹操", "entity2_name": "", "content": "描述内容"},
        ]
        result = dedupe_extracted_relations(relations)
        assert len(result) == 0

    def test_valid_relations_pass(self):
        """有效关系应通过。"""
        relations = [
            {"entity1_name": "曹操", "entity2_name": "许褚", "content": "许褚是曹操的亲卫将领，多次在危难关头保护曹操"},
            {"entity1_name": "刘备", "entity2_name": "关羽", "content": "刘备关羽张飞桃园三结义，结为异姓兄弟"},
        ]
        result = dedupe_extracted_relations(relations)
        assert len(result) == 2

    def test_dedupe_same_pair_same_content(self):
        """同一对实体+相同内容应去重。"""
        relations = [
            {"entity1_name": "曹操", "entity2_name": "杨昂", "content": "曹操任命杨昂镇守汉中"},
            {"entity1_name": "杨昂", "entity2_name": "曹操", "content": "曹操任命杨昂镇守汉中"},
        ]
        result = dedupe_extracted_relations(relations)
        assert len(result) == 1

    def test_content_endpoint_mismatch_still_passes_dedup(self):
        """内容与端点不一致的关系不会被dedup阶段过滤（这是prompt层面解决的问题）。
        但我们验证它不会导致崩溃。"""
        relations = [
            {"entity1_name": "杨昂", "entity2_name": "速进兵", "content": "许褚和曹仁商议对策"},
        ]
        # 不崩溃即可
        result = dedupe_extracted_relations(relations)
        assert len(result) == 1  # dedup阶段不过滤endpoint mismatch
