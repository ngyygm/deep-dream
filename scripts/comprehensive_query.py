#!/usr/bin/env python3
"""
综合查询脚本 - 自动编排查询策略

功能：
- 自动识别问题类型
- 自动选择查询策略
- 自动扩展信息不足的情况
- 支持多步推理查询

使用方法：
    python comprehensive_query.py --question "汪淼和牛顿之间发生了什么对话？"
    python comprehensive_query.py --question "主角是如何发现秘密的？" --auto-expand
"""

import sys
import argparse
import re
from pathlib import Path

# 添加DeepDream到路径（支持多种路径）
from pathlib import Path

# 方法1: 从scripts目录向上找
script_dir = Path(__file__).parent
tmg_dir = script_dir.parent.parent.parent
if (tmg_dir / "processor").exists():
    sys.path.insert(0, str(tmg_dir))
else:
    # 方法2: 使用绝对路径
    sys.path.insert(0, str(Path("/home/linkco/exa/DeepDream")))

try:
    from processor.storage import StorageManager
except ImportError:
    print("错误: 无法导入 StorageManager")
    print("请确保 DeepDream 在正确的位置")
    sys.exit(1)



class QuestionAnalyzer:
    """问题分析器"""

    @staticmethod
    def analyze(question):
        """分析问题类型"""
        question_lower = question.lower()

        # 识别问题类型
        if QuestionAnalyzer._is_entity_relation_question(question_lower):
            return 'entity_relation'

        elif QuestionAnalyzer._is_single_entity_question(question_lower):
            return 'single_entity'

        elif QuestionAnalyzer._is_network_question(question_lower):
            return 'network'

        elif QuestionAnalyzer._is_timeline_question(question_lower):
            return 'timeline'

        elif QuestionAnalyzer._is_search_question(question_lower):
            return 'search'

        elif QuestionAnalyzer._is_reconstruction_question(question_lower):
            return 'reconstruction'

        else:
            return 'general'

    @staticmethod
    def _is_entity_relation_question(question):
        """是否是实体关系问题"""
        patterns = [
            r'(.{1,5})和(.{1,5})的?(关系|互动|对话|交流)',
            r'(.{1,5})与(.{1,5})的?(关系|互动|对话|交流)',
            r'(.{1,5})和(.{1,5})',
        ]
        return any(re.search(p, question) for p in patterns)

    @staticmethod
    def _is_single_entity_question(question):
        """是否是单实体问题"""
        patterns = [
            r'(.{1,10})是谁',
            r'(.{1,10})是什么',
            r'告诉我关于(.{1,10})',
            r'描述(.{1,10})',
        ]
        return any(re.search(p, question) for p in patterns)

    @staticmethod
    def _is_network_question(question):
        """是否是关系网络问题"""
        patterns = [
            r'关系网络',
            r'(展示|显示).{0,5}关系',
            r'和谁有联系',
            r'关联图',
        ]
        return any(re.search(p, question) for p in patterns)

    @staticmethod
    def _is_timeline_question(question):
        """是否是时间线问题"""
        patterns = [
            r'(时间|变化|发展|演变)',
            r'(开始|最初|最初时)',
            r'(过程|经历)',
            r'时间线',
        ]
        return any(re.search(p, question) for p in patterns)

    @staticmethod
    def _is_search_question(question):
        """是否是搜索问题"""
        patterns = [
            r'(搜索|查找|找)',
            r'(哪些|哪些地方|哪里)',
            r'(提到|关于)',
        ]
        return any(re.search(p, question) for p in patterns)

    @staticmethod
    def _is_reconstruction_question(question):
        """是否是事件重建问题"""
        patterns = [
            r'(如何|怎么)(发现|知道|了解)',
            r'(经过|过程|来龙去脉)',
            r'(重建|还原|描述)(事件|过程)',
        ]
        return any(re.search(p, question) for p in patterns)


class QueryOrchestrator:
    """查询编排器"""

    def __init__(self, storage):
        self.storage = storage

    def execute_query(self, question, auto_expand=True):
        """执行查询"""
        print("=" * 80)
        print(f"综合查询: {question}")
        print("=" * 80)

        # 分析问题
        print("\n【步骤1】分析问题类型")
        print("-" * 80)

        question_type = QuestionAnalyzer.analyze(question)
        print(f"问题类型: {question_type}")

        # 根据类型执行查询
        if question_type == 'entity_relation':
            return self._handle_entity_relation(question, auto_expand)

        elif question_type == 'single_entity':
            return self._handle_single_entity(question, auto_expand)

        elif question_type == 'network':
            return self._handle_network(question)

        elif question_type == 'search':
            return self._handle_search(question)

        elif question_type == 'timeline':
            return self._handle_timeline(question)

        elif question_type == 'reconstruction':
            return self._handle_reconstruction(question, auto_expand)

        else:
            return self._handle_general(question, auto_expand)

    def _handle_entity_relation(self, question, auto_expand):
        """处理实体关系问题"""
        print("\n【步骤2】提取实体名称")
        print("-" * 80)

        # 提取两个实体名称（简单实现）
        entities = self._extract_entity_names(question)
        print(f"提取的实体: {entities}")

        if len(entities) < 2:
            print("错误: 需要两个实体名称")
            return None

        # 查询关系
        print(f"\n【步骤3】查询实体关系")
        print("-" * 80)

        # 这里可以调用 query_entity_relation.py 的逻辑
        # 为简化，直接查询
        entity1 = self._search_entity(entities[0])
        entity2 = self._search_entity(entities[1])

        if not entity1 or not entity2:
            print("错误: 找不到指定实体")
            return None

        relations = self.storage.get_relations_by_entities(
            entity1.family_id,
            entity2.family_id
        )

        print(f"\n找到 {len(relations)} 条直接关系")

        if relations:
            for i, rel in enumerate(relations, 1):
                print(f"\n关系 {i}:")
                print(f"  内容: {rel.content}")
                print(f"  时间: {rel.event_time}")
        else:
            print("未找到直接关系")

        if auto_expand and len(relations) < 2:
            print("\n【步骤4】自动扩展查询")
            print("-" * 80)
            print("关系数量较少，正在扩展...")

        return {'relations': relations, 'entity1': entity1, 'entity2': entity2}

    def _handle_single_entity(self, question, auto_expand):
        """处理单实体问题"""
        print("\n【步骤2】提取实体名称")
        print("-" * 80)

        entity_name = self._extract_single_entity_name(question)
        print(f"提取的实体: {entity_name}")

        entity = self._search_entity(entity_name)
        if not entity:
            print("错误: 找不到实体")
            return None

        print(f"\n【步骤3】查询实体信息")
        print("-" * 80)
        print(f"名称: {entity.name}")
        print(f"描述: {entity.content}")

        # 查询关系
        relations = self.storage.get_entity_relations_by_family_id(
            entity.family_id,
            limit=20
        )

        print(f"\n【步骤4】查询关系网络")
        print("-" * 80)
        print(f"找到 {len(relations)} 条关系")

        return {'entity': entity, 'relations': relations}

    def _handle_network(self, question):
        """处理关系网络问题"""
        print("\n【步骤2】提取实体名称")
        print("-" * 80)

        entity_name = self._extract_single_entity_name(question)
        print(f"中心实体: {entity_name}")

        entity = self._search_entity(entity_name)
        if not entity:
            return None

        print(f"\n提示: 使用 query_relations_network.py 获取完整网络图")
        print(f"命令: python scripts/query_relations_network.py --entity \"{entity_name}\" --depth 2")

        return {'entity': entity}

    def _handle_search(self, question):
        """处理搜索问题"""
        print("\n【步骤2】提取搜索关键词")
        print("-" * 80)

        # 提取关键词（简单实现）
        keywords = self._extract_keywords(question)
        print(f"搜索关键词: {keywords}")

        print(f"\n提示: 使用 search_memory_cache.py 进行内容搜索")
        print(f"命令: python scripts/search_memory_cache.py --keywords \"{keywords}\"")

        return {'keywords': keywords}

    def _handle_timeline(self, question):
        """处理时间线问题"""
        print("\n【步骤2】提取实体名称")
        print("-" * 80)

        entity_name = self._extract_single_entity_name(question)
        print(f"实体: {entity_name}")

        print(f"\n提示: 使用 query_single_entity.py --include-versions 查看时间线")
        print(f"命令: python scripts/query_single_entity.py --entity \"{entity_name}\" --include-versions")

        return {'entity': entity_name}

    def _handle_reconstruction(self, question, auto_expand):
        """处理事件重建问题"""
        print("\n【步骤2】分析事件关键词")
        print("-" * 80)

        keywords = self._extract_keywords(question)
        print(f"关键词: {keywords}")

        print("\n【步骤3】搜索相关实体和缓存")
        print("-" * 80)

        # 搜索相关实体
        entities = []
        for kw in keywords.split():
            found = self.storage.search_entities_by_similarity(
                query_name=kw,
                threshold=0.3,
                max_results=3
            )
            entities.extend(found)

        print(f"找到 {len(entities)} 个相关实体")

        # 搜索缓存
        cache_files = list(self.storage.cache_json_dir.glob("*.json"))
        matched_caches = []

        for cache_file in cache_files[:20]:
            try:
                cache = self.storage.load_episode(cache_file.stem)
                if cache and any(kw.lower() in cache.content.lower() for kw in keywords.split()):
                    matched_caches.append(cache)
            except:
                continue

        print(f"找到 {len(matched_caches)} 个相关缓存")

        return {'entities': entities, 'caches': matched_caches}

    def _handle_general(self, question, auto_expand):
        """处理通用问题"""
        print("\n【步骤2】尝试多种查询策略")
        print("-" * 80)

        # 尝试提取实体
        entity_name = self._extract_single_entity_name(question)

        if entity_name:
            print(f"找到实体名称: {entity_name}")
            return self._handle_single_entity(question, auto_expand)
        else:
            print("未找到明确的实体，尝试内容搜索")
            return self._handle_search(question)

    def _search_entity(self, entity_name, threshold=0.3):
        """搜索实体"""
        entities = self.storage.search_entities_by_similarity(
            query_name=entity_name,
            threshold=threshold,
            max_results=1
        )
        return entities[0] if entities else None

    def _extract_entity_names(self, text):
        """提取实体名称（简单实现）"""
        # 查找中文字符串
        matches = re.findall(r'[\u4e00-\u9fff]{2,10}', text)
        return matches[:2] if matches else []

    def _extract_single_entity_name(self, text):
        """提取单个实体名称"""
        matches = re.findall(r'[\u4e00-\u9fff]{2,10}', text)
        return matches[0] if matches else ""

    def _extract_keywords(self, text):
        """提取关键词"""
        # 简单实现：移除常见疑问词，保留内容词
        stopwords = ['的', '是', '了', '和', '与', '在', '有', '什么', '怎么', '如何']
        keywords = text
        for sw in stopwords:
            keywords = keywords.replace(sw, ' ')
        return ' '.join(keywords.split())


def main():
    parser = argparse.ArgumentParser(
        description='综合查询 - 自动编排查询策略',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--question', required=True, help='要查询的问题')
    parser.add_argument('--storage-path', default='/home/linkco/exa/DeepDream/graph/santi')
    parser.add_argument('--auto-expand', action='store_true',
                       help='当信息不足时自动扩展')

    args = parser.parse_args()

    storage = StorageManager(args.storage_path)

    # 创建编排器并执行查询
    orchestrator = QueryOrchestrator(storage)
    result = orchestrator.execute_query(args.question, args.auto_expand)

    print("\n" + "=" * 80)
    print("查询完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
