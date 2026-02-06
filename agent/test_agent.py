"""
Agent 测试脚本

用于验证 Memory Retrieval Agent 的完整流程
"""
from decimal import Context
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent import MemoryRetrievalAgent


def test_basic_query():
    """测试基本查询流程"""
    print("\n" + "=" * 60)
    print("测试1: 基本查询流程（使用模拟 LLM）")
    print("=" * 60)
    
    # 创建 Agent（不配置 LLM，使用模拟客户端）
    agent = MemoryRetrievalAgent(
        verbose=True,
        log_level="moderate"
    )
    
    # 执行查询
    result = agent.query("这是一个测试问题")
    
    print(f"\n查询结果:")
    print(f"  - 迭代次数: {result.total_iterations}")
    print(f"  - 工具调用: {result.total_tool_calls}")
    print(f"  - 执行时间: {result.execution_time:.2f}s")
    print(f"  - 检索记忆数: {len(result.retrieved_memories)}")
    

def test_with_storage():
    """测试与真实存储的交互"""
    print("\n" + "=" * 60)
    print("测试2: 与真实存储交互")
    print("=" * 60)
    
    storage_path = project_root / "graph" / "santi"
    
    if not storage_path.exists():
        print(f"跳过测试：存储路径不存在 {storage_path}")
        return
    
    try:
        from processor.storage import StorageManager
        
        # 创建存储管理器
        sm = StorageManager(storage_path=str(storage_path))
        
        # 创建 Agent
        agent = MemoryRetrievalAgent(
            storage_managers=[sm],
            verbose=True,
            log_level="moderate"
        )
        
        # 测试工具调用
        print("\n测试工具直接调用:")
        
        # 搜索实体
        from agent.tools import SearchEntityTool
        search_tool = SearchEntityTool(sm)
        result = search_tool.execute(query="史强", threshold=0.3, max_results=5)
        print(f"\n搜索 '史强' 结果: {result.get('count', 0)} 个实体")
        if result.get("entities"):
            for e in result["entities"][:3]:
                print(f"  - {e.get('name')}: {e.get('content', '')[:50]}...")
        
    except ImportError as e:
        print(f"跳过测试：无法导入必要模块 - {e}")
    except Exception as e:
        print(f"测试失败: {e}")


def test_with_llm():
    """测试与真实 LLM 的交互"""
    print("\n" + "=" * 60)
    print("测试3: 与真实 LLM 交互")
    print("=" * 60)
    
    storage_path = project_root / "graph" / "santi"
    
    if not storage_path.exists():
        print(f"跳过测试：存储路径不存在 {storage_path}")
        return
    
    # LLM 配置（使用本地 Ollama）
    llm_config = {
        "api_key": "ollama",  # Ollama 不需要真正的 API key
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "gemma3:27b",  # 或其他已安装的模型
        "temperature": 0.7,
        "max_tokens": 8000
    }
    
    try:
        from processor.storage import StorageManager
        
        sm = StorageManager(storage_path=str(storage_path))
        
        agent = MemoryRetrievalAgent(
            storage_managers=[sm],
            llm_config=llm_config,
            verbose=True,
            log_level="moderate"
        )
        
        # 执行查询
        questions = [
            "史强是谁？",
            "史强和汪淼是什么关系？",
            "史强和汪淼第二次见面是什么时候？",
        ]
        
        for question in questions:
            print(f"\n问题: {question}")
            result = agent.query(question)
            
            print(f"\n查询结果:")
            print(f"  - 迭代次数: {result.total_iterations}")
            print(f"  - 工具调用: {result.total_tool_calls}")
            print(f"  - 执行时间: {result.execution_time:.2f}s")
            print(f"  - 检索记忆数: {len(result.retrieved_memories)}")
            
            # 打印答案（如果有）
            answer = agent.get_answer(result)
            if answer:
                confidence = agent.get_confidence(result)
                print(f"\n推理答案 (置信度: {confidence:.0%}):")
                print("-" * 40)
                print(answer)
                print("-" * 40)
            
            # 打印上下文
            context = agent.get_context_text(result)
            if context:
                print(f"\n上下文文本:")
                print("-" * 40)
                print(context)
                print("-" * 40)
            
    except ImportError as e:
        print(f"跳过测试：无法导入必要模块 - {e}")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_reasoning_cache():
    """测试推理缓存功能"""
    print("\n" + "=" * 60)
    print("测试5: 推理缓存功能")
    print("=" * 60)
    
    try:
        from agent.context import ReasoningCache, QuestionType, GoalStatus
        
        # 创建推理缓存
        cache = ReasoningCache()
        
        # 初始化状态
        state = cache.init_state(
            question="史强和汪淼第二次见面是什么时候？",
            question_type=QuestionType.TEMPORAL_REASONING
        )
        
        print(f"问题类型: {state.question_type.value}")
        
        # 添加子目标
        goal1 = cache.add_sub_goal("找到史强和汪淼两个实体")
        goal2 = cache.add_sub_goal("获取他们之间的所有关系", depends_on=[goal1.goal_id])
        goal3 = cache.add_sub_goal("确定各关系的时间顺序", depends_on=[goal2.goal_id])
        
        print(f"\n子目标:")
        for goal in state.sub_goals:
            print(f"  - {goal.goal_id}: {goal.description} [{goal.status.value}]")
        
        # 模拟完成第一个子目标
        cache.update_goal_status(goal1.goal_id, GoalStatus.COMPLETED, {"found": True})
        
        # 添加已知事实
        cache.add_entity_fact("ent_123", {"name": "史强", "type": "person"})
        cache.add_entity_fact("ent_456", {"name": "汪淼", "type": "person"})
        
        # 添加缺失信息
        cache.add_missing_info("史强和汪淼之间的关系列表")
        cache.add_missing_info("各关系的时间信息")
        
        # 添加假设
        hyp = cache.add_hypothesis("他们的第二次见面可能是在某个调查活动中", confidence=0.6)
        
        # 记录查询
        cache.record_query(
            tool_name="search_entity",
            parameters={"query": "史强"},
            iteration=1,
            success=True,
            result_summary="找到1个实体"
        )
        
        # 打印状态摘要
        print(f"\n推理状态摘要:")
        print("-" * 40)
        print(cache.get_state_summary())
        print("-" * 40)
        
        # 检查进度
        progress = cache.get_reasoning_progress()
        print(f"\n推理进度: {progress}")
        
        print("\n✓ 推理缓存测试通过")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_reasoning_with_llm():
    """测试带推理功能的 LLM 交互"""
    print("\n" + "=" * 60)
    print("测试6: 带推理功能的 LLM 交互")
    print("=" * 60)
    
    storage_path = project_root / "graph" / "santi"
    
    if not storage_path.exists():
        print(f"跳过测试：存储路径不存在 {storage_path}")
        return
    
    llm_config = {
        "api_key": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "gemma3:27b",
        "temperature": 0.7,
        "max_tokens": 8000
    }
    
    try:
        from processor.storage import StorageManager
        
        sm = StorageManager(storage_path=str(storage_path))
        
        agent = MemoryRetrievalAgent(
            storage_managers=[sm],
            llm_config=llm_config,
            verbose=True,
            log_level="verbose"  # 详细日志
        )
        
        # 测试复杂推理问题
        question = "史强和汪淼第一次认识是在什么情况下？"
        
        print(f"\n复杂推理问题: {question}")
        print("\n" + "-" * 60)
        
        result = agent.query(question, enable_reasoning=True)
        
        print("\n" + "-" * 60)
        print(f"\n查询结果:")
        print(f"  - 迭代次数: {result.total_iterations}")
        print(f"  - 工具调用: {result.total_tool_calls}")
        print(f"  - 执行时间: {result.execution_time:.2f}s")
        
        # 打印答案
        answer = agent.get_answer(result)
        if answer:
            confidence = agent.get_confidence(result)
            print(f"\n最终答案 (置信度: {confidence:.0%}):")
            print("=" * 40)
            print(answer)
            print("=" * 40)
        else:
            print("\n未能得出明确答案")
        
        # 打印推理链路
        for trace in result.reasoning_trace:
            if isinstance(trace, dict) and trace.get("type") == "summary":
                print(f"\n推理链路:")
                for step in trace.get("reasoning_chain", []):
                    if isinstance(step, dict):
                        print(f"  {step.get('step', '?')}. {step.get('action', '')}")
                    else:
                        print(f"  - {step}")
        
    except ImportError as e:
        print(f"跳过测试：无法导入必要模块 - {e}")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_tools_directly():
    """直接测试工具"""
    print("\n" + "=" * 60)
    print("测试4: 直接测试工具")
    print("=" * 60)
    
    storage_path = project_root / "graph" / "santi"
    
    if not storage_path.exists():
        print(f"跳过测试：存储路径不存在 {storage_path}")
        return
    
    try:
        from processor.storage import StorageManager
        from agent.tools import (
            SearchEntityTool,
            GetRelationsTool,
            GetVersionsTool,
            SearchRelationsTool
        )
        
        sm = StorageManager(storage_path=str(storage_path))
        
        # 测试搜索实体
        print("\n1. 搜索实体 '汪淼':")
        search_tool = SearchEntityTool(sm)
        result = search_tool.execute(query="汪淼", threshold=0.3)
        print(f"   结果: {result.get('message')}")
        
        if result.get("entities"):
            entity = result["entities"][0]
            entity_id = entity.get("entity_id")
            print(f"   实体ID: {entity_id}")
            print(f"   内容: {entity.get('content', '')[:100]}...")
            
            # 测试获取关系
            print(f"\n2. 获取实体关系:")
            rel_tool = GetRelationsTool(sm)
            rel_result = rel_tool.execute(entity_id=entity_id, limit=5)
            print(f"   结果: {rel_result.get('message')}")
            
            if rel_result.get("relations"):
                for rel in rel_result["relations"][:3]:
                    print(f"   - [{rel.get('entity1_name')}] -- [{rel.get('entity2_name')}]")
                    print(f"     {rel.get('content', '')[:80]}...")
            
            # 测试获取版本
            print(f"\n3. 获取实体版本历史:")
            ver_tool = GetVersionsTool(sm)
            ver_result = ver_tool.execute(target_type="entity", target_id=entity_id)
            print(f"   结果: {ver_result.get('message')}")
            if ver_result.get("earliest_time"):
                print(f"   最早时间: {ver_result['earliest_time']}")
        
        # 测试搜索关系
        print(f"\n4. 搜索关系 '见面':")
        search_rel_tool = SearchRelationsTool(sm)
        rel_search_result = search_rel_tool.execute(query="见面", threshold=0.1, max_results=3)
        print(f"   结果: {rel_search_result.get('message')}")
        
        if rel_search_result.get("relations"):
            for rel in rel_search_result["relations"]:
                print(f"   - [{rel.get('entity1_name')}] -- [{rel.get('entity2_name')}]")
                print(f"     {rel.get('content', '')[:80]}...")
        
        # 测试多跳路径搜索
        print(f"\n5. 测试多跳路径搜索:")
        from agent.tools.get_relation_paths import GetRelationPathsTool
        path_tool = GetRelationPathsTool(sm)
        
        # 先获取两个实体
        entity1_result = search_tool.execute(query="史强", threshold=0.3)
        entity2_result = search_tool.execute(query="汪淼", threshold=0.3)
        
        if entity1_result.get("entities") and entity2_result.get("entities"):
            entity1_id = entity1_result["entities"][0].get("entity_id")
            entity2_id = entity2_result["entities"][0].get("entity_id")
            
            print(f"   查找 {entity1_id} 到 {entity2_id} 的路径...")
            path_result = path_tool.execute(
                entity1_id=entity1_id,
                entity2_id=entity2_id,
                max_hops=3,
                max_paths=5
            )
            print(f"   结果: {path_result.get('message')}")
            
            if path_result.get("paths"):
                for i, path in enumerate(path_result["paths"]):
                    print(f"   路径 {i+1} ({path.get('hop_count')} 跳):")
                    print(f"     {path.get('path_description', '')[:100]}...")
        else:
            print(f"   跳过：无法找到测试实体")
        
    except ImportError as e:
        print(f"跳过测试：无法导入必要模块 - {e}")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_multi_hop_relations(entity1_id=None, entity2_id=None, entity1_name=None, entity2_name=None):
    """测试两个实体ID之间的多跳关系检索（跨所有版本）
    
    Args:
        entity1_id: 第一个实体ID（可选）
        entity2_id: 第二个实体ID（可选）
        entity1_name: 第一个实体名称（用于搜索，可选）
        entity2_name: 第二个实体名称（用于搜索，可选）
    """
    print("\n" + "=" * 60)
    print("测试: 两个实体ID之间的多跳关系检索（跨所有版本）")
    print("=" * 60)
    
    storage_path = project_root / "graph" / "santi"
    
    if not storage_path.exists():
        print(f"跳过测试：存储路径不存在 {storage_path}")
        return
    
    try:
        from processor.storage import StorageManager
        from agent.tools.get_relation_paths import GetRelationPathsTool
        from agent.tools import SearchEntityTool
        
        sm = StorageManager(storage_path=str(storage_path))
        path_tool = GetRelationPathsTool(sm)
        search_tool = SearchEntityTool(sm)
        
        # 获取两个实体ID
        if not entity1_id or not entity2_id:
            # 尝试通过名称搜索
            if entity1_name:
                print(f"\n正在搜索实体1: {entity1_name}")
                entity1_result = search_tool.execute(query=entity1_name, threshold=0.3, max_results=1)
                if entity1_result.get("entities"):
                    entity1_id = entity1_result["entities"][0].get("entity_id")
                    print(f"  找到: {entity1_result['entities'][0].get('name')} (ID: {entity1_id})")
            
            if entity2_name:
                print(f"\n正在搜索实体2: {entity2_name}")
                entity2_result = search_tool.execute(query=entity2_name, threshold=0.3, max_results=1)
                if entity2_result.get("entities"):
                    entity2_id = entity2_result["entities"][0].get("entity_id")
                    print(f"  找到: {entity2_result['entities'][0].get('name')} (ID: {entity2_id})")
            
            # 如果还是没有，尝试默认示例
            if not entity1_id or not entity2_id:
                print("\n使用默认示例实体...")
                entity1_result = search_tool.execute(query="史强", threshold=0.3, max_results=1)
                entity2_result = search_tool.execute(query="汪淼", threshold=0.3, max_results=1)
                
                if entity1_result.get("entities") and entity2_result.get("entities"):
                    entity1_id = entity1_result["entities"][0].get("entity_id")
                    entity2_id = entity2_result["entities"][0].get("entity_id")
                    print(f"  实体1: {entity1_result['entities'][0].get('name')} (ID: {entity1_id})")
                    print(f"  实体2: {entity2_result['entities'][0].get('name')} (ID: {entity2_id})")
        
        if not entity1_id or not entity2_id:
            print("错误: 无法获取实体ID，请通过参数指定或确保数据库中有示例实体")
            return
        
        # 显示实体信息
        entity1 = sm.get_entity_by_id(entity1_id)
        entity2 = sm.get_entity_by_id(entity2_id)
        
        if not entity1:
            print(f"错误: 实体1不存在 (ID: {entity1_id})")
            return
        if not entity2:
            print(f"错误: 实体2不存在 (ID: {entity2_id})")
            return
        
        print(f"\n实体信息:")
        print(f"  实体1: {entity1.name}")
        print(f"    - ID: {entity1.entity_id}")
        print(f"    - 内容: {entity1.content[:100]}...")
        
        # 获取实体1的所有版本
        versions1 = sm.get_entity_versions(entity1_id)
        print(f"    - 版本数: {len(versions1)}")
        if versions1:
            print(f"    - 最早版本时间: {min(v.physical_time for v in versions1)}")
            print(f"    - 最新版本时间: {max(v.physical_time for v in versions1)}")
        
        print(f"\n  实体2: {entity2.name}")
        print(f"    - ID: {entity2.entity_id}")
        print(f"    - 内容: {entity2.content[:100]}...")
        
        # 获取实体2的所有版本
        versions2 = sm.get_entity_versions(entity2_id)
        print(f"    - 版本数: {len(versions2)}")
        if versions2:
            print(f"    - 最早版本时间: {min(v.physical_time for v in versions2)}")
            print(f"    - 最新版本时间: {max(v.physical_time for v in versions2)}")
        
        # 测试直接关系
        print(f"\n" + "-" * 60)
        print("1. 检查直接关系（所有版本）")
        print("-" * 60)
        direct_relations = sm.get_relations_by_entities(entity1_id, entity2_id)
        print(f"找到 {len(direct_relations)} 个直接关系（跨所有版本）")
        
        if direct_relations:
            # 按relation_id去重，显示每个关系的最新版本
            seen_relation_ids = set()
            for rel in sorted(direct_relations, key=lambda r: r.physical_time, reverse=True):
                if rel.relation_id not in seen_relation_ids:
                    seen_relation_ids.add(rel.relation_id)
                    print(f"\n  关系ID: {rel.relation_id}")
                    print(f"    内容: {rel.content[:150]}...")
                    print(f"    时间: {rel.physical_time}")
                    
                    # 获取该关系的所有版本
                    rel_versions = sm.get_relation_versions(rel.relation_id)
                    if len(rel_versions) > 1:
                        print(f"    版本数: {len(rel_versions)}")
                        for v in sorted(rel_versions, key=lambda r: r.physical_time):
                            print(f"      - {v.physical_time}: {v.content[:80]}...")
        
        # 测试多跳关系
        print(f"\n" + "-" * 60)
        print("2. 检索多跳关系路径（跨所有版本）")
        print("-" * 60)
        
        max_hops = 2
        max_paths = 50
        
        print(f"参数: max_hops={max_hops}, max_paths={max_paths}")
        print(f"正在搜索 {entity1.name} 到 {entity2.name} 之间的路径...")
        
        path_result = path_tool.execute(
            entity1_id=entity1_id,
            entity2_id=entity2_id,
            max_hops=max_hops,
            max_paths=max_paths,
            include_relation_content=True
        )
        
        print(f"\n结果: {path_result.get('message')}")
        
        if path_result.get("success") and path_result.get("paths"):
            paths = path_result["paths"]
            shortest_length = path_result.get("shortest_path_length", -1)
            
            print(f"\n找到 {len(paths)} 条路径，最短路径长度为 {shortest_length} 跳")
            
            for i, path in enumerate(paths, 1):
                print(f"\n路径 {i} ({path.get('hop_count')} 跳):")
                print(f"  {path.get('path_description', '')}")
                
                # 显示路径上的节点
                nodes = path.get("nodes", [])
                print(f"\n  节点 ({len(nodes)} 个):")
                for j, node in enumerate(nodes):
                    print(f"    {j+1}. {node.get('name')} (ID: {node.get('entity_id')})")
                    if node.get('content'):
                        print(f"       内容: {node.get('content')[:80]}...")
                
                # 显示路径上的边（关系）
                edges = path.get("edges", [])
                print(f"\n  关系边 ({len(edges)} 条):")
                for j, edge in enumerate(edges):
                    print(f"    {j+1}. [{edge.get('entity1_name')}] -- [{edge.get('entity2_name')}]")
                    print(f"       关系ID: {edge.get('relation_id')}")
                    print(f"       时间: {edge.get('physical_time')}")
                    if edge.get('content'):
                        print(f"       内容: {edge.get('content')[:100]}...")
                    
                    # 检查该关系的所有版本
                    rel_id = edge.get('relation_id')
                    if rel_id:
                        rel_versions = sm.get_relation_versions(rel_id)
                        if len(rel_versions) > 1:
                            print(f"       版本数: {len(rel_versions)}")
                            for v in sorted(rel_versions, key=lambda r: r.physical_time):
                                print(f"         - {v.physical_time}: {v.content[:60]}...")
        else:
            print("\n未找到路径")
        
        print("\n✓ 多跳关系检索测试完成")
        
    except ImportError as e:
        print(f"跳过测试：无法导入必要模块 - {e}")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent 测试脚本")
    parser.add_argument("--test", type=str, default="all",
                       choices=["all", "basic", "storage", "llm", "tools", "reasoning", "reasoning_llm", "multi_hop"],
                       help="要运行的测试")
    parser.add_argument("--entity1_id", type=str, default=None,
                       help="第一个实体ID（用于multi_hop测试）")
    parser.add_argument("--entity2_id", type=str, default=None,
                       help="第二个实体ID（用于multi_hop测试）")
    parser.add_argument("--entity1_name", type=str, default=None,
                       help="第一个实体名称（用于multi_hop测试，将通过搜索获取ID）")
    parser.add_argument("--entity2_name", type=str, default=None,
                       help="第二个实体名称（用于multi_hop测试，将通过搜索获取ID）")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Memory Retrieval Agent 测试")
    print("=" * 60)
    
    if args.test in ["all", "basic"]:
        test_basic_query()
    
    if args.test in ["all", "tools"]:
        test_tools_directly()
    
    if args.test in ["all", "storage"]:
        test_with_storage()
    
    if args.test in ["all", "reasoning"]:
        test_reasoning_cache()
    
    if args.test in ["all", "llm"]:
        test_with_llm()
    
    if args.test in ["reasoning_llm"]:
        test_reasoning_with_llm()
    
    if args.test in ["all", "multi_hop"]:
        test_multi_hop_relations(
            entity1_id=args.entity1_id,
            entity2_id=args.entity2_id,
            entity1_name=args.entity1_name,
            entity2_name=args.entity2_name
        )
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
