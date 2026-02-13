"""
使用示例：展示如何使用 Temporal Memory Graph 处理文档
"""
from pathlib import Path
from processor import TemporalMemoryGraphProcessor


def example_basic_usage(processor):
    """基本使用示例"""
    
    # 自动读取 data 文件夹下的所有 .txt 文件
    data_dir = Path(__file__).parent / "data"
    document_paths = [
        str(data_dir / f.name) 
        for f in data_dir.glob("*.txt")
        if f.is_file()
    ]
    
    if not document_paths:
        print("警告：data 文件夹下没有找到 .txt 文件")
        return
    
    print(f"找到 {len(document_paths)} 个文档文件")
    
    processor.process_documents(
        document_paths, 
        verbose=True,
        similarity_threshold=0.5,  # 实体搜索相似度阈值（默认值，如果未指定下面的三个独立阈值则使用此值）
        max_similar_entities=5,  # 语义向量初筛后返回的最大相似实体数量（默认10）
        content_snippet_length=100,  # 用于相似度搜索的实体content截取长度（默认50字符）
        relation_content_snippet_length=256,  # 用于embedding计算的关系content截取长度（默认50字符）
        # 三种搜索方法的独立阈值配置（可选，如果未指定则使用similarity_threshold）
        jaccard_search_threshold=0.3,  # Jaccard搜索（name_only）的相似度阈值
        embedding_name_search_threshold=0.7,  # Embedding搜索（name_only）的相似度阈值
        embedding_full_search_threshold=0.3,  # Embedding搜索（name+content）的相似度阈值
        # 实体抽取配置
        entity_extraction_max_iterations=1,  # 实体抽取最大迭代次数（默认3次）后强制停止，防止无限循环
        entity_extraction_iterative=True,    # 是否启用迭代实体抽取（默认True）
        entity_post_enhancement=False,       # 是否启用实体后验增强（默认False，启用后会结合缓存记忆和当前text对实体content进行更细致的补全挖掘）
        # 关系抽取配置
        relation_extraction_max_iterations=1,  # 关系抽取最大迭代次数（默认3次）
        relation_extraction_absolute_max_iterations=5,  # 关系抽取绝对最大迭代次数（默认10次），超过
        relation_extraction_iterative=True,      # 是否启用迭代关系抽取（默认True）
        # LLM并行配置
        llm_threads=1,  # LLM并行访问线程数量（默认1，用于实体增强等可并行处理的阶段）
        # 缓存记忆配置
        load_cache_memory=True  # 是否加载缓存记忆（默认False，如果为True，会从storage_path下的memory_caches/json目录查找最新的cache并加载）
    )
    
    # 获取统计信息
    stats = processor.get_statistics()
    print(f"\n处理完成！统计信息: {stats}")


def example_consolidate_knowledge_graph(processor):
    """知识图谱整理示例
    
    用于整理已有的知识库，识别并合并重复实体，创建别名关系。
    适合在处理完文档后运行，用于优化知识图谱质量。
    """
    # 运行知识图谱整理
    # 该方法会：
    # 1. 获取所有实体
    # 2. 对每个实体，按name搜索相似实体（前max_candidates个）
    # 3. 对每个实体，按name+content搜索相似实体（前max_candidates个）
    # 4. 使用LLM分析候选实体，判断是否为同一实体或存在别名关系
    # 5. 执行实体合并（将重复的entity_id合并为一个）
    # 6. 创建别名关系边（如"雷政委"是"雷志成"的别名）
    # 7. 记录整理过程的缓存记忆
    result = processor.consolidate_knowledge_graph_entity(
        verbose=True,  # 输出详细信息
        similarity_threshold=0.4,  # 相似度搜索阈值
        max_candidates=20,  # 每次搜索返回的最大候选实体数
        batch_candidates=10,  # 每次批量处理的候选实体数（如果设置了且小于max_candidates，则分批处理）
        content_snippet_length=128,  # 传入LLM的实体content最大长度
        parallel=False,  # 启用多线程并行处理（需要llm_threads > 1）
        enable_name_match_step=False,  # 是否启用步骤1.5（按名称完全匹配进行初步整理）
        enable_pre_search=False  # 是否启用预搜索（步骤2）
    )
    
    # 输出整理结果
    print("\n整理结果详情:")
    print(f"  - 分析的实体数: {result['entities_analyzed']}")
    print(f"  - 合并的实体记录数: {result['entities_merged']}")
    print(f"  - 创建的别名关系数: {result['alias_relations_created']}")
    
    if result['merge_details']:
        print("\n合并操作详情:")
        for merge in result['merge_details']:
            print(f"  - {merge.get('merged_source_ids', [])} -> {merge.get('target_entity_id', '')}")
            if merge.get('reason'):
                print(f"    原因: {merge.get('reason')}")
    
    if result['alias_details']:
        print("\n别名关系详情:")
        for alias in result['alias_details']:
            print(f"  - {alias.get('from_name', '')} -> {alias.get('to_name', '')}")
            print(f"    描述: {alias.get('content', '')}")


if __name__ == "__main__":
    print("Temporal Memory Graph 使用示例")
    print("=" * 50)
    
    # 选择要运行的示例
    import sys

    # 初始化处理器
    processor = TemporalMemoryGraphProcessor(
        storage_path="./graph/santi",  # 存储路径
        window_size=800,  # 窗口大小：500字符
        overlap=200,  # 重叠大小：200字符
        llm_api_key="ollama",  # LLM API密钥（可选）
        llm_model="qwen3:14b",  # LLM模型名称（可选）
        llm_base_url="http://127.0.0.1:11434/v1",  # LLM API基础URL（可选）
        llm_think_mode=False,  # LLM是否开启think模式（默认True）。如果为False，会在prompt结尾添加/no_think
        # Embedding模型配置（可选）
        embedding_model_path="/home/linkco/exa/models/Qwen3-Embedding-0.6B",  # Embedding模型本地文件路径（优先使用）
        # embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",  # 或使用HuggingFace模型名称
        embedding_device="cuda:2"  # Embedding计算设备 ("cpu" 或 "cuda")
    )


    print("\n1. 基本使用示例（模拟LLM）:")
    example_basic_usage(processor)
    
    print("\n2. 知识图谱整理示例:")
    example_consolidate_knowledge_graph(processor)
    
    