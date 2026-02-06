from agent import MemoryRetrievalAgent, create_agent

# 快速创建Agent
agent = create_agent(
    storage_path='./graph/santi',
    llm_api_key="b0d9820b40ea427cb1df20af36fe2fc7.hv3LrdurAFxiXDGV",  # LLM API密钥（可选）
    llm_model="gemma3:27b",  # LLM模型名称（可选）
    # llm_base_url="http://219.223.187.56:9996/v1",  # LLM API基础URL（可选）
    llm_base_url="http://127.0.0.1:11434/v1",  # LLM API基础URL（可选）
)

# 执行检索
messages = [{"role": "user", "content": "叶文洁是谁？"}]
result = agent.retrieve(messages)

print(f"找到 {len(result.entities)} 个实体")
print(f"增强上下文: {result.augmented_context}")