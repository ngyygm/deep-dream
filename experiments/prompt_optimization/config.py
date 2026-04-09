"""Prompt optimization experiment configuration."""
import os

# LLM Configuration — read from service_config.json
_LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_CONFIG = {
    "api_key": _LLM_API_KEY,
    "model": "GLM-4-Flash",
    "base_url": "https://open.bigmodel.cn/api/paas/v4",
    "max_tokens": 5000,
    "timeout": 120,
}

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DISTILL_DIR = os.path.join(PROJECT_DIR, "distill_pipeline")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RESOURCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")

# Experiment parameters
TEST_WINDOWS_TOTAL = 20
RESOURCE_WINDOWS_TOTAL = 15
NOVELS = ["红楼梦", "三国演义", "水浒传", "西游记"]
NOVEL_WEIGHTS = {"红楼梦": 6, "三国演义": 6, "水浒传": 4, "西游记": 4}
RANDOM_SEED = 42

# Step directory names
STEP_DIRS = {
    1: "01_update_cache",
    2: "02_extract_entities",
    3: "03_extract_relations",
    4: "04_supplement_entities",
    5: "05_entity_enhancement",
    6: "06_entity_alignment",
    7: "07_relation_alignment",
}

# Noise detection constants
COMMON_VERBS = {
    "过", "来", "去", "看", "说", "想", "做", "走", "跑", "坐",
    "站", "听", "吃", "喝", "是", "有", "在", "到", "上", "下",
    "出", "起", "开", "回", "给", "让", "被", "把", "从", "向",
}
MENTION_PATTERNS = ["提到", "知道", "说", "问", "表示", "认为", "觉得", "指出", "说道", "问起"]
GENERIC_RELATION_PATTERNS = ["的关联关系", "的关系", "有关联", "有联系"]
