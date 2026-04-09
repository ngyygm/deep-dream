"""Configuration for dynamic prompt optimization experiments."""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DYNAMIC_OPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(DYNAMIC_OPT_DIR, "results")
TREES_DIR = os.path.join(DYNAMIC_OPT_DIR, "trees")

# Reuse paths from the existing prompt_optimization config
_PROMPT_OPT_DIR = os.path.join(os.path.dirname(DYNAMIC_OPT_DIR), "prompt_optimization")
DISTILL_DIR = os.path.join(PROJECT_DIR, "distill_pipeline")

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
NOVELS = ["红楼梦", "三国演义", "水浒传", "西游记"]
NOVEL_WEIGHTS = {"红楼梦": 6, "三国演义": 6, "水浒传": 4, "西游记": 4}
TEST_WINDOWS_TOTAL = 20
ALIGNMENT_ENTRIES_TOTAL = 30
RANDOM_SEED = 42

STEP_DIRS = {
    1: "01_update_cache",
    2: "02_extract_entities",
    3: "03_extract_relations",
    4: "04_supplement_entities",
    5: "05_entity_enhancement",
    6: "06_entity_alignment",
    7: "07_relation_alignment",
}

# ---------------------------------------------------------------------------
# LLM (reuse from prompt_optimization)
# ---------------------------------------------------------------------------
LLM_API_KEY = "2a827bfeb3b9403d82d4132bec7d1bf0.22qzXO7q7C6fqC42"
LLM_CONFIG = {
    "api_key": LLM_API_KEY,
    "model": "GLM-4-Flash",
    "base_url": "https://open.bigmodel.cn/api/paas/v4",
    "max_tokens": 5000,
    "timeout": 120,
}

# ---------------------------------------------------------------------------
# Search parameters
# ---------------------------------------------------------------------------
MAX_DEPTH = 6           # Maximum tree depth
MAX_BRANCHES = 3        # Maximum branches per node
QUICK_SAMPLE_N = 5      # Samples for quick filtering
FULL_SAMPLE_N = 20      # Samples for full evaluation

# ---------------------------------------------------------------------------
# Quick-filter thresholds
# ---------------------------------------------------------------------------
QUICK_MIN_PARSE_SUCCESS = 1.0   # Quick filter: must parse perfectly
QUICK_MIN_SCORE_RATIO = 0.85    # Quick filter: must be >= 85% of parent score
