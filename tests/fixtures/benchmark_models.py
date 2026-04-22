"""
Multi-Model Remember Pipeline Benchmark.

Runs 6 different LLM models through the remember pipeline with 8 diverse test texts,
compares extraction quality, alignment accuracy, and processing time.

Usage:
    python tests/benchmark_models.py

Requirements:
    - Neo4j running at bolt://localhost:7687
    - Embedding model at /home/linkco/exa/models/Qwen3-Embedding-0.6B (cuda:2)
    - API keys configured below

Output:
    - tests/benchmark_results.json  — summary metrics per model
    - tests/benchmark_details.json  — full entity/relation content per model per text
"""

import json
import os
import sys
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = [
    {
        "name": "glm-4-flash",
        "safe_name": "glm4flash",
        "llm_api_key": "8a8f599833dc41e397b30f522e2fd909.rSAbD6WFmncouiD7",
        "llm_model": "glm-4-flash",
        "llm_base_url": "https://open.bigmodel.cn/api/paas/v4",
    },
]

# Neo4j config (from service_config.json)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tmg2024secure"

# Embedding config
EMBEDDING_MODEL_PATH = "/home/linkco/exa/models/Qwen3-Embedding-0.6B"
EMBEDDING_DEVICE = "cuda:0"
VECTOR_DIM = 1024

# ---------------------------------------------------------------------------
# Test Texts (8 diverse domains, 1000–5000 chars)
# ---------------------------------------------------------------------------

TEST_TEXTS = {}

# Text A: 红楼梦人物群像 (~3000 chars) — entity merge accuracy test
TEST_TEXTS["红楼梦人物群像"] = """
贾宝玉是《红楼梦》中的男主角，荣国府贾政之子。他衔通灵宝玉而生，性情温润，不喜功名利禄，
偏爱诗词歌赋。贾宝玉与林黛玉自幼在荣国府长大，二人情投意合，常以诗词传情。林黛玉是贾宝玉
的表妹，父亲林如海曾任巡盐御史，母亲贾敏是贾母最疼爱的女儿。林黛玉体弱多病，才情出众，
性格敏感多疑，作有《葬花吟》等名篇。

薛宝钗是薛姨妈之女，随母亲和哥哥薛蟠进京待选。薛宝钗容貌丰美，举止端庄，为人处世圆滑老练。
她佩戴金锁，与贾宝玉的通灵宝玉暗合"金玉良缘"之说。薛宝钗精通诗词，才学不在林黛玉之下，
但性格比黛玉更加豁达。她常劝宝玉读书上进，这让宝玉颇为反感。

王熙凤是贾琏之妻，荣国府的实际管家人。她精明能干，口齿伶俐，善于察言观色。王熙凤虽然不识字，
但处事果断，将偌大的荣国府管理得井井有条。她与贾母关系亲密，深得贾母欢心。然而王熙凤为人心狠手辣，
曾弄权铁槛寺，逼死尤二姐。

贾母是荣国府的最高权威，史太君，疼爱孙子贾宝玉。贾母虽然年事已高，但精神矍铄，喜欢热闹。
她尤其疼爱林黛玉，因为黛玉是她的外孙女。贾母对薛宝钗也很欣赏，但在宝玉的婚事上，
她更倾向于选择林黛玉。不过最终在王夫人和薛姨妈的安排下，贾宝玉还是娶了薛宝钗为妻。

林黛玉在得知宝玉与宝钗成婚的消息后，焚稿断痴情，最终泪尽而亡。贾宝玉在得知黛玉已死后，
万念俱灰，虽与宝钗成亲，却始终无法忘怀黛玉。最终贾宝玉看破红尘，遁入空门，
留下薛宝钗独守空房。整个故事以贾、史、王、薛四大家族的兴衰为背景，
展现了封建社会中女性的悲剧命运和人生的虚幻无常。

史湘云是贾母的侄孙女，性格开朗活泼，心直口快。她自幼父母双亡，由叔父史家抚养长大。
史湘云才华横溢，曾在海棠诗社中大放异彩。她与林黛玉的性格形成鲜明对比，
黛玉多愁善感，湘云则豁达爽朗。湘云曾醉卧芍药花下，成为大观园中最美丽的风景之一。

妙玉是大观园中栊翠庵的尼姑，出身官宦世家，因自幼多病遁入空门。她孤高自许，目下无人，
但对贾宝玉却另眼相看。妙玉精通茶道，曾以梅花上的雪水烹茶招待宝钗和黛玉。
她性格清冷孤傲，却也是个多情之人，内心深处对世俗生活仍有留恋。
"""

# Text B: 三国演义 (~2000 chars) — historical military entities
TEST_TEXTS["三国演义"] = """
刘备，字玄德，涿郡涿县人，自称汉室宗亲，中山靖王刘胜之后。刘备少年丧父，与母亲贩履织席为生。
后与关羽、张飞桃园三结义，誓言"不求同年同月同日生，但愿同年同月同日死"。
刘备先后投靠公孙瓒、陶谦、曹操、袁绍、刘表等人，几经沉浮。三顾茅庐请出诸葛亮后，
在赤壁之战中大败曹操，占据荆州、益州，最终在成都称帝，建立蜀汉政权。

关羽，字云长，河东解良人。关羽武艺超群，忠义无双。他温酒斩华雄，斩颜良诛文丑，
千里走单骑，过五关斩六将，水淹七军，威震华夏。后被东吴吕蒙白衣渡江偷袭，兵败麦城，
父子皆被孙权所杀。关羽死后被追谥为"壮缪侯"，后世尊为"武圣"。

张飞，字翼德，涿郡人。张飞勇猛异常，曾在长坂坡上一声怒吼，吓退曹操百万大军。
他与关羽一同追随刘备征战天下，但性格暴躁，常因酒后鞭打士卒。在关羽被杀后，
张飞急于报仇，被部下范疆、张达刺杀。

诸葛亮，字孔明，号卧龙，琅琊阳都人。诸葛亮是三国时期最杰出的政治家和军事家。
他未出茅庐便定下三分天下之计。辅佐刘备建立蜀汉后，诸葛亮鞠躬尽瘁，六出祁山北伐中原。
他发明了木牛流马、连弩等军事器械，留下《出师表》《诫子书》等名篇。
最终诸葛亮因积劳成疾，病逝于五丈原，年仅五十四岁。

曹操，字孟德，沛国谯县人。曹操是杰出的政治家、军事家和文学家。他挟天子以令诸侯，
统一北方，建立魏国。曹操推行屯田制，兴修水利，唯才是举，促进了北方经济的恢复和发展。
他精通兵法，著有《孙子略解》。在文学上，曹操是建安文学的领袖，
留下《短歌行》《观沧海》《龟虽寿》等传世名篇。
"""

# Text C: 科技公司产品线 (~1500 chars) — tech entity extraction
TEST_TEXTS["科技公司产品线"] = """
智源科技（Beijing Zhiyuan Tech Co.）成立于2018年，总部位于北京中关村科技园，
是一家专注于大语言模型研发的人工智能公司。公司创始人张明远博士毕业于清华大学计算机系，
曾在谷歌大脑团队工作三年，参与过BERT和T5模型的研发。

智源科技的核心产品是"源神"系列大语言模型。源神-7B是开源的基础模型，
参数量70亿，在Hugging Face上获得了超过50万的下载量。源神-72B是旗舰模型，
在MMLU、C-Eval、GSM8K等多个基准测试中达到了同量级最优水平。
源神-Chat是基于源神-72B微调的对话模型，支持多轮对话、代码生成和函数调用。

公司的技术栈基于PyTorch和DeepSpeed框架，使用Megatron-LM进行分布式训练。
训练集群由256张A100 GPU组成，使用RoCE网络互连。推理服务基于vLLM框架，
支持Continuous Batching和PagedAttention技术。

智源科技在2023年完成了B轮融资，融资额2亿美元，由红杉资本领投，高瓴创投跟投。
公司目前有员工约300人，其中研发人员占比超过70%。主要客户包括腾讯、百度、字节跳动等互联网公司，
以及多家银行和保险机构。

公司的首席技术官李思华曾任职于OpenAI，负责GPT-3的工程化工作。
她领导团队开发了源神模型的RLHF对齐技术，使模型在安全性和有用性方面都达到了行业领先水平。
产品总监王浩然负责源神-Chat的产品化工作，此前他曾任职于微软亚洲研究院。
"""

# Text D: 新闻报道 (~1000 chars) — temporal entity extraction
TEST_TEXTS["财经新闻报道"] = """
新华社上海10月15日电 记者 王晓明 报道：国家发展和改革委员会今日发布数据，
今年前三季度全国固定资产投资达到42.5万亿元，同比增长5.8%。其中，高技术产业投资同比增长12.3%，
制造业投资增长9.1%，基础设施投资增长6.2%。

中国人民银行同日公布的数据显示，9月末广义货币M2余额289.6万亿元，同比增长10.3%。
前三季度社会融资规模增量为31.6万亿元，比上年同期多增2.8万亿元。

在股市方面，上证综合指数收盘报3258.76点，上涨1.2%；深证成份指数报10856.32点，上涨1.5%。
北向资金全天净买入87.6亿元，为连续第5个交易日净买入。

新能源汽车行业表现亮眼，比亚迪三季度销量突破100万辆，同比增长38.2%。
宁德时代动力电池装机量达到120GWh，全球市场份额达到37.5%。
理想汽车9月交付量达到4.2万辆，创历史新高。

商务部表示，将继续优化营商环境，加大吸引外资力度。今年1-9月全国实际使用外资金额
达到9199.7亿元人民币，其中高技术产业实际使用外资增长42%。
"""

# Text E: 技术文档 (~1500 chars) — code-related entities
TEST_TEXTS["开源项目文档"] = """
FastAPI-Vue-Admin 是一个基于 FastAPI + Vue3 + Element Plus 的后台管理系统，
采用前后端分离架构。后端使用 Python 3.11 + FastAPI 0.104 + SQLAlchemy 2.0 + Redis 5.0，
前端使用 Vue 3.3 + TypeScript 5.2 + Vite 5.0 + Pinia 2.1。

项目目录结构如下：
- backend/ — FastAPI 后端代码
  - app/api/ — API 路由层（v1 版本）
  - app/core/ — 核心配置（security.py, config.py）
  - app/models/ — SQLAlchemy ORM 模型
  - app/schemas/ — Pydantic 数据模型
  - app/services/ — 业务逻辑层
  - app/utils/ — 工具函数
- frontend/ — Vue3 前端代码
  - src/views/ — 页面组件
  - src/components/ — 公共组件
  - src/api/ — API 请求封装
  - src/store/ — Pinia 状态管理
- docker/ — Docker 部署配置
- docs/ — 项目文档

认证系统使用 JWT（JSON Web Token）双 token 方案：
Access Token 有效期 30 分钟，Refresh Token 有效期 7 天。
密码使用 bcrypt 算法加密，强度因子为 12。

数据库使用 PostgreSQL 15，ORM 采用 SQLAlchemy 2.0 的异步引擎。
缓存使用 Redis，支持单机和集群模式。消息队列使用 Celery + RabbitMQ，
用于处理异步任务如邮件发送、数据导出等。

项目使用 Alembic 管理数据库迁移，CI/CD 使用 GitHub Actions。
测试框架使用 pytest + pytest-asyncio，代码覆盖率目标为 80% 以上。
容器化部署使用 Docker Compose，包含 app、worker、nginx、postgres、redis 五个服务。

项目由开发者李明发起，目前在 GitHub 上有 2.3k stars，120+ forks。
核心贡献者包括前端工程师陈小红（负责 Vue3 重构）和后端工程师赵磊（负责异步架构改造）。
"""

# Text F: 系统日志 (~1000 chars) — structured text extraction
TEST_TEXTS["系统运维日志"] = """
[2024-03-15 08:23:45] [ERROR] [order-service] [192.168.1.101] OrderProcessingException: Failed to process order #ORD-20240315082345
    at com.example.order.service.OrderService.process(OrderService.java:234)
    at com.example.order.controller.OrderController.create(OrderController.java:89)
    Caused by: java.net.ConnectException: Connection refused to payment-service:8080
    Root cause: payment-service on host 192.168.1.102 is unreachable

[2024-03-15 08:24:01] [WARN] [api-gateway] [10.0.0.1] Rate limit exceeded for client IP 203.0.113.45
    Endpoint: POST /api/v2/orders
    Current rate: 156 req/s (limit: 100 req/s)
    Action: Request throttled, HTTP 429 returned

[2024-03-15 08:25:12] [INFO] [user-service] [192.168.1.103] User login successful
    User ID: usr_8a7f3c2d, Tenant: acme-corp, Role: admin
    Auth method: OAuth2 via Google SSO
    Session ID: sess_4e9b12a8f3
    Client: Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0

[2024-03-15 08:30:00] [ERROR] [inventory-service] [192.168.1.104] Database connection pool exhausted
    Active connections: 50/50, Wait queue: 23
    Last query: SELECT * FROM products WHERE category_id = 15 FOR UPDATE
    Thread: pool-worker-47, Duration: 8.2s
    Suggestion: Increase hikari.maximumPoolSize or optimize slow queries

[2024-03-15 08:35:22] [INFO] [notification-service] [192.168.1.105] Email batch sent
    Recipients: 2,847, Template: order_confirmation_v3
    SMTP server: smtp.acme-corp.com:587
    Success: 2,841, Failed: 6 (bounced), Latency: avg 1.2s

[2024-03-15 08:40:15] [CRITICAL] [kubernetes-scheduler] [10.0.0.50] Pod evicted: order-service-pod-7d8f9
    Node: worker-node-03, Reason: Node disk pressure (usage 95%)
    Evicted volumes: order-logs (12GB), order-cache (3.5GB)
    Rescheduled to: worker-node-05 at 08:40:28
"""

# Text G: 聊天对话 (~1500 chars) — colloquial text extraction
TEST_TEXTS["技术讨论对话"] = """
小王：各位，我最近在调研微服务框架，大家有什么推荐吗？我们团队用的是 Java 技术栈。

老张：如果你们用 Java 的话，Spring Cloud 生态最成熟。我们公司就是用 Spring Cloud + Nacos 做注册中心，
配合 Sentinel 做限流降级。不过说实话，配置确实比较复杂，尤其是分布式事务那块。

小李：我觉得可以考虑 Go 语言的 Kratos 框架。我们去年从 Spring Cloud 迁移到 Kratos 后，
服务启动时间从 30 秒降到了 2 秒，内存占用也从 512MB 降到了 64MB。
而且 Go 的并发模型在处理高 QPS 的场景下表现很好。

小王：Kratos 我也看过，但问题是团队里没人会 Go，学习成本太高了。
而且我们有大量的 Spring Security 的鉴权逻辑，迁移过去不现实。

老张：对，技术选型不能光看性能，团队能力和现有代码资产也很重要。
如果你们现在的系统没有明显的性能瓶颈，我觉得没必要换语言。

小李：那可以考虑用 Spring Cloud 2022 版本，新版本对 Kubernetes 的支持好了很多。
另外建议你们试试 Spring Native，编译成本地镜像后启动也很快。

小王：对了，我们的数据库现在用的是 MySQL 5.7，DBA 建议升级到 8.0。
大家觉得有必要吗？数据量大概每天 500 万条写入。

老张：MySQL 8.0 的性能提升确实很大，特别是 InnoDB 的改进和窗口函数的支持。
500 万日写入量的话，建议升级到 8.0，同时考虑分库分表。
我们用的是 ShardingSphere，支持得还不错。

小李：如果写入量继续增长的话，也可以考虑引入 TiDB 作为 MySQL 的替代。
TiDB 兼容 MySQL 协议，迁移成本低，而且天然支持水平扩展。
我们有个业务从 MySQL 迁移到 TiDB 后，写入吞吐提升了 3 倍。
"""

# Text H: 网页内容 (~2000 chars) — web page content extraction
TEST_TEXTS["百科网页内容"] = """
量子计算 — 维基百科

量子计算是一种利用量子力学原理进行信息处理的计算方式。与经典计算机使用比特（0或1）不同，
量子计算机使用量子比特（qubit），可以同时处于0和1的叠加态。

发展历史
量子计算的概念最早由理查德·费曼在1981年提出。费曼指出，模拟量子系统的经典计算机
需要指数级的时间，而量子计算机可以高效地完成这一任务。1985年，大卫·多伊奇提出了
量子图灵机的概念，奠定了量子计算的理论基础。

1994年，彼得·秀尔提出了秀尔算法，证明了量子计算机可以在多项式时间内分解大整数，
这对当前的RSA加密系统构成了潜在威胁。1996年，洛夫·格罗弗提出了格罗弗算法，
可以在无序数据库中以O(√N)的时间复杂度搜索目标元素。

主要量子计算平台
IBM Quantum：IBM 在2016年推出了 IBM Quantum Experience，允许公众通过云端访问量子计算机。
目前 IBM 最强大的量子处理器是 Condor，拥有1121个量子比特。

Google Sycamore：Google 的量子人工智能实验室在2019年宣布实现了"量子霸权"，
其53量子比特的 Sycamore 处理器在200秒内完成了经典超级计算机需要1万年才能完成的计算任务。

中国"九章"量子计算机：中国科学技术大学潘建伟院士团队在2020年构建了76个光子的量子计算原型机"九章"，
在处理高斯玻色采样问题上比最快的超级计算机快100万亿倍。

应用领域
量子计算的主要应用包括：密码学（量子密码、后量子密码）、药物发现（分子模拟）、
金融优化（投资组合优化）、材料科学（新材料设计）、人工智能（量子机器学习）。

技术挑战
量子计算面临的主要技术挑战包括量子退相干（量子态易受环境干扰而坍缩）、
量子纠错（需要大量物理量子比特来编码一个逻辑量子比特）、
以及量子比特的可扩展性（目前最大的量子处理器约有1000个量子比特，
而实用的量子计算机可能需要数百万个量子比特）。
"""


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def load_embedding_client():
    """Load the global shared embedding client."""
    from processor.storage.embedding import EmbeddingClient
    print(f"[INFO] Loading embedding model from {EMBEDDING_MODEL_PATH} on {EMBEDDING_DEVICE}...")
    client = EmbeddingClient(model_path=EMBEDDING_MODEL_PATH, device=EMBEDDING_DEVICE, use_local=True)
    print("[INFO] Embedding model loaded.")
    return client


def create_processor(model_config: dict, embedding_client, tmp_dir: str):
    """Create a TemporalMemoryGraphProcessor for a specific model."""
    from processor.pipeline.orchestrator import TemporalMemoryGraphProcessor

    config = {
        "storage": {
            "backend": "neo4j",
            "neo4j": {
                "uri": NEO4J_URI,
                "user": NEO4J_USER,
                "password": NEO4J_PASSWORD,
            },
            "vector_dim": VECTOR_DIM,
        },
    }

    proc = TemporalMemoryGraphProcessor(
        storage_path=tmp_dir,
        window_size=800,
        overlap=100,
        llm_api_key=model_config["llm_api_key"],
        llm_model=model_config["llm_model"],
        llm_base_url=model_config["llm_base_url"],
        embedding_client=embedding_client,
        embedding_use_local=False,
        max_llm_concurrency=1,
        llm_context_window_tokens=128000,
        llm_max_tokens=8192,
        config=config,
    )
    return proc


def clear_neo4j():
    """Clear all entities, relations, episodes, and concepts from Neo4j."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session(database="neo4j") as session:
        # Delete in order: relations first (they reference entities), then entities, episodes, etc.
        for label in ["MENTIONS", "ContentPatch", "EntityRedirect", "Relation", "Entity", "Episode"]:
            try:
                result = session.run(f"MATCH (n:{label}) DETACH DELETE n RETURN count(n) AS cnt")
                record = result.single()
                cnt = record["cnt"] if record else 0
                if cnt > 0:
                    print(f"  [CLEANUP] Deleted {cnt} {label} nodes")
            except Exception as e:
                print(f"  [CLEANUP] Error deleting {label}: {e}")
        # Also clean Concept labels
        try:
            session.run("MATCH (n:Concept) REMOVE n:Concept")
        except Exception:
            pass
        # Clean remaining nodes
        try:
            result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) AS cnt")
            record = result.single()
            cnt = record["cnt"] if record else 0
            if cnt > 0:
                print(f"  [CLEANUP] Deleted {cnt} remaining nodes")
        except Exception as e:
            print(f"  [CLEANUP] Error cleaning remaining: {e}")
    driver.close()


def extract_results(storage) -> Dict[str, Any]:
    """Extract all entities and relations from storage."""
    entities = storage.get_all_entities(exclude_embedding=True)
    relations = storage.get_all_relations()

    entity_data = []
    for e in entities:
        entity_data.append({
            "name": e.name,
            "family_id": e.family_id,
            "content": e.content or "",
            "summary": getattr(e, 'summary', '') or "",
            "confidence": getattr(e, 'confidence', None),
            "source_document": getattr(e, 'source_document', '') or "",
        })

    relation_data = []
    for r in relations:
        # Resolve entity names from absolute_ids
        e1_abs = r.entity1_absolute_id
        e2_abs = r.entity2_absolute_id
        e1_name = ""
        e2_name = ""
        # Try to find entity names from all entities
        for e in entities:
            if e.absolute_id == e1_abs:
                e1_name = e.name
            if e.absolute_id == e2_abs:
                e2_name = e.name
        if not e1_name:
            e1_name = e1_abs[:20] + "..." if e1_abs else "?"
        if not e2_name:
            e2_name = e2_abs[:20] + "..." if e2_abs else "?"

        relation_data.append({
            "entity1": e1_name,
            "entity2": e2_name,
            "family_id": r.family_id,
            "content": r.content or "",
            "confidence": getattr(r, 'confidence', None),
            "source_document": getattr(r, 'source_document', '') or "",
        })

    return {"entities": entity_data, "relations": relation_data}


def compute_metrics(entity_data: list, relation_data: list, wall_time: float) -> Dict[str, Any]:
    """Compute quantitative metrics for a model run."""
    entity_count = len(entity_data)
    relation_count = len(relation_data)

    # Average entity content length
    content_lengths = [len(e["content"]) for e in entity_data]
    avg_content_len = sum(content_lengths) / max(entity_count, 1)

    # Short entities (content < 15 chars)
    short_entities = sum(1 for e in entity_data if len(e["content"].strip()) < 15)

    # System leak entities (content contains pipeline system words)
    system_words = ["初步筛选", "精细化判断", "候选实体", "批量裁决", "抽取的关系", "待处理"]
    system_leak = sum(1 for e in entity_data if any(w in e["content"] for w in system_words))

    # Duplicate core names
    names = [e["name"] for e in entity_data]
    seen_names = set()
    duplicate_names = 0
    for n in names:
        if n in seen_names:
            duplicate_names += 1
        seen_names.add(n)

    # Orphan entities (entities with no relations)
    entity_names_with_relations = set()
    for r in relation_data:
        entity_names_with_relations.add(r["entity1"])
        entity_names_with_relations.add(r["entity2"])
    orphan_entities = sum(1 for e in entity_data if e["name"] not in entity_names_with_relations)

    # Average confidence
    entity_confidences = [e["confidence"] for e in entity_data if e["confidence"] is not None]
    avg_entity_confidence = sum(entity_confidences) / max(len(entity_confidences), 1)
    relation_confidences = [r["confidence"] for r in relation_data if r["confidence"] is not None]
    avg_relation_confidence = sum(relation_confidences) / max(len(relation_confidences), 1)

    return {
        "entity_count": entity_count,
        "relation_count": relation_count,
        "avg_entity_content_len": round(avg_content_len, 1),
        "short_entities": short_entities,
        "system_leak_entities": system_leak,
        "duplicate_core_names": duplicate_names,
        "orphan_entities": orphan_entities,
        "avg_entity_confidence": round(avg_entity_confidence, 3),
        "avg_relation_confidence": round(avg_relation_confidence, 3),
        "wall_time_seconds": round(wall_time, 1),
    }


# ---------------------------------------------------------------------------
# Keyword-based content accuracy checks
# ---------------------------------------------------------------------------

ACCURACY_CHECKS = {
    "红楼梦人物群像": {
        "贾宝玉不应含薛家": {
            "entity_names": ["贾宝玉"],
            "forbidden_keywords": ["薛家", "金锁", "丰美", "端庄"],
            "description": "贾宝玉 content should not contain 薛宝钗的描述",
        },
        "薛宝钗不应含通灵宝玉": {
            "entity_names": ["薛宝钗"],
            "forbidden_keywords": ["通灵宝玉", "衔玉", "遁入空门"],
            "description": "薛宝钗 content should not contain 贾宝玉独有特征",
        },
        "林黛玉不应含金玉良缘": {
            "entity_names": ["林黛玉"],
            "forbidden_keywords": ["金玉良缘", "金锁", "薛蟠"],
            "description": "林黛玉 content should not contain 薛家相关",
        },
        "王熙凤不应含诗词歌赋": {
            "entity_names": ["王熙凤"],
            "forbidden_keywords": ["诗词歌赋", "葬花吟", "海棠诗社"],
            "description": "王熙凤 content should not contain 其他人物特征",
        },
    },
    "三国演义": {
        "诸葛亮不应含桃园结义": {
            "entity_names": ["诸葛亮"],
            "forbidden_keywords": ["桃园结义", "三结义"],
            "description": "诸葛亮未参与桃园结义",
        },
        "关羽不应含三顾茅庐": {
            "entity_names": ["关羽"],
            "forbidden_keywords": ["三顾茅庐", "卧龙", "出师表"],
            "description": "关羽 content 不应含诸葛亮特征",
        },
    },
    "科技公司产品线": {
        "张明远不应含RLHF细节": {
            "entity_names": ["张明远"],
            "forbidden_keywords": ["RLHF", "对齐技术", "首席技术官"],
            "description": "张明远是创始人，不是CTO",
        },
    },
}


def check_content_accuracy(text_name: str, entity_data: list) -> Dict[str, Any]:
    """Check content accuracy using keyword rules."""
    checks = ACCURACY_CHECKS.get(text_name, {})
    if not checks:
        return {}

    results = {}
    for check_name, check in checks.items():
        violations = []
        for entity in entity_data:
            if entity["name"] in check["entity_names"]:
                for kw in check["forbidden_keywords"]:
                    if kw in entity["content"]:
                        violations.append({
                            "entity": entity["name"],
                            "keyword": kw,
                            "content_snippet": entity["content"][:200],
                        })
        results[check_name] = {
            "passed": len(violations) == 0,
            "violations": violations,
            "description": check["description"],
        }
    return results


# ---------------------------------------------------------------------------
# Main Benchmark Runner
# ---------------------------------------------------------------------------

def run_model(model_config: dict, embedding_client, text_items: list) -> Dict[str, Any]:
    """Run one model through all test texts, return results."""
    model_name = model_config["name"]
    safe_name = model_config["safe_name"]
    print(f"\n{'='*60}")
    print(f"  Model: {model_name} ({safe_name})")
    print(f"{'='*60}")

    tmp_dir = f"/tmp/benchmark_{safe_name}"
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        proc = create_processor(model_config, embedding_client, tmp_dir)
    except Exception as e:
        print(f"  [ERROR] Failed to create processor: {e}")
        traceback.print_exc()
        return {"error": str(e), "model": model_name}

    per_text_results = {}
    all_details = {}
    total_start = time.time()

    for text_name, text_content in text_items:
        text_len = len(text_content.strip())
        print(f"\n  [{model_name}] Processing: {text_name} ({text_len} chars)...")
        text_start = time.time()

        try:
            result = proc.remember_text(
                text_content.strip(),
                doc_name=text_name,
                verbose=False,
                verbose_steps=False,
            )
            text_time = time.time() - text_start

            # Extract entities and relations
            extracted = extract_results(proc.storage)

            # Compute metrics
            metrics = compute_metrics(extracted["entities"], extracted["relations"], text_time)

            # Check content accuracy
            accuracy = check_content_accuracy(text_name, extracted["entities"])

            per_text_results[text_name] = {
                "metrics": metrics,
                "accuracy": accuracy,
                "chunks_processed": result.get("chunks_processed", 0),
            }
            all_details[text_name] = extracted

            ent_cnt = len(extracted["entities"])
            rel_cnt = len(extracted["relations"])
            print(f"  [{model_name}] Done: {text_name} → {ent_cnt} entities, {rel_cnt} relations, {text_time:.1f}s")

        except Exception as e:
            text_time = time.time() - text_start
            print(f"  [{model_name}] ERROR on {text_name}: {e}")
            traceback.print_exc()
            per_text_results[text_name] = {"error": str(e), "metrics": {"wall_time_seconds": round(text_time, 1)}}
            all_details[text_name] = {"entities": [], "relations": []}

    total_time = time.time() - total_start
    print(f"\n  [{model_name}] Total time: {total_time:.1f}s")

    # Aggregate metrics
    all_entities = []
    all_relations = []
    for extracted in all_details.values():
        all_entities.extend(extracted.get("entities", []))
        all_relations.extend(extracted.get("relations", []))

    summary_metrics = compute_metrics(all_entities, all_relations, total_time)
    summary_metrics["per_text_time_seconds"] = round(total_time / max(len(text_items), 1), 1)

    # Clean up Neo4j for next model
    print(f"  [{model_name}] Cleaning up Neo4j...")
    try:
        clear_neo4j()
    except Exception as e:
        print(f"  [{model_name}] Cleanup error: {e}")

    # Clean tmp dir
    import shutil
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    return {
        "model": model_name,
        "summary": summary_metrics,
        "per_text": per_text_results,
        "details": all_details,
    }


def print_summary_table(all_results: list):
    """Print a summary comparison table to terminal."""
    print("\n" + "=" * 100)
    print("  Model Benchmark Results")
    print("=" * 100)

    # Header
    header = f"{'Model':<20s} | {'Ent':>4s} | {'Rel':>4s} | {'Short':>5s} | {'Leak':>4s} | {'Dup':>3s} | {'Orph':>4s} | {'Conf':>6s} | {'Time':>7s}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        if "error" in r:
            print(f"{r['model']:<20s} | ERROR: {r['error']}")
            continue
        s = r["summary"]
        print(f"{r['model']:<20s} | {s['entity_count']:>4d} | {s['relation_count']:>4d} | "
              f"{s['short_entities']:>5d} | {s['system_leak_entities']:>4d} | {s['duplicate_core_names']:>3d} | "
              f"{s['orphan_entities']:>4d} | {s['avg_entity_confidence']:>6.3f} | {s['wall_time_seconds']:>6.1f}s")

    # Per-text detail
    print("\n" + "=" * 100)
    print("  Per-Text Detail (entities / relations / time)")
    print("=" * 100)

    text_names = list(TEST_TEXTS.keys())
    # Header
    header_parts = [f"{'Text':<16s}"]
    for r in all_results:
        if "error" not in r:
            name = r["model"]
            if len(name) > 12:
                name = name[:12]
            header_parts.append(f"{name:>14s}")
    print(" | ".join(header_parts))
    print("-" * (len(" | ".join(header_parts))))

    for tn in text_names:
        row_parts = [f"{tn:<16s}"]
        for r in all_results:
            if "error" in r:
                continue
            pt = r["per_text"].get(tn, {})
            m = pt.get("metrics", {})
            if "error" in pt:
                row_parts.append(f"{'ERROR':>14s}")
            else:
                ent = m.get("entity_count", 0)
                rel = m.get("relation_count", 0)
                t = m.get("wall_time_seconds", 0)
                row_parts.append(f"{ent:>3d}/{rel:>3d}/{t:>4.0f}s")
        print(" | ".join(row_parts))

    # Accuracy check results
    print("\n" + "=" * 100)
    print("  Content Accuracy (keyword checks)")
    print("=" * 100)

    for r in all_results:
        if "error" in r:
            continue
        model_name = r["model"]
        total_checks = 0
        passed_checks = 0
        failed_checks = []
        for tn, pt in r["per_text"].items():
            acc = pt.get("accuracy", {})
            for check_name, check_result in acc.items():
                total_checks += 1
                if check_result.get("passed", True):
                    passed_checks += 1
                else:
                    failed_checks.append(f"{tn}/{check_name}")
                    for v in check_result.get("violations", []):
                        failed_checks.append(f"  → {v['entity']}: contains '{v['keyword']}'")

        print(f"\n  {model_name}: {passed_checks}/{total_checks} checks passed")
        if failed_checks:
            for fc in failed_checks:
                print(f"    ✗ {fc}")
        else:
            print(f"    ✓ All checks passed!")


def main():
    print("=" * 60)
    print("  Multi-Model Remember Pipeline Benchmark")
    print(f"  {len(MODELS)} models × {len(TEST_TEXTS)} texts")
    print(f"  Started at: {datetime.now().isoformat()}")
    print("=" * 60)

    # Load shared embedding client
    embedding_client = load_embedding_client()

    # Prepare text items
    text_items = list(TEST_TEXTS.items())

    # Clear Neo4j before starting
    print("\n[INFO] Clearing Neo4j before benchmark...")
    clear_neo4j()

    # Run models sequentially (shared Neo4j database)
    all_results = []
    for model_config in MODELS:
        result = run_model(model_config, embedding_client, text_items)
        all_results.append(result)

    # Print summary
    print_summary_table(all_results)

    # Save results
    results_path = PROJECT_ROOT / "tests" / "benchmark_results.json"
    details_path = PROJECT_ROOT / "tests" / "benchmark_details.json"

    # Save summary (without full content)
    summary_data = []
    for r in all_results:
        if "error" in r:
            summary_data.append(r)
            continue
        summary_data.append({
            "model": r["model"],
            "summary": r["summary"],
            "per_text": r["per_text"],
        })

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Summary saved to {results_path}")

    # Save full details (with entity/relation content)
    details_data = {}
    for r in all_results:
        if "error" in r:
            continue
        model_details = {}
        for text_name, extracted in r["details"].items():
            model_details[text_name] = {
                "entities": extracted["entities"],
                "relations": extracted["relations"],
            }
        details_data[r["model"]] = model_details

    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details_data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Full details saved to {details_path}")

    print(f"\n[INFO] Benchmark completed at: {datetime.now().isoformat()}")

    # Final Neo4j cleanup
    try:
        clear_neo4j()
    except Exception:
        pass


if __name__ == "__main__":
    main()
