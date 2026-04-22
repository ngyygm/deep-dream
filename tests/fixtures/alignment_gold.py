"""
Gold-standard alignment test pairs for testing entity alignment accuracy.

This module provides pre-built entity/relation pairs organized by difficulty
level. Each pair contains two entity descriptions that should be judged as
the same entity, different entities, or are intentionally borderline.

Usage::

    from tests.fixtures.alignment_gold import (
        SAME_ENTITY_PAIRS,
        DIFFERENT_ENTITY_PAIRS,
        BORDERLINE_ENTITY_PAIRS,
    )

Each entry is a dict with the following keys:

- id: unique identifier for the test pair
- name_a / name_b: entity names to compare
- content_a / content_b: descriptive text (50-150 chars) for each entity
- expected_same: whether the pair should be judged as the same entity
- difficulty: "easy", "medium", or "hard"
"""

__all__ = ["SAME_ENTITY_PAIRS", "DIFFERENT_ENTITY_PAIRS", "BORDERLINE_ENTITY_PAIRS"]

# ---------------------------------------------------------------------------
# Same-entity pairs — expected_same=True
# ---------------------------------------------------------------------------

SAME_ENTITY_PAIRS = [
    {
        "id": "person_fullname_vs_lastname_en",
        "name_a": "Alexander Fleming",
        "content_a": "Scottish bacteriologist who discovered penicillin in 1928, revolutionizing medicine and saving millions of lives worldwide.",
        "name_b": "Fleming",
        "content_b": "The bacteriologist credited with discovering the antibiotic substance penicillin, later winning the Nobel Prize in Physiology or Medicine.",
        "expected_same": True,
        "difficulty": "easy",
    },
    {
        "id": "person_fullname_vs_lastname_cn",
        "name_a": "袁隆平",
        "content_a": "中国著名农业科学家，被誉为「杂交水稻之父」，一生致力于杂交水稻技术的研究与推广。",
        "name_b": "袁隆平（杂交水稻之父）",
        "content_b": "中国农学家，研发出首批杂交水稻品种，大幅提高了全球粮食产量，获国家最高科学技术奖。",
        "expected_same": True,
        "difficulty": "easy",
    },
    {
        "id": "person_bilingual",
        "name_a": "李白",
        "content_a": "唐代伟大的浪漫主义诗人，被后人誉为「诗仙」，代表作有《将进酒》《静夜思》等千古名篇。",
        "name_b": "Li Bai",
        "content_b": "A celebrated Chinese poet of the Tang dynasty, known as the Immortal of Poetry, whose works remain among the most beloved in Chinese literature.",
        "expected_same": True,
        "difficulty": "medium",
    },
    {
        "id": "org_fullname_vs_abbr",
        "name_a": "Cloud Native Computing Foundation",
        "content_a": "A Linux Foundation project that sustains and integrates open-source technologies like Kubernetes and Prometheus for cloud-native software.",
        "name_b": "CNCF",
        "content_b": "The foundation managing cloud-native open-source projects, hosting Kubernetes, Envoy, and other critical infrastructure software under its umbrella.",
        "expected_same": True,
        "difficulty": "easy",
    },
    {
        "id": "concept_bilingual",
        "name_a": "量子霸权",
        "content_a": "指量子计算机在某个特定问题上超越最强大的经典超级计算机的计算能力的里程碑。",
        "name_b": "Quantum Supremacy",
        "content_b": "The point at which a quantum computer can solve a problem that no classical computer could solve in a feasible amount of time.",
        "expected_same": True,
        "difficulty": "medium",
    },
    {
        "id": "title_suffix",
        "name_a": "张伟教授",
        "content_a": "北京大学计算机科学系教授，主要从事人工智能与自然语言处理领域的教学和科研工作。",
        "name_b": "张伟",
        "content_b": "北京大学计算机系的学者，专注于人工智能和自然语言处理方向，发表过多篇高水平论文。",
        "expected_same": True,
        "difficulty": "medium",
    },
    {
        "id": "parenthetical_annotation",
        "name_a": "Kubernetes（K8s）",
        "content_a": "An open-source container orchestration system for automating deployment, scaling, and management of containerized applications.",
        "name_b": "Kubernetes",
        "content_b": "A production-grade container orchestration platform originally developed by Google, now maintained by the CNCF community.",
        "expected_same": True,
        "difficulty": "easy",
    },
    {
        "id": "person_name_variant",
        "name_a": "Tim Berners-Lee",
        "content_a": "English computer scientist best known as the inventor of the World Wide Web, HTML, and the HTTP protocol.",
        "name_b": "Timothy Berners-Lee",
        "content_b": "The inventor of the World Wide Web and director of the W3C, credited with transforming global information sharing.",
        "expected_same": True,
        "difficulty": "easy",
    },
    {
        "id": "place_abbreviation",
        "name_a": "北京大学",
        "content_a": "中国顶尖综合性大学，成立于1898年，位于北京市海淀区，以人文社科和理科见长。",
        "name_b": "北大",
        "content_b": "北京的一所著名高校，前身为京师大学堂，是中国近代第一所国立综合性大学。",
        "expected_same": True,
        "difficulty": "easy",
    },
    {
        "id": "product_name_variant",
        "name_a": "ChatGPT",
        "content_a": "An AI chatbot developed by OpenAI based on large language models, capable of conversing fluently and assisting with diverse tasks.",
        "name_b": "ChatGPT-4",
        "content_b": "OpenAI's conversational AI assistant powered by GPT-4, providing advanced reasoning and natural language understanding capabilities.",
        "expected_same": True,
        "difficulty": "medium",
    },
    {
        "id": "concept_expanded",
        "name_a": "AI",
        "content_a": "The field of computer science focused on creating systems capable of performing tasks that normally require human intelligence.",
        "name_b": "人工智能",
        "content_b": "计算机科学的一个分支，致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法和技术。",
        "expected_same": True,
        "difficulty": "medium",
    },
    {
        "id": "person_with_role",
        "name_a": "马云",
        "content_a": "中国企业家，阿里巴巴集团联合创始人，曾是亚洲首富，致力于教育公益事业。",
        "name_b": "马云（阿里巴巴创始人）",
        "content_b": "阿里巴巴集团的缔造者，中国互联网经济的标志性人物，推动了中国电子商务的蓬勃发展。",
        "expected_same": True,
        "difficulty": "easy",
    },
    {
        "id": "tech_with_version",
        "name_a": "Python",
        "content_a": "A high-level, general-purpose programming language known for its clean syntax, dynamic typing, and extensive standard library.",
        "name_b": "Python 3",
        "content_b": "The latest major version of the Python programming language, introducing Unicode strings, async/await, and many modern features.",
        "expected_same": True,
        "difficulty": "easy",
    },
    {
        "id": "historical_name",
        "name_a": "贞观之治",
        "content_a": "唐太宗李世民在位期间的太平盛世，政治清明、经济繁荣、社会安定，被誉为中国历史上的黄金时代。",
        "name_b": "贞观",
        "content_b": "唐太宗年号，这一时期推行开明政策，虚心纳谏，国力空前强盛，开创了盛唐基业。",
        "expected_same": True,
        "difficulty": "medium",
    },
    {
        "id": "org_bilingual",
        "name_a": "谷歌",
        "content_a": "全球最大的互联网搜索引擎公司，提供搜索、云计算、在线广告等核心产品与服务。",
        "name_b": "Google",
        "content_b": "An American multinational technology company specializing in Internet-related services, search, cloud computing, and software.",
        "expected_same": True,
        "difficulty": "easy",
    },
]

# ---------------------------------------------------------------------------
# Different-entity pairs — expected_same=False
# ---------------------------------------------------------------------------

DIFFERENT_ENTITY_PAIRS = [
    {
        "id": "author_vs_work",
        "name_a": "曹雪芹",
        "content_a": "清代著名文学家，出身于没落的贵族家庭，以毕生精力创作了不朽名著《红楼梦》。",
        "name_b": "红楼梦",
        "content_b": "中国古典四大名著之一，以贾宝玉和林黛玉的爱情悲剧为主线，描绘了一个封建大家族的兴衰。",
        "expected_same": False,
        "difficulty": "easy",
    },
    {
        "id": "person_vs_event",
        "name_a": "曹操",
        "content_a": "东汉末年杰出的政治家、军事家、文学家，曹魏政权的奠基人，以「挟天子以令诸侯」闻名。",
        "name_b": "官渡之战",
        "content_b": "东汉建安五年曹操与袁绍之间的决定性战役，曹操以少胜多，奠定了统一北方的基础。",
        "expected_same": False,
        "difficulty": "easy",
    },
    {
        "id": "city_vs_province",
        "name_a": "长沙",
        "content_a": "湖南省省会城市，位于湘江下游，是长江中游地区重要的中心城市，以美食和娱乐产业著称。",
        "name_b": "湖南",
        "content_b": "中国中部省份，简称湘，省会长沙，以丘陵地貌和湘江水系为特征，盛产稻米和茶叶。",
        "expected_same": False,
        "difficulty": "easy",
    },
    {
        "id": "company_vs_product",
        "name_a": "Google",
        "content_a": "An American multinational technology conglomerate specializing in search, advertising, cloud computing, and AI research.",
        "name_b": "Android",
        "content_b": "A mobile operating system based on the Linux kernel, designed primarily for touchscreen devices like smartphones and tablets.",
        "expected_same": False,
        "difficulty": "easy",
    },
    {
        "id": "similar_different_people",
        "name_a": "Howard Florey",
        "content_a": "Australian pharmacologist who led the team that purified penicillin and demonstrated its clinical effectiveness as an antibiotic.",
        "name_b": "Ernst Chain",
        "content_b": "German-born British biochemist who worked alongside Florey to isolate and purify penicillin, sharing the Nobel Prize for the work.",
        "expected_same": False,
        "difficulty": "medium",
    },
    {
        "id": "parent_vs_subsidiary",
        "name_a": "阿里巴巴",
        "content_a": "中国最大的电子商务集团，旗下拥有淘宝、天猫、阿里云等多个业务板块，由马云创立。",
        "name_b": "淘宝网",
        "content_b": "阿里巴巴集团旗下的C2C在线购物平台，是中国最大的网络零售和消费者交易平台之一。",
        "expected_same": False,
        "difficulty": "medium",
    },
    {
        "id": "concept_vs_implementation",
        "name_a": "量子计算",
        "content_a": "利用量子力学原理（叠加和纠缠）进行信息处理的新型计算范式，有望解决经典计算机无法高效处理的问题。",
        "name_b": "Sycamore",
        "content_b": "Google研发的53量子比特超导量子处理器，2019年曾实现量子优越性实验，完成经典超算难以模拟的计算任务。",
        "expected_same": False,
        "difficulty": "medium",
    },
    {
        "id": "person_vs_book",
        "name_a": "孔子",
        "content_a": "春秋末期思想家、教育家，儒家学派创始人，提倡仁义礼智信，对中国文化影响深远。",
        "name_b": "论语",
        "content_b": "儒家经典著作，记录了孔子及其弟子的言行，是研究孔子思想和儒家学说最重要的文献。",
        "expected_same": False,
        "difficulty": "easy",
    },
    {
        "id": "language_vs_framework",
        "name_a": "Python",
        "content_a": "A versatile, high-level programming language emphasizing code readability, with a rich ecosystem of third-party libraries.",
        "name_b": "Django",
        "content_b": "A high-level Python web framework that encourages rapid development and clean, pragmatic design of database-driven websites.",
        "expected_same": False,
        "difficulty": "easy",
    },
    {
        "id": "university_different",
        "name_a": "北京大学",
        "content_a": "中国顶尖综合性大学，创立于1898年，以人文社科和基础科学见长，位于北京市海淀区。",
        "name_b": "清华大学",
        "content_b": "中国顶尖研究型大学，创建于1911年，以工科和理科优势闻名，同样位于北京市海淀区。",
        "expected_same": False,
        "difficulty": "easy",
    },
    {
        "id": "person_same_name_diff_org",
        "name_a": "张伟（北京大学教授）",
        "content_a": "北京大学计算机科学系教授，研究方向为自然语言处理，发表多篇顶级会议论文。",
        "name_b": "张伟（清华大学教授）",
        "content_b": "清华大学电子工程系教授，主要从事信号处理与通信系统研究，承担国家重点研发项目。",
        "expected_same": False,
        "difficulty": "medium",
    },
    {
        "id": "country_vs_capital",
        "name_a": "中国",
        "content_a": "位于东亚的国家，拥有五千年文明史，是世界人口大国和第二大经济体，首都为北京。",
        "name_b": "北京",
        "content_b": "中国首都，全国政治、文化和国际交流中心，拥有故宫、长城等世界文化遗产。",
        "expected_same": False,
        "difficulty": "easy",
    },
    {
        "id": "hardware_vs_software",
        "name_a": "NVIDIA",
        "content_a": "An American technology company that designs GPUs for gaming, professional visualization, and AI training and inference.",
        "name_b": "CUDA",
        "content_b": "A parallel computing platform and programming model developed by NVIDIA for general-purpose GPU computing.",
        "expected_same": False,
        "difficulty": "medium",
    },
    {
        "id": "event_vs_period",
        "name_a": "安史之乱",
        "content_a": "唐朝中期由安禄山和史思明发动的叛乱，历时八年，使唐朝由盛转衰，是中国历史的重大转折点。",
        "name_b": "唐朝",
        "content_b": "中国历史上的大一统王朝（618-907年），以开放繁荣著称，诗歌、艺术和科技成就达到顶峰。",
        "expected_same": False,
        "difficulty": "easy",
    },
    {
        "id": "discipline_vs_application",
        "name_a": "人工智能",
        "content_a": "计算机科学的核心分支，致力于构建能够感知、推理、学习和决策的智能系统，涵盖多个子领域。",
        "name_b": "机器学习",
        "content_b": "人工智能的重要子领域，研究如何让计算机从数据中自动学习规律并做出预测，包括监督学习、无监督学习等方法。",
        "expected_same": False,
        "difficulty": "medium",
    },
]

# ---------------------------------------------------------------------------
# Borderline pairs — intentionally ambiguous
# ---------------------------------------------------------------------------

BORDERLINE_ENTITY_PAIRS = [
    {
        "id": "apple_company_vs_fruit",
        "name_a": "苹果公司",
        "content_a": "美国跨国科技企业，由乔布斯创立，主要产品包括iPhone、Mac和iPad，市值居全球前列。",
        "name_b": "苹果",
        "content_b": "蔷薇科苹果属植物的果实，是全球广泛种植和食用的水果，营养丰富，含有多种维生素和膳食纤维。",
        "expected_same": False,
        "difficulty": "hard",
    },
    {
        "id": "ai_vs_ml",
        "name_a": "人工智能",
        "content_a": "研究如何使计算机模拟人类智能行为的学科，涵盖感知、推理、学习、决策等多个研究方向。",
        "name_b": "机器学习",
        "content_b": "人工智能的核心方法之一，通过算法让计算机从数据中提取模式和规律，实现预测和分类。",
        "expected_same": False,
        "difficulty": "hard",
    },
    {
        "id": "google_vs_google_cloud",
        "name_a": "Google",
        "content_a": "An American multinational technology conglomerate offering search, advertising, cloud computing, and consumer electronics.",
        "name_b": "Google Cloud",
        "content_b": "A suite of cloud computing services by Google, providing infrastructure, data analytics, and machine learning tools to enterprises.",
        "expected_same": False,
        "difficulty": "hard",
    },
    {
        "id": "java_language_vs_island",
        "name_a": "Java",
        "content_a": "A general-purpose, object-oriented programming language designed for cross-platform compatibility and enterprise software.",
        "name_b": "Java",
        "content_b": "An island in Indonesia, the world's most populous island, home to the capital Jakarta and over 150 million residents.",
        "expected_same": False,
        "difficulty": "hard",
    },
    {
        "id": "quantum_computing_vs_quantum_supremacy",
        "name_a": "量子计算",
        "content_a": "利用量子力学原理进行信息处理的新型计算技术，研究领域涵盖量子算法、纠错和硬件实现。",
        "name_b": "量子霸权",
        "content_b": "量子计算超越经典计算能力的里程碑，谷歌的Sycamore处理器在2019年首次宣称实现了这一目标。",
        "expected_same": False,
        "difficulty": "hard",
    },
    {
        "id": "tang_dynasty_vs_tang_poetry",
        "name_a": "唐朝",
        "content_a": "中国历史上的大一统王朝（618-907年），国力强盛，文化繁荣，与各国交流频繁。",
        "name_b": "唐诗",
        "content_b": "唐代创作的诗歌总称，是中国古典诗歌的巅峰，涌现出李白、杜甫、白居易等伟大诗人。",
        "expected_same": False,
        "difficulty": "hard",
    },
    {
        "id": "concept_bilingual_hard",
        "name_a": "深度学习",
        "content_a": "机器学习的一个分支，使用多层神经网络从大量数据中自动学习特征表示，在图像和语音识别中表现优异。",
        "name_b": "Deep Learning",
        "content_b": "A subset of machine learning using deep neural networks with multiple layers to automatically learn data representations.",
        "expected_same": True,
        "difficulty": "hard",
    },
    {
        "id": "k8s_vs_kubernetes",
        "name_a": "K8s",
        "content_a": "An open-source container orchestration platform for automating deployment and scaling of containerized applications.",
        "name_b": "Kubernetes",
        "content_b": "A production-grade container orchestration system originally developed by Google and now maintained by CNCF.",
        "expected_same": True,
        "difficulty": "easy",
    },
]
