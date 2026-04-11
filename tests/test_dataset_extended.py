"""
Extended test dataset: 30+ cases across 4 dimensions.

Dimensions:
  1. Length: short (<100 chars), medium (100-300), long (300-600), very_long (600+)
  2. Domain: technology, history, science, literature, medicine, finance, philosophy, geography, sports, art
  3. Language: cn (Chinese), en (English), mixed (Chinese+English)
  4. Type: narrative, dialogue, list, technical, academic, news, personal, poetic, legal, FAQ

Each entry specifies which dimensions it covers.
"""

__all__ = ['EXTENDED_DATASETS']

EXTENDED_DATASETS = {
    # ─── Dimension 1: Length extremes ──────────────────────────────────
    "short_1": {
        "text": "李白是唐代著名诗人，字太白，号青莲居士。他擅长浪漫主义诗歌创作，被称为\"诗仙\"。",
        "length": "short",
        "domain": "literature",
        "language": "cn",
        "type": "narrative",
        "expected_entities": ["李白", "唐代"],
        "expected_relations_min": 1,
    },
    "short_2_en": {
        "text": "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability and supports multiple programming paradigms.",
        "length": "short",
        "domain": "technology",
        "language": "en",
        "type": "narrative",
        "expected_entities": ["Python", "Guido van Rossum"],
        "expected_relations_min": 1,
    },
    "medium_1_medicine": {
        "text": (
            "青霉素（Penicillin）是世界上第一种广泛使用的抗生素，由英国细菌学家Alexander Fleming"
            "于1928年发现。Fleming在圣玛丽医院的实验室中偶然发现，培养葡萄球菌的培养基上"
            "长出了一团青绿色的霉菌，霉菌周围的细菌被杀死了。经过进一步研究，他确认这种霉菌"
            "属于青霉菌属（Penicillium），并分泌出一种能抑制细菌生长的物质，他将其命名为青霉素。"
            "1940年，Howard Florey和Ernst Chain成功提纯了青霉素并实现了大规模生产，使之在"
            "第二次世界大战中挽救了无数伤兵的生命。Fleming、Florey和Chain因此共同获得了"
            "1945年诺贝尔生理学或医学奖。青霉素的发现标志着抗生素时代的开始，彻底改变了"
            "现代医学的治疗方式，使得许多曾经致命的细菌感染变得可以治愈。"
        ),
        "length": "medium",
        "domain": "medicine",
        "language": "cn",
        "type": "narrative",
        "expected_entities": ["青霉素", "Alexander Fleming", "Howard Florey", "Ernst Chain"],
        "expected_relations_min": 3,
    },
    "medium_2_finance": {
        "text": (
            "The 2008 financial crisis, also known as the Global Financial Crisis (GFC), was the most severe"
            " economic disaster since the Great Depression of the 1930s. It was triggered by the collapse of"
            " the housing bubble in the United States, which led to a crisis in the subprime mortgage market."
            " Lehman Brothers, one of the largest investment banks in the US, filed for bankruptcy on September"
            " 15, 2008, marking the largest bankruptcy filing in US history. The crisis quickly spread globally,"
            " affecting financial institutions worldwide. The US government responded with the Troubled Asset"
            " Relief Program (TARP), authorizing $700 billion to purchase toxic assets and inject capital into"
            " banks. Federal Reserve Chairman Ben Bernanke played a key role in managing the crisis response."
            " The crisis led to the passage of the Dodd-Frank Wall Street Reform Act in 2010, which imposed"
            " stricter regulations on financial institutions."
        ),
        "length": "medium",
        "domain": "finance",
        "language": "en",
        "type": "narrative",
        "expected_entities": ["2008 financial crisis", "Lehman Brothers", "Ben Bernanke"],
        "expected_relations_min": 2,
    },
    "very_long_1_philosophy": {
        "text": (
            "儒家思想是中国古代最重要的哲学体系之一，由孔子（公元前551年—公元前479年）创立。"
            "孔子名丘，字仲尼，出生于鲁国陬邑（今山东省曲阜市），是中国历史上最具影响力的思想家、"
            "教育家。他提倡\"仁\"、\"义\"、\"礼\"、\"智\"、\"信\"五种核心德目，强调\"仁者爱人\"的道德理念，"
            "主张以德治国、以礼教化。孔子创办私学，打破了贵族对教育的垄断，提出\"有教无类\"的教育思想，"
            "据说有弟子三千人，其中贤人七十二。他的言行被弟子整理成《论语》一书，成为儒家经典的核心文献。\n\n"
            "孔子之后，儒家思想经过孟子（约前372年—前289年）和荀子（约前313年—前238年）的发展，"
            "形成了不同的分支。孟子主张\"性善论\"，认为人天生具有仁、义、礼、智四种善端的萌芽，"
            "只要加以培养就能成为完整的道德品质。他提出了\"民为贵，社稷次之，君为轻\"的民本思想，"
            "和\"富贵不能淫，贫贱不能移，威武不能屈\"的大丈夫理想。荀子则主张\"性恶论\"，认为人的"
            "本性趋利避害，需要通过后天的教育和礼法来约束和改造。荀子的学生韩非子和李斯后来成为"
            "法家的代表人物，韩非子集法家之大成，提出了\"法、术、势\"三者结合的治国理论，深刻影响"
            "了秦始皇统一六国的政治实践。\n\n"
            "汉代，董仲舒向汉武帝提出\"罢黜百家，独尊儒术\"的建议并被采纳，儒家思想从此成为中国的"
            "官方意识形态，统治中国思想界长达两千多年。宋明理学是儒学发展的重要阶段。北宋的周敦颐、"
            "程颢、程颐和南宋的朱熹将儒学与佛道思想融合，建立了以\"天理\"为核心的理学体系。朱熹的"
            "《四书章句集注》成为科举考试的标准教材，其影响一直延续到清末。明代王阳明则发展了"
            "\"心学\"，主张\"知行合一\"和\"致良知\"，认为知识和行动不可分离，良知是先天的道德直觉，"
            "只需要去蔽明心就能达到圣贤境界。王阳明的心学与朱熹的理学形成了宋明儒学中最重要的"
            "两大流派，后世尊称朱熹和程颢、程颐为代表的学派为\"程朱理学\"，王阳明为代表的为"
            "\"陆王心学\"（陆指南宋陆九渊）。\n\n"
            "儒学的影响远超中国国界。它传播到朝鲜半岛、日本、越南等东亚国家，形成了\"儒学文化圈\"。"
            "在朝鲜王朝时期，朱子学成为官方意识形态。在日本，江户时代的朱子学派和阳明学派都有"
            "深远影响，明治维新时期的许多政治家都深受儒学教育。在现代，儒学的核心价值观——仁爱、"
            "诚信、和谐、孝道——仍然深刻影响着东亚社会的伦理观念和人际关系。新加坡建国总理李光耀"
            "曾大力倡导\"亚洲价值观\"，其核心正是儒家文化中的家庭伦理和社会秩序观念。"
        ),
        "length": "very_long",
        "domain": "philosophy",
        "language": "cn",
        "type": "narrative",
        "expected_entities": ["孔子", "孟子", "荀子", "朱熹", "王阳明", "董仲舒", "韩非子"],
        "expected_relations_min": 8,
    },

    # ─── Dimension 2: Domain diversity ──────────────────────────────────
    "geography_short": {
        "text": (
            "尼罗河是世界上最长的河流，全长约6650公里，流经11个非洲国家。它发源于东非高原的"
            "维多利亚湖，最终注入地中海。古埃及文明就诞生在尼罗河沿岸，河水定期泛滥带来了肥沃"
            "的冲积土壤，使得农业得以发展。埃及首都开罗就位于尼罗河三角洲的顶端。"
        ),
        "length": "medium",
        "domain": "geography",
        "language": "cn",
        "type": "narrative",
        "expected_entities": ["尼罗河", "维多利亚湖", "地中海", "开罗", "古埃及文明"],
        "expected_relations_min": 2,
    },
    "sports_medium": {
        "text": (
            "梅西（Lionel Messi），1987年6月24日出生于阿根廷罗萨里奥，是阿根廷职业足球运动员，"
            "被广泛认为是足球历史上最伟大的球员之一。梅西从小就展现出过人的足球天赋，但11岁时"
            "被诊断出患有生长激素缺乏症。巴塞罗那俱乐部看中了他的才华，同意承担他的治疗费用，"
            "梅西因此移居西班牙加入拉玛西亚青训营。2004年，年仅17岁的梅西首次代表巴塞罗那一线队"
            "出场。此后他在巴萨效力了21个赛季，出场778次，打入672球，助攻305次，创造了单一俱乐部"
            "最多进球的纪录。他帮助巴萨获得了10个西甲冠军、4个欧冠冠军和7个国王杯冠军。"
            "个人荣誉方面，梅西获得了8次金球奖（Ballon d'Or），是历史上获得该奖项最多的球员。"
            "2021年，梅西离开巴萨加盟巴黎圣日耳曼（PSG），2023年又转会至美国迈阿密国际队（Inter Miami）。"
            "在国家队层面，梅西率领阿根廷队获得了2021年美洲杯冠军和2022年卡塔尔世界杯冠军，"
            "终于在职业生涯晚期圆了世界杯梦想，被阿根廷人视为民族英雄。他的竞争对手克里斯蒂亚诺·"
            "罗纳尔多（Cristiano Ronaldo）同样伟大，两人在长达15年的时间里共同统治了世界足坛，"
            "被称为\"绝代双骄\"。"
        ),
        "length": "long",
        "domain": "sports",
        "language": "mixed",
        "type": "narrative",
        "expected_entities": ["梅西", "巴塞罗那", "金球奖", "C罗", "阿根廷"],
        "expected_relations_min": 5,
    },
    "art_short": {
        "text": (
            "梵高（Vincent van Gogh，1853-1890）是荷兰后印象派画家，代表作有《星夜》、《向日葵》"
            "和《麦田群鸦》。他一生饱受精神疾病困扰，37岁时在法国瓦兹河畔奥维尔自杀身亡。"
            "尽管生前几乎无人问津，他的作品在死后却成为世界上最著名、最昂贵的画作之一。"
            "梵高用浓烈的色彩和粗犷的笔触表达内心的情感，对20世纪的表现主义产生了深远影响。"
        ),
        "length": "medium",
        "domain": "art",
        "language": "mixed",
        "type": "narrative",
        "expected_entities": ["梵高", "星夜", "向日葵"],
        "expected_relations_min": 2,
    },

    # ─── Dimension 3: Language diversity ────────────────────────────────
    "english_tech": {
        "text": (
            "Rust is a systems programming language sponsored by Mozilla Research, designed for performance"
            " and safety, especially safe concurrency. Rust syntactically resembles C++ but provides memory"
            " safety without garbage collection by using a borrow checker to validate references. Graydon Hoare"
            " started Rust as a personal project in 2006, and Mozilla began sponsoring it in 2009. Rust 1.0"
            " was released in 2015. The language has been voted the 'most loved programming language' in the"
            " Stack Overflow Developer Survey every year since 2016. It is used by companies like Microsoft,"
            " Google, Amazon, and Facebook for systems-level programming. The Linux kernel officially added"
            " Rust support in 2022, making it the second language allowed in kernel development after C."
            " Rust's ownership model, lifetimes, and trait system make it uniquely suited for building"
            " reliable and efficient software."
        ),
        "length": "long",
        "domain": "technology",
        "language": "en",
        "type": "technical",
        "expected_entities": ["Rust", "Mozilla", "Graydon Hoare", "Linux"],
        "expected_relations_min": 3,
    },
    "mixed_cn_en": {
        "text": (
            "OpenAI是一家总部位于San Francisco的AI研究公司，由Sam Altman担任CEO。"
            "2022年11月，OpenAI发布了ChatGPT，这款基于GPT-3.5的对话AI在短短两个月内获得了"
            "超过1亿用户，成为历史上增长最快的消费级应用。GPT-4于2023年3月发布，具备多模态能力，"
            "能够理解图片和文本。Microsoft是OpenAI最大的投资者，投资总额超过130亿美元，"
            "并将GPT模型集成到Bing搜索和Microsoft 365 Copilot中。"
            "在中国市场，百度推出了文心一言（ERNIE Bot），阿里巴巴推出了通义千问，"
            "字节跳动推出了豆包AI，Z.ai推出了GLM系列模型。这些公司之间的竞争推动了"
            "大语言模型技术的快速发展。2024年，Anthropic公司推出的Claude 3系列模型在多项"
            "基准测试中超越了GPT-4，而Google的Gemini模型也展示了强大的多模态推理能力。"
        ),
        "length": "long",
        "domain": "technology",
        "language": "mixed",
        "type": "narrative",
        "expected_entities": ["OpenAI", "Sam Altman", "ChatGPT", "Microsoft", "Google", "Anthropic"],
        "expected_relations_min": 4,
    },

    # ─── Dimension 4: Content type diversity ────────────────────────────
    "dialogue_1": {
        "text": (
            "张教授：李明，你对量子力学的基本原理了解多少？\n"
            "李明：张教授，我知道薛定谔方程是量子力学的核心方程，但我对测量问题还不太理解。\n"
            "张教授：很好。测量问题确实是量子力学最有争议的部分。哥本哈根诠释认为，测量行为"
            "会导致波函数坍缩。但多世界诠释的提出者Hugh Everett认为，每次测量不会导致坍缩，"
            "而是宇宙分裂成多个分支。\n"
            "李明：这听起来很不可思议。那费曼的路径积分方法又是怎么回事？\n"
            "张教授：Richard Feynman提出，粒子从A点到B点的运动，实际上是同时经过所有可能的路径，"
            "最终的观测结果是所有路径的概率振幅之和。这和经典物理中粒子走唯一确定路径的概念"
            "完全不同。费曼因此和Julian Schwinger、朝永振一郎共同获得了1965年诺贝尔物理学奖。"
        ),
        "length": "medium",
        "domain": "science",
        "language": "mixed",
        "type": "dialogue",
        "expected_entities": ["薛定谔方程", "Feynman", "Hugh Everett", "哥本哈根诠释"],
        "expected_relations_min": 2,
    },
    "list_1": {
        "text": (
            "2024年全球市值最高的科技公司排名：\n"
            "1. 苹果（Apple Inc.）— 市值约3.4万亿美元，总部位于加州库比蒂诺，CEO为Tim Cook，"
            "主要产品包括iPhone、MacBook、iPad和Apple Watch\n"
            "2. 微软（Microsoft）— 市值约3.1万亿美元，总部位于华盛顿州雷德蒙德，CEO为Satya Nadella，"
            "主要业务包括Azure云服务、Office 365和Windows操作系统\n"
            "3. 英伟达（NVIDIA）— 市值约2.8万亿美元，总部位于加州圣克拉拉，CEO为黄仁勋（Jensen Huang），"
            "主要产品为GPU芯片，在AI训练和推理领域占据主导地位\n"
            "4. Alphabet（Google母公司）— 市值约2.2万亿美元，CEO为Sundar Pichai，"
            "主要业务包括Google搜索、YouTube、Google Cloud和Android\n"
            "5. 亚马逊（Amazon）— 市值约1.9万亿美元，CEO为Andy Jassy，"
            "主要业务包括AWS云服务、电子商务和Prime会员\n"
            "这五家公司被统称为美国科技\"五巨头\"（Big Five），总市值超过13万亿美元。"
        ),
        "length": "long",
        "domain": "business",
        "language": "mixed",
        "type": "list",
        "expected_entities": ["苹果", "微软", "英伟达", "Alphabet", "亚马逊", "Tim Cook", "黄仁勋"],
        "expected_relations_min": 5,
    },
    "academic_1": {
        "text": (
            "Abstract: This paper presents a novel approach to neural machine translation (NMT) based on the"
            " Transformer architecture proposed by Vaswani et al. (2017). Unlike previous recurrent neural"
            " network (RNN) based approaches, the Transformer relies entirely on self-attention mechanisms,"
            " dispensing with recurrence and convolutions entirely. Our experiments on the WMT 2014"
            " English-to-German translation task demonstrate that the proposed model achieves a BLEU score"
            " of 28.4, outperforming existing state-of-the-art models. The self-attention mechanism allows"
            " the model to capture long-range dependencies more effectively than LSTM-based models proposed"
            " by Hochreiter and Schmidhuber (1997). We also show that the model is significantly more"
            " parallelizable than RNN-based approaches, reducing training time from days to hours on"
            " modern GPU hardware. The key innovation is the multi-head attention mechanism, which allows"
            " the model to jointly attend to information from different representation subspaces at different"
            " positions. This work builds upon the attention mechanism first introduced by Bahdanau et al."
            " (2014) for machine translation and extends it to a fully attention-based architecture."
        ),
        "length": "long",
        "domain": "science",
        "language": "en",
        "type": "academic",
        "expected_entities": ["Transformer", "Vaswani", "BLEU", "LSTM", "self-attention"],
        "expected_relations_min": 3,
    },
    "legal_1": {
        "text": (
            "《中华人民共和国数据安全法》于2021年6月10日由第十三届全国人民代表大会常务委员会"
            "第二十九次会议通过，自2021年9月1日起施行。该法共七章五十五条，确立了数据安全"
            "管理的基本制度框架。其中第四条规定，维护数据安全应当坚持总体国家安全观，"
            "建立健全数据安全治理体系。第十二条要求国家建立数据分类分级保护制度，根据数据在"
            "经济社会发展中的重要程度，以及一旦遭到篡改、破坏、泄露或者非法获取、非法利用，"
            "对国家安全、公共利益或者个人、组织合法权益造成的危害程度，对数据实行分类分级保护。"
            "第三十六条明确规定了向境外提供数据的条件：非关键信息基础设施运营者在中华人民共和国"
            "境内收集和产生的数据，向境外提供前应当进行数据安全评估。国家网信部门负责统筹协调"
            "数据安全评估工作。违反该法第四十六条规定，向境外提供重要数据且未进行安全评估的，"
            "由有关主管部门责令改正，给予警告，可以并处十万元以上一百万元以下罚款。"
        ),
        "length": "long",
        "domain": "law",
        "language": "cn",
        "type": "legal",
        "expected_entities": ["数据安全法", "全国人大常委会", "数据分类分级", "网信部门"],
        "expected_relations_min": 2,
    },
    "faq_1": {
        "text": (
            "关于深度学习的常见问题解答：\n\n"
            "Q1: 什么是深度学习？\n"
            "A: 深度学习是机器学习的一个子领域，使用多层神经网络来自动学习数据的层次化表征。"
            "Yann LeCun、Geoffrey Hinton和Yoshua Bengio被认为是深度学习的三位先驱，"
            "他们共同获得了2018年图灵奖。\n\n"
            "Q2: 深度学习和传统机器学习有什么区别？\n"
            "A: 传统机器学习需要人工设计特征（feature engineering），而深度学习通过多层网络"
            "自动学习特征表示。例如在图像识别中，传统方法需要设计SIFT或HOG特征，"
            "而卷积神经网络（CNN）可以直接从原始像素学习有效的特征。\n\n"
            "Q3: 什么是Transformer模型？\n"
            "A: Transformer是2017年由Google Brain团队提出的基于自注意力机制的模型架构，"
            "BERT和GPT都是基于Transformer的模型。BERT使用双向编码器，适合理解类任务；"
            "GPT使用单向解码器，适合生成类任务。\n\n"
            "Q4: 什么是大语言模型（LLM）？\n"
            "A: 大语言模型是指参数量在数十亿到数千亿之间的Transformer模型，通过在海量文本上"
            "预训练获得广泛的知识和语言理解能力。代表性模型包括OpenAI的GPT-4、"
            "Google的PaLM和Meta的LLaMA。"
        ),
        "length": "long",
        "domain": "technology",
        "language": "mixed",
        "type": "faq",
        "expected_entities": ["深度学习", "Yann LeCun", "Geoffrey Hinton", "Transformer", "BERT", "GPT"],
        "expected_relations_min": 4,
    },
    "personal_diary": {
        "text": (
            "今天是我来北京工作的第一天。早上从北京南站坐地铁4号线到海淀黄庄，"
            "然后步行到公司。公司在中关村软件园，周围全是科技公司——百度、联想、"
            "快手都在附近。中午和同事小王一起吃了食堂，他告诉我公司用的是Go语言"
            "做后端，前端用React。下午配置开发环境，安装了VS Code和Docker。"
            "mentor叫刘工，他说项目用微服务架构，部署在Kubernetes上，"
            "数据库用的是PostgreSQL和Redis。下班后去五道口逛了逛，"
            "发现一家很好的咖啡店叫\"雕刻时光\"。明天要开始写第一个需求了。"
        ),
        "length": "medium",
        "domain": "personal",
        "language": "cn",
        "type": "personal",
        "expected_entities": ["北京", "中关村软件园", "Go语言", "React", "Kubernetes", "PostgreSQL"],
        "expected_relations_min": 2,
    },
    "poetic_1": {
        "text": (
            "《水调歌头·明月几时有》——苏轼\n\n"
            "明月几时有？把酒问青天。不知天上宫阙，今夕是何年。"
            "我欲乘风归去，又恐琼楼玉宇，高处不胜寒。起舞弄清影，何似在人间。\n\n"
            "转朱阁，低绮户，照无眠。不应有恨，何事长向别时圆？"
            "人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。\n\n"
            "这首词是苏轼在宋神宗熙宁九年（1076年）中秋节时，在密州（今山东诸城）"
            "任知州时所作。当时苏轼因为反对王安石变法而自请外调，与弟弟苏辙已分别七年未见。"
            "词中\"但愿人长久，千里共婵娟\"成为千古名句，表达了对远方亲人的美好祝愿。"
            "苏轼是北宋文学家、书法家和画家，与其父苏洵、弟弟苏辙合称\"三苏\"，"
            "同列\"唐宋八大家\"之中。"
        ),
        "length": "medium",
        "domain": "literature",
        "language": "cn",
        "type": "poetic",
        "expected_entities": ["苏轼", "苏辙", "王安石", "水调歌头"],
        "expected_relations_min": 2,
    },
    "news_1": {
        "text": (
            "新华社北京2026年3月15日电 — 中国科学院国家天文台今日宣布，由该台主导建设的"
            "FAST射电望远镜（\"中国天眼\"）在银河系外发现了一个新的快速射电暴（FRB）源，"
            "编号FRB 20260312A。该信号来自距地球约30亿光年的一个矮星系，在短短两个月内"
            "被探测到重复爆发超过200次。研究团队负责人、国家天文台首席研究员李菂表示，"
            "这是FAST自2020年探测到首个新FRB以来的又一重大发现，为理解FRB的物理机制"
            "提供了重要线索。该发现已发表在《自然·天文学》期刊上。FAST位于贵州省黔南州"
            "平塘县，口径500米，是目前世界上最大单口径射电望远镜，于2016年落成，"
            "由南仁东担任首席科学家兼总工程师主持设计建设。南仁东于2017年逝世，"
            "但其科学遗产仍在持续产生重要发现。"
        ),
        "length": "long",
        "domain": "science",
        "language": "cn",
        "type": "news",
        "expected_entities": ["FAST", "FRB", "国家天文台", "李菂", "南仁东", "贵州"],
        "expected_relations_min": 3,
    },
    "extreme_short": {
        "text": "爱因斯坦提出了相对论，E=mc²。",
        "length": "short",
        "domain": "science",
        "language": "mixed",
        "type": "narrative",
        "expected_entities": ["爱因斯坦", "相对论"],
        "expected_relations_min": 1,
    },
    "pure_english_long": {
        "text": (
            "The Renaissance was a cultural movement that profoundly affected European intellectual life"
            " in the early modern period. Beginning in Italy, particularly in Florence, in the 14th century,"
            " it spread to the rest of Europe by the 16th century. The Renaissance is considered the bridge"
            " between the Middle Ages and modern civilization. Key figures include Leonardo da Vinci,"
            " who painted the Mona Lisa and designed flying machines centuries before the Wright brothers"
            " achieved powered flight; Michelangelo, who sculpted David and painted the Sistine Chapel ceiling;"
            " and Raphael, whose School of Athens fresco represents the pinnacle of High Renaissance art.\n\n"
            "The Medici family, particularly Lorenzo de' Medici (known as 'the Magnificent'), were crucial"
            " patrons of Renaissance art and learning. Their bank, the Medici Bank, was the most powerful"
            " financial institution in 15th-century Europe. The invention of the printing press by Johannes"
            " Gutenberg around 1440 revolutionized the dissemination of knowledge, making books affordable"
            " and contributing to the Protestant Reformation launched by Martin Luther in 1517.\n\n"
            "Niccolo Machiavelli, a Florentine diplomat and philosopher, wrote The Prince in 1513,"
            " a seminal work of political philosophy that introduced the concept that 'the ends justify"
            " the means.' Galileo Galilei, born in Pisa in 1564, made groundbreaking contributions to"
            " astronomy, physics, and the scientific method. His support for the heliocentric model proposed"
            " by Copernicus brought him into conflict with the Catholic Church. The Renaissance also saw"
            " advances in navigation, leading to the Age of Discovery. Christopher Columbus reached the"
            " Americas in 1492 under the Spanish flag, and Vasco da Gama sailed around Africa to India in 1498,"
            " opening direct sea trade between Europe and Asia."
        ),
        "length": "very_long",
        "domain": "history",
        "language": "en",
        "type": "narrative",
        "expected_entities": ["Leonardo da Vinci", "Michelangelo", "Medici", "Gutenberg", "Galileo", "Columbus"],
        "expected_relations_min": 5,
    },
    "technical_spec": {
        "text": (
            "系统架构设计规范 v2.1\n\n"
            "一、总体架构\n"
            "本系统采用微服务架构，主要包含以下组件：API Gateway（Kong）、服务注册与发现"
            "（Consul）、配置中心、消息队列（Apache Kafka）、缓存层（Redis Cluster）、"
            "数据库（PostgreSQL主从 + Elasticsearch全文检索）和对象存储（MinIO）。\n\n"
            "二、核心服务\n"
            "1. 用户服务（user-service）：负责用户注册、认证（JWT）、权限管理（RBAC模型）\n"
            "2. 订单服务（order-service）：处理订单创建、支付（集成支付宝和微信支付）、"
            "退款和物流追踪\n"
            "3. 商品服务（product-service）：商品管理、库存控制（分布式锁）、价格引擎\n"
            "4. 搜索服务（search-service）：基于Elasticsearch的商品搜索和推荐\n"
            "5. 通知服务（notification-service）：短信（阿里云SMS）、邮件（SMTP）、"
            "App推送（Firebase Cloud Messaging）\n\n"
            "三、基础设施\n"
            "所有服务容器化部署在Kubernetes集群上，使用Helm Chart管理配置。CI/CD流程"
            "使用GitLab CI，代码提交后自动触发单元测试、集成测试和Docker镜像构建。"
            "监控使用Prometheus + Grafana组合，日志收集使用ELK Stack（Elasticsearch + "
            "Logstash + Kibana）。链路追踪使用Jaeger，实现了OpenTelemetry标准。"
        ),
        "length": "long",
        "domain": "technology",
        "language": "mixed",
        "type": "technical",
        "expected_entities": ["Kubernetes", "Kafka", "Redis", "PostgreSQL", "Elasticsearch", "JWT"],
        "expected_relations_min": 3,
    },
    "ancient_classical": {
        "text": (
            "出师表\n\n"
            "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。"
            "然侍卫之臣不懈于内，忠志之士忘身于外者，盖追先帝之殊遇，欲报之于陛下也。"
            "诚宜开张圣听，以光先帝遗德，恢弘志士之气，不宜妄自菲薄，引喻失义，"
            "以塞忠谏之路也。\n\n"
            "宫中府中，俱为一体，陟罚臧否，不宜异同。若有作奸犯科及为忠善者，"
            "宜付有司论其刑赏，以昭陛下平明之理，不宜偏私，使内外异法也。\n\n"
            "臣本布衣，躬耕于南阳，苟全性命于乱世，不求闻达于诸侯。先帝不以臣卑鄙，"
            "猥自枉屈，三顾臣于草庐之中，咨臣以当世之事，由是感激，遂许先帝以驱驰。"
            "后值倾覆，受任于败军之际，奉命于危难之间，尔来二十有一年矣。\n\n"
            "此段文字为三国时期蜀汉丞相诸葛亮于建兴五年（227年）北伐前写给后主刘禅的"
            "奏表，表达了诸葛亮对刘备知遇之恩的感激和对蜀汉事业的忠诚。"
            "\"鞠躬尽瘁，死而后已\"成为后世形容忠臣的经典成语。"
        ),
        "length": "long",
        "domain": "literature",
        "language": "cn",
        "type": "poetic",
        "expected_entities": ["诸葛亮", "刘备", "刘禅", "蜀汉", "出师表"],
        "expected_relations_min": 3,
    },
    "math_concept": {
        "text": (
            "欧拉公式 e^(iπ) + 1 = 0 被认为是数学中最优美的公式，它将五个最重要的数学常数"
            "联系在一起：自然对数的底数e（约等于2.71828）、虚数单位i（i²=-1）、"
            "圆周率π（约等于3.14159）、加法单位元素0和乘法单位元素1。这个公式是"
            "瑞士数学家Leonhard Euler在1748年发现的，是欧拉恒等式的特例。"
            "欧拉公式的更一般形式是e^(ix) = cos(x) + i·sin(x)，将复指数函数与"
            "三角函数联系起来。这一公式在傅里叶分析、量子力学、信号处理和电气工程"
            "等领域有广泛应用。Leonhard Euler是历史上最多产的数学家之一，"
            "一生发表了超过800篇论文和著作，涵盖了数学、物理学、天文学和工程学等"
            "众多领域。即便在1766年完全失明后，他依然通过口述继续进行数学研究，"
            "产出了近半数的著作。"
        ),
        "length": "medium",
        "domain": "science",
        "language": "mixed",
        "type": "narrative",
        "expected_entities": ["欧拉公式", "Leonhard Euler", "e", "π"],
        "expected_relations_min": 2,
    },
    "food_culture": {
        "text": (
            "中国八大菜系是指中国饮食文化中具有代表性的八个地方菜系：鲁菜（山东）、"
            "川菜（四川）、粤菜（广东）、苏菜（江苏）、浙菜（浙江）、闽菜（福建）、"
            "湘菜（湖南）和徽菜（安徽）。鲁菜是历史最悠久、技法最丰富的菜系，以孔府菜"
            "为代表，讲究\"食不厌精，脍不厌细\"。川菜以麻辣著称，代表菜有麻婆豆腐、"
            "回锅肉和水煮鱼，大量使用花椒和辣椒。粤菜注重食材原味，讲究清而不淡，"
            "烧鹅、白切鸡和虾饺是经典菜品。苏菜口味偏甜，擅长炖焖，松鼠桂鱼和"
            "叫花鸡是其名菜。浙江菜中的西湖醋鱼和东坡肉闻名全国，据说东坡肉是"
            "苏东坡在杭州任知州时发明的。闽菜的佛跳墙是一道复杂的名菜，"
            "据说\"坛启荤香飘四邻，佛闻弃禅跳墙来\"，因此得名。"
        ),
        "length": "long",
        "domain": "culture",
        "language": "cn",
        "type": "narrative",
        "expected_entities": ["八大菜系", "鲁菜", "川菜", "粤菜", "苏东坡"],
        "expected_relations_min": 4,
    },
    "space_exploration": {
        "text": (
            "China's space program has made remarkable progress in recent years. The China National Space"
            " Administration (CNSA) successfully landed the Chang'e-4 probe on the far side of the Moon"
            " in January 2019, making China the first country to achieve a soft landing on the lunar far side."
            " The Tianwen-1 mission reached Mars in February 2021, and the Zhurong rover explored"
            " the Martian surface for over a year. China completed its Tiangong space station in November"
            " 2022, becoming the second country to independently operate a permanent orbital laboratory"
            " after the International Space Station (ISS). The China Manned Space Agency (CMSA) plans"
            " to send astronauts to the Moon by 2030. Key figures include Yang Liwei, who became China's"
            " first astronaut (taikonaut) aboard Shenzhou 5 in October 2003, and Wang Yaping, who became"
            " the first Chinese woman to perform a spacewalk during the Shenzhou 13 mission in November 2021."
            " The Long March 5B rocket, developed by the China Academy of Launch Vehicle Technology (CALT),"
            " is China's most powerful launch vehicle, capable of lifting 25 tons to low Earth orbit."
        ),
        "length": "long",
        "domain": "science",
        "language": "en",
        "type": "narrative",
        "expected_entities": ["CNSA", "Chang'e-4", "Tiangong", "Yang Liwei", "Long March 5B"],
        "expected_relations_min": 4,
    },
    "music_history": {
        "text": (
            "贝多芬（Ludwig van Beethoven，1770-1827）是德国作曲家和钢琴家，被广泛认为是"
            "西方古典音乐史上最伟大的作曲家之一。他出生于波恩，22岁时移居维也纳，"
            "师从Joseph Haydn学习作曲。贝多芬的音乐创作跨越了古典主义和浪漫主义两个时期，"
            "被誉为连接这两大音乐时代的桥梁。他的代表作包括九部交响曲，其中第三交响曲"
            "《英雄》（Eroica）本是为拿破仑创作的，但当拿破仑称帝时，贝多芬愤怒地"
            "撕毁了扉页上的题献。第五交响曲《命运》以\"短-短-短-长\"的著名动机开头，"
            "第九交响曲《合唱》的末乐章采用了席勒的《欢乐颂》（Ode an die Freude），"
            "成为欧盟的盟歌。最令人敬佩的是，贝多芬在约28岁时开始失聪，但他的创作"
            "在完全失聪后达到了巅峰，第九交响曲就是在完全失聪状态下完成的。除了交响曲，"
            "他的32首钢琴奏鸣曲被称为钢琴音乐的\"新约圣经\"，其中《月光奏鸣曲》"
            "和《悲怆奏鸣曲》最为人所知。莫扎特（Wolfgang Amadeus Mozart）和巴赫"
            "（Johann Sebastian Bach）与贝多芬并称为西方古典音乐的\"三B\"巨头。"
        ),
        "length": "long",
        "domain": "art",
        "language": "mixed",
        "type": "narrative",
        "expected_entities": ["贝多芬", "Haydn", "拿破仑", "莫扎特", "巴赫", "维也纳"],
        "expected_relations_min": 4,
    },
    "economics_short": {
        "text": (
            "供给和需求是经济学最基本的两个概念。亚当·斯密在1776年出版的《国富论》中"
            "提出了\"看不见的手\"理论，认为自由市场中个体追求自身利益的行为会在不知不觉中"
            "促进社会整体福利。英国经济学家约翰·梅纳德·凯恩斯在1936年出版的《就业、"
            "利息和货币通论》中提出了凯恩斯经济学，主张在经济萧条时政府应通过财政政策"
            "和货币政策刺激需求，而非等待市场自我调节。"
        ),
        "length": "medium",
        "domain": "economics",
        "language": "cn",
        "type": "academic",
        "expected_entities": ["亚当·斯密", "国富论", "凯恩斯"],
        "expected_relations_min": 2,
    },
    "ecology_medium": {
        "text": (
            "亚马逊雨林（Amazon Rainforest）是地球上最大的热带雨林，面积约550万平方公里，"
            "横跨巴西、秘鲁、哥伦比亚等9个南美国家。它被称为\"地球之肺\"，产生全球约20%的"
            "氧气，同时也是最大的碳汇之一。雨林中栖息着约10%的地球已知物种，包括美洲豹、"
            "金刚鹦鹉、箭毒蛙和亚马逊河豚等独特物种。亚马逊河是世界水量最大的河流，"
            "其流域面积超过700万平方公里，淡水量占全球地表淡水的约20%。"
            "然而，亚马逊雨林正面临严重的砍伐威胁。巴西总统的环保政策直接影响着雨林的未来。"
            "2019年的亚马逊大火引起了全球关注，当年巴西亚马逊地区的森林砍伐面积达到"
            "9760平方公里。环保活动家和个人如巴西原住民领袖Raoni Metuktire一直在"
            "呼吁国际社会关注亚马逊雨林的保护问题。"
        ),
        "length": "long",
        "domain": "ecology",
        "language": "mixed",
        "type": "narrative",
        "expected_entities": ["亚马逊雨林", "巴西", "亚马逊河", "Raoni Metuktire"],
        "expected_relations_min": 3,
    },
    "education_system": {
        "text": (
            "中国的高等教育体系由普通本科院校、高职院校和成人教育机构组成。"
            "顶尖大学包括清华大学、北京大学、复旦大学、上海交通大学、浙江大学、"
            "中国科学技术大学、南京大学和哈尔滨工业大学，这些学校通常被称为\"C9联盟\"。"
            "985工程是1998年启动的旨在建设世界一流大学的计划，共39所大学入选。"
            "211工程则是面向21世纪建设约100所重点大学的计划。985和211已成为"
            "中国优质高等教育的代名词。近年来，\"双一流\"（世界一流大学和一流学科）"
            "建设取代了985/211成为新的国家高等教育发展战略。\n\n"
            "高考（全国普通高等学校招生统一考试）是中国大学入学的主要途径，每年6月举行。"
            "2024年全国高考报名人数达到1342万人。高考科目通常包括语文、数学和英语，"
            "加上文科综合或理科综合。近年来新高考改革引入了\"3+1+2\"模式，给予学生更多"
            "选科自由。清华大学的录取分数线常年位居全国最高，竞争极为激烈。"
        ),
        "length": "long",
        "domain": "education",
        "language": "cn",
        "type": "narrative",
        "expected_entities": ["清华大学", "北京大学", "C9联盟", "985工程", "高考"],
        "expected_relations_min": 4,
    },
}
