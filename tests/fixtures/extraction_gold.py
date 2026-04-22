"""
Gold-standard test inputs for the Deep-Dream remember extraction pipeline.

Each case provides human-annotated expected entities and relations, used to
measure extraction completeness (recall) and alignment accuracy.  Entities in
``must_find`` must be extracted with >= 80 % recall (test FAILS below this).
Entities in ``should_find`` require >= 50 % recall (WARN but do not fail).
``must_not_find`` has zero tolerance -- any match is a hard FAIL.

Relation tuples use the form (entity1, entity2) and require >= 60 % recall.

All 8 cases derived from real_world_datasets.py carry the exact source text.
4 additional cases (dialogue_tech, ambiguous_names, english_long, code_mixed)
are original inputs designed to cover dialogue format, ambiguous same-name
entities, pure-English multi-window text, and code-Chinese mixed content.
"""

__all__ = ['EXTRACTION_GOLD']

# Default must_not_find list shared by every case -- these are pipeline
# artefacts / meta-terms that should never appear as extracted entities.
_DEFAULT_MUST_NOT_FIND = [
    "处理进度", "系统状态", "抽取结果", "缓存数据", "步骤", "token", "json", "llm",
]

EXTRACTION_GOLD = {
    # ------------------------------------------------------------------
    # 1. biography_mayun  (from biography_short in real_world_datasets.py)
    # ------------------------------------------------------------------
    "biography_mayun": {
        "text": (
            "马云，1964年10月15日出生于浙江省杭州市，是中国著名企业家和阿里巴巴集团创始人。"
            "马云早年毕业于杭州师范学院（现杭州师范大学）外语系，曾在杭州电子工业学院担任英语教师。"
            "1995年，马云首次接触互联网，创办了中国最早的互联网公司之一\"中国黄页\"。"
            "1999年，马云在杭州湖畔花园的公寓中与17位伙伴共同创立了阿里巴巴，最初定位为"
            "B2B电子商务平台，旨在帮助中国中小企业对接全球市场。他凭借敏锐的商业嗅觉和"
            "卓越的演讲能力，带领阿里巴巴从一个小型创业公司发展为全球最大的电子商务集团之一。"
            "马云还推动创立了淘宝网（2003年）和支付宝（2004年），深刻改变了中国人的购物和支付方式。"
            "2014年，阿里巴巴在纽约证券交易所上市，创下了当时全球最大IPO记录。"
            "2019年9月10日，马云正式卸任阿里巴巴董事局主席一职，由张勇接任。"
        ),
        "domain": "biography",
        "expected_entities": {
            "must_find": [
                "马云", "阿里巴巴", "淘宝网", "支付宝", "杭州师范学院",
            ],
            "should_find": [
                "杭州", "纽约证券交易所", "张勇", "B2B", "中国黄页",
            ],
            "must_not_find": _DEFAULT_MUST_NOT_FIND,
        },
        "expected_relations": {
            "must_find": [
                ("马云", "阿里巴巴"),
                ("阿里巴巴", "淘宝网"),
                ("阿里巴巴", "支付宝"),
            ],
        },
    },

    # ------------------------------------------------------------------
    # 2. tech_go  (from tech_long in real_world_datasets.py)
    # ------------------------------------------------------------------
    "tech_go": {
        "text": (
            "Go语言（又称Golang）是由Google的Rob Pike和Ken Thompson于2007年开始设计、"
            "2009年以开源方式发布的静态类型编译型编程语言。Rob Pike此前在贝尔实验室参与了"
            "Plan 9操作系统和UTF-8编码的设计，而Ken Thompson则是Unix操作系统的共同创造者"
            "和C语言的前身B语言的作者。Go语言的设计初衷是解决Google内部大规模软件工程中"
            "遇到的编译速度慢、依赖管理复杂和并发编程困难等问题。\n\n"
            "Go语言最突出的特性是其轻量级的并发模型。通过goroutine和channel机制，开发者可以"
            "轻松编写高并发程序。一个goroutine仅占用约2KB的栈空间，因此单个Go程序可以同时运行"
            "数百万个goroutine。相比之下，传统的操作系统线程通常需要1MB以上的栈空间。Go的channel"
            "遵循\"不要通过共享内存来通信，而要通过通信来共享内存\"的设计哲学，有效避免了锁竞争问题。"
            "Go还提供了select语句用于多路复用channel操作，以及sync包中的互斥锁（Mutex）、"
            "等待组（WaitGroup）等传统同步原语，为不同并发场景提供了灵活的工具集。\n\n"
            "在编译方面，Go使用自举编译器，编译速度极快。Go的编译过程不依赖外部C编译器（从Go 1.5"
            "开始），生成的二进制文件是静态链接的，部署时无需额外依赖。这一点与Rust语言形成了鲜明对比："
            "Rust虽然在内存安全和零成本抽象方面更为严格和强大，但其编译速度较慢，学习曲线也更陡峭。"
            "Go选择了垃圾回收而非Rust的所有权系统来管理内存，牺牲了一定的内存控制精度，换取了更简单的开发体验。"
            "Go的垃圾回收器从早期的标记-清除算法逐步演进到并发标记-清除（CMS），再到现在使用的"
            "并发三色标记算法，暂停时间已大幅缩短至亚毫秒级别。\n\n"
            "Go语言在类型系统上采取了务实的折中方案。它没有类和继承，而是通过结构体嵌入（embedding）"
            "和接口（interface）实现代码复用和多态。Go的接口是隐式满足的——只要类型实现了接口要求的"
            "所有方法，就自动满足该接口，无需显式声明。这种鸭子类型的设计大幅降低了代码耦合度。"
            "Go 1.18引入了泛型（generics），进一步完善了类型系统的表达能力，使开发者可以编写"
            "类型安全的通用数据结构和算法。\n\n"
            "Go语言在云计算和微服务领域获得了广泛应用。Docker、Kubernetes、Terraform、etcd、"
            "Prometheus、Grafana等著名的云原生项目均使用Go编写。Go的标准库提供了丰富的网络编程"
            "和HTTP服务支持，加上goroutine的高效并发能力，使其成为构建网络服务和分布式系统的理想选择。"
            "截至2024年，Go在TIOBE编程语言排行榜中稳居前十，被Google、Uber、Twitch、Netflix、"
            "字节跳动、腾讯等众多科技公司广泛采用。Go语言的模块系统（Go Modules）从1.16版本起"
            "成为默认的依赖管理方案，解决了长期困扰Go社区的版本管理和依赖复制问题。"
        ),
        "domain": "technology",
        "expected_entities": {
            "must_find": [
                "Go语言", "Google", "Rob Pike", "Ken Thompson", "goroutine",
                "Docker", "Kubernetes",
            ],
            "should_find": [
                "Unix", "C语言", "Rust", "UTF-8", "泛型", "垃圾回收",
            ],
            "must_not_find": _DEFAULT_MUST_NOT_FIND,
        },
        "expected_relations": {
            "must_find": [
                ("Go语言", "Google"),
                ("Rob Pike", "Ken Thompson"),
                ("Go语言", "goroutine"),
            ],
        },
    },

    # ------------------------------------------------------------------
    # 3. history_tang  (from history_medium in real_world_datasets.py)
    # ------------------------------------------------------------------
    "history_tang": {
        "text": (
            "唐朝（618年—907年）是中国历史上最辉煌的朝代之一，由李渊建立，定都长安。"
            "唐太宗李世民在位期间（626年—649年），开创了著名的\"贞观之治\"，被视为中国古代"
            "治世的典范。李世民善于纳谏，重用房玄龄和杜如晦为宰相，二人并称\"房谋杜断\"，"
            "辅佐太宗推行均田制、完善科举制度、平定四方。房玄龄擅长谋略规划，杜如晦则善于"
            "果断决策，两人优势互补，成为唐朝初年最重要的政治搭档。此外，谏议大夫魏征以直言敢谏"
            "闻名，先后进谏二百余次，深得李世民信任。李世民曾感叹：\"以铜为镜，可以正衣冠；"
            "以古为镜，可以知兴替；以人为镜，可以明得失。\"魏征去世后，李世民痛失一面镜子。\n\n"
            "贞观年间，唐朝国力强盛，长安成为当时世界上最大的城市，人口超过百万，是丝绸之路"
            "的东方起点。唐朝实行开放包容的文化政策，吸引了大量外国使节、商人和留学生。"
            "来自日本、新罗、波斯、大食等国的使臣频繁往来，长安城内的西市汇聚了各国的珍奇商品。"
            "玄奘西行取经就发生在这一时期，他从长安出发，历经艰险到达天竺（今印度），"
            "在那烂陀寺学习佛法十余年，带回大量佛教经典并主持翻译工作。\n\n"
            "贞观之治为后来的开元盛世奠定了坚实基础。唐玄宗李隆基统治前期的开元年间（713年—741年），"
            "唐朝达到了政治、经济、文化的全面繁荣，史称\"开元盛世\"。诗人杜甫后来回忆道："
            "\"忆昔开元全盛日，小邑犹藏万家室。\"可惜天宝年间发生的安史之乱（755年—763年）"
            "成为唐朝由盛转衰的转折点，此后虽有元和中兴等短暂复兴，但藩镇割据、宦官专权等问题"
            "日益严重，最终在907年被朱温所灭，唐朝覆灭。"
        ),
        "domain": "history",
        "expected_entities": {
            "must_find": [
                "唐朝", "李渊", "李世民", "贞观之治", "房玄龄", "杜如晦",
                "魏征", "长安", "玄奘", "安史之乱",
            ],
            "should_find": [
                "开元盛世", "李隆基", "丝绸之路", "那烂陀寺", "朱温",
            ],
            "must_not_find": _DEFAULT_MUST_NOT_FIND,
        },
        "expected_relations": {
            "must_find": [
                ("李世民", "贞观之治"),
                ("房玄龄", "杜如晦"),
                ("玄奘", "天竺"),
            ],
        },
    },

    # ------------------------------------------------------------------
    # 4. science_quantum  (from science_long in real_world_datasets.py)
    # ------------------------------------------------------------------
    "science_quantum": {
        "text": (
            "量子计算是一种利用量子力学原理进行信息处理的新型计算范式。与经典计算机使用"
            "比特（bit）表示0或1不同，量子计算机使用量子比特（qubit），可以同时处于0和1的"
            "叠加态。这一特性源自量子力学中的叠加原理，奥地利物理学家薛定谔曾用著名的"
            "\"薛定谔的猫\"思想实验来阐释量子叠加的奇特性质：在未观测之前，猫处于既死又活的"
            "叠加状态。薛定谔因此成为量子力学奠基人之一，他提出的薛定谔方程是量子力学的"
            "核心方程，描述了量子系统的波函数随时间的演化规律。薛定谔方程分为含时和不含时"
            "两种形式，前者描述量子态随时间的动态演化，后者则用于确定系统的能量本征态。"
            "德国物理学家Max Born进一步给出了波函数的统计诠释，指出波函数的模方代表粒子"
            "在某位置出现的概率密度，这一解释成为哥本哈根诠释的核心内容。\n\n"
            "量子纠缠是量子计算的另一个关键资源。当两个量子比特处于纠缠态时，对一个比特的"
            "测量会瞬间影响另一个比特的状态，无论它们相距多远。爱因斯坦将这一现象称为"
            "\"幽灵般的超距作用\"（spooky action at a distance），并认为量子力学是不完备的。"
            "然而1964年，北爱尔兰物理学家John Bell提出了Bell不等式，后来的实验（特别是"
            "2022年诺贝尔物理学奖得主Alain Aspect、John Clauser和Anton Zeilinger的实验）"
            "证实了量子纠缠的真实性。量子纠缠使得量子计算机能够在特定问题上实现指数级的计算加速，"
            "例如Peter Shor于1994年提出的Shor算法可以在多项式时间内分解大整数，对现有的"
            "RSA加密体系构成潜在威胁。Lov Grover于1996年提出的Grover算法则可以在无序数据库中"
            "实现平方级搜索加速，为密码学和优化问题提供了新的工具。\n\n"
            "在工程实现方面，Google于2019年宣布其53量子比特的Sycamore处理器在特定任务上"
            "实现了\"量子优越性\"（quantum supremacy），仅用200秒完成了经典超级计算机需要约"
            "一万年才能完成的随机量子电路采样计算。IBM对此提出异议，认为经过优化的经典算法"
            "可以在更短时间内完成该任务。IBM在超导量子比特技术方面持续投入，推出了Eagle"
            "（127量子比特）、Osprey（433量子比特）和Condor（1121量子比特）等处理器。"
            "IBM还提出了量子体积（Quantum Volume）这一综合衡量指标，不仅考量量子比特数量，"
            "还考虑了量子门保真度和连通性，更加全面地评估量子处理器的实际计算能力。\n\n"
            "超导量子比特是目前最主流的量子计算硬件方案之一。Google和IBM都采用这一路线，"
            "利用在极低温（约15毫开尔文，由稀释制冷机实现）下运行的超导约瑟夫森结来实现量子比特。"
            "除了超导方案，还有离子阱（代表企业有IonQ和霍尼韦尔/Quantinuum，利用电磁场捕获"
            "并操控单个离子）、光量子（代表企业PsiQuantum，利用光子作为量子比特）和拓扑量子"
            "（微软主攻方向，利用马约拉纳费米子实现天然抗噪声的量子比特）等多种技术路线在并行发展。"
            "量子计算目前面临的主要挑战包括退相干（量子态极易受环境噪声干扰而崩溃）、"
            "量子纠错（需要大量物理量子比特来编码一个逻辑量子比特，Surface Code方案大约需要"
            "1000个物理量子比特来保护一个逻辑量子比特）和规模化扩展。这意味着实用的容错量子计算机"
            "可能需要数百万个物理量子比特。2023年，哈佛大学和MIT的研究团队在量子纠错方面取得了"
            "重要突破，成功实现了48个逻辑量子比特的容错操作，向实用化量子计算迈出了关键一步。"
        ),
        "domain": "science",
        "expected_entities": {
            "must_find": [
                "量子计算", "量子比特", "薛定谔", "Google", "IBM",
                "Sycamore", "Shor算法", "Grover算法", "Bell不等式",
            ],
            "should_find": [
                "薛定谔的猫", "Max Born", "量子纠缠", "量子纠错", "IonQ",
            ],
            "must_not_find": _DEFAULT_MUST_NOT_FIND,
        },
        "expected_relations": {
            "must_find": [
                ("Google", "Sycamore"),
                ("量子计算", "量子比特"),
                ("薛定谔", "薛定谔方程"),
            ],
        },
    },

    # ------------------------------------------------------------------
    # 5. literature_honglou  (from literature_medium in real_world_datasets.py)
    # ------------------------------------------------------------------
    "literature_honglou": {
        "text": (
            "《红楼梦》是中国古典文学的巅峰之作，由清代作家曹雪芹创作。曹雪芹（约1715年—约1763年）"
            "出身于曾经显赫一时的江宁织造曹家，其曾祖父曹玺、祖父曹寅先后担任江宁织造一职，"
            "深得康熙帝信任。曹家在康熙六次南巡中四次接驾，鼎盛一时。然而雍正年间，曹家因经济亏空"
            "和政治牵连被抄家，从此家道中落。这种由盛转衰的人生经历深刻影响了《红楼梦》的创作。"
            "小说以贾宝玉、林黛玉和薛宝钗之间的爱情悲剧为主线，通过贾、史、王、薛四大家族的兴衰，"
            "展现了封建社会末期的广阔画卷。\n\n"
            "贾宝玉是荣国府的嫡孙，衔玉而生，性格叛逆，厌恶仕途经济，却对女性充满尊重和怜惜。"
            "他视功名利禄为\"国贼禄鬼\"之流，在大观园中与众姐妹和丫鬟过着诗意的青春生活。"
            "林黛玉是宝玉的姑表妹，才华横溢却多愁善感，父母双亡后寄居贾府。她与宝玉之间的"
            "\"木石前盟\"象征着超越世俗的真挚爱情，她的《葬花吟》和《秋窗风雨夕》等诗作展现了她"
            "超凡的文学才华和敏感的内心世界。薛宝钗则是宝玉的姨表姐，代表\"金玉良缘\"，端庄贤淑，"
            "通情达理，符合封建礼教对女性的期待，佩戴的金锁与宝玉的通灵宝玉恰成一对。"
            "黛玉和宝钗的对比——一个是\"世外仙姝寂寞林\"，一个是\"山中高士晶莹雪\"——构成了小说中"
            "最动人的情感冲突。\n\n"
            "《红楼梦》与《三国演义》、《水浒传》、《西游记》并称为中国古典小说四大名著。"
            "曹雪芹生前完成了前八十回，后四十回一般认为是高鹗续写。它不仅是一部文学作品，"
            "更是一部百科全书式的文化宝典，涉及诗词、医药、饮食、建筑、服饰、戏曲等方方面面。"
            "红学作为专门研究《红楼梦》的学问，已经发展成为一门国际性的学术领域，分为索隐派、"
            "考证派和评论派等多个流派，吸引了胡适、俞平伯、周汝昌等无数学者的深入研究。"
        ),
        "domain": "literature",
        "expected_entities": {
            "must_find": [
                "红楼梦", "曹雪芹", "贾宝玉", "林黛玉", "薛宝钗", "高鹗",
            ],
            "should_find": [
                "贾府", "大观园", "胡适", "四大名著",
            ],
            "must_not_find": _DEFAULT_MUST_NOT_FIND,
        },
        "expected_relations": {
            "must_find": [
                ("曹雪芹", "红楼梦"),
                ("贾宝玉", "林黛玉"),
            ],
        },
    },

    # ------------------------------------------------------------------
    # 6. business_tesla  (from business_short in real_world_datasets.py)
    # ------------------------------------------------------------------
    "business_tesla": {
        "text": (
            "特斯拉（Tesla Inc.）是美国电动汽车及清洁能源公司，由Martin Eberhard和Marc Tarpenning"
            "于2003年在加州硅谷创立。Elon Musk于2004年加入A轮融资并成为最大股东，后出任CEO。"
            "特斯拉先后推出了Roadster、Model S、Model 3、Model Y和Cybertruck等多款电动车型，"
            "推动了全球汽车行业向电动化转型。Model 3成为全球销量最高的电动汽车之一。"
            "2019年，特斯拉在上海临港建立了超级工厂（Gigafactory Shanghai），这是中国首家"
            "外商独资的汽车制造工厂，也是特斯拉在美国以外的首座整车工厂，大幅提升了特斯拉"
            "在亚太市场的交付能力。Elon Musk同时也是SpaceX的创始人兼CEO，SpaceX开发了"
            "猎鹰9号（Falcon 9）可回收火箭和龙飞船（Dragon），在商业航天领域取得了突破性进展。"
            "此外，Musk还创立了Neuralink和The Boring Company，并于2022年收购了Twitter（后更名为X）。"
        ),
        "domain": "business",
        "expected_entities": {
            "must_find": [
                "特斯拉", "Elon Musk", "SpaceX", "Model 3", "上海",
            ],
            "should_find": [
                "Model S", "Cybertruck", "Falcon 9", "Neuralink",
            ],
            "must_not_find": _DEFAULT_MUST_NOT_FIND,
        },
        "expected_relations": {
            "must_find": [
                ("Elon Musk", "特斯拉"),
                ("Elon Musk", "SpaceX"),
            ],
        },
    },

    # ------------------------------------------------------------------
    # 7. mixed_internet  (from mixed_domains in real_world_datasets.py)
    # ------------------------------------------------------------------
    "mixed_internet": {
        "text": (
            "互联网的发展是技术、历史与商业交汇的典型案例。1969年，美国国防部高级研究计划署"
            "（DARPA）建立了ARPANET，这是互联网的前身，最初仅连接了加州大学洛杉矶分校（UCLA）、"
            "斯坦福研究院（SRI）、加州大学圣芭芭拉分校（UCSB）和犹他大学四个节点。"
            "ARPANET采用分组交换技术，灵感来自RAND公司的Paul Baran和英国国家物理实验室的"
            "Donald Davies各自独立提出的分布式通信网络概念。1974年，Vint Cerf和Bob Kahn共同"
            "设计了TCP/IP协议，为不同网络之间的互联互通奠定了基础。Vint Cerf因此被尊为"
            "\"互联网之父\"之一。1983年1月1日，ARPANET正式从NCP协议切换到TCP/IP协议，"
            "这一天被许多技术史学者视为现代互联网的诞生日。\n\n"
            "1989年，英国科学家Tim Berners-Lee在欧洲核子研究中心（CERN）提出了万维网"
            "（World Wide Web）的构想，发明了HTTP协议、HTML语言和URL寻址系统，使普通人"
            "也能便捷地浏览和发布信息。1993年，伊利诺伊大学的Marc Andreessen开发了Mosaic浏览器，"
            "这是第一个支持内嵌图片的图形化网页浏览器，极大地推动了万维网的普及。"
            "Andreessen后来创立了网景公司（Netscape），其Navigator浏览器在1990年代中期占据了"
            "主导地位，但也引发了与微软Internet Explorer之间著名的\"浏览器大战\"。\n\n"
            "1990年代中后期，互联网开始大规模商业化。1994年，杨致远和David Filo在斯坦福大学"
            "创建了雅虎（Yahoo），成为最早的互联网门户网站之一，开创了\"门户时代\"。"
            "1995年，Jeff Bezos在车库里创办了亚马逊（Amazon），最初只卖书，后来发展为"
            "全球最大的电子商务平台。1998年，Larry Page和Sergey Brin在斯坦福大学开发了"
            "PageRank算法，创立了谷歌（Google）。Google凭借卓越的搜索技术迅速崛起，"
            "成为全球最有价值的科技公司之一，其AdWords广告业务彻底改变了互联网商业模式，"
            "开启了\"搜索经济\"时代。2004年，Google在纳斯达克上市，此后不断扩展业务版图，"
            "先后推出了Gmail、Google Maps、Android操作系统、Chrome浏览器等产品。\n\n"
            "在中国，互联网的发展同样深刻影响了商业格局。1997年，丁磊在广州创立了网易。"
            "1998年，马化腾在深圳创立了腾讯，推出了QQ即时通讯工具。1999年，马云在杭州创立了"
            "阿里巴巴，最初从事B2B电子商务。2000年，李彦宏在北京创立了百度。这四家公司"
            "被称为中国互联网的\"四大门户\"，开启了中国的互联网时代。2003年，阿里巴巴推出淘宝网，"
            "与eBay在中国市场展开激烈竞争，最终淘宝凭借免费策略和本地化运营击败了eBay。"
            "同年，支付宝上线，解决了中国互联网交易的信任问题。阿里巴巴的崛起代表了中国互联网"
            "企业的典型发展路径：从模仿到创新，从本土市场到全球化。如今，阿里巴巴已成为涵盖"
            "电子商务、云计算（阿里云）、数字支付和物流的庞大商业生态系统。\n\n"
            "从ARPANET到Google，从TCP/IP到淘宝，互联网的演变展现了技术创新如何推动社会变革。"
            "Vint Cerf的TCP/IP协议将分散的网络连接成一个整体，Tim Berners-Lee的万维网让信息"
            "触手可及，Larry Page的PageRank重新定义了信息检索，马云的电子商务模式重塑了商业形态。"
            "这些看似独立的技术和商业创新，共同编织了现代数字世界的网络。展望未来，"
            "量子计算作为下一代计算技术，未来可能再次颠覆互联网的基础架构——正如Google和IBM"
            "正在量子计算领域展开的新一轮技术竞赛所预示的那样。区块链技术（以比特币为代表）"
            "和人工智能（以ChatGPT为代表）也正在书写互联网演化的下一个篇章。"
        ),
        "domain": "mixed",
        "expected_entities": {
            "must_find": [
                "ARPANET", "TCP/IP", "Vint Cerf", "Tim Berners-Lee",
                "Google", "Larry Page", "马云", "阿里巴巴", "淘宝网",
            ],
            "should_find": [
                "DARPA", "Marc Andreessen", "Jeff Bezos", "丁磊",
                "马化腾", "李彦宏", "百度",
            ],
            "must_not_find": _DEFAULT_MUST_NOT_FIND,
        },
        "expected_relations": {
            "must_find": [
                ("Vint Cerf", "TCP/IP"),
                ("Tim Berners-Lee", "万维网"),
                ("Larry Page", "Google"),
                ("马云", "阿里巴巴"),
            ],
        },
    },

    # ------------------------------------------------------------------
    # 8. tech_k8s  (from tech_short in real_world_datasets.py)
    # ------------------------------------------------------------------
    "tech_k8s": {
        "text": (
            "Kubernetes是一个开源的容器编排系统，最初由Google设计并捐赠给云原生计算基金会（CNCF）。"
            "它基于Google内部运行了十多年的Borg系统，用于自动化部署、扩展和管理容器化应用程序。"
            "Kubernetes与Docker容器技术紧密配合，支持多种容器运行时，已成为云计算领域的核心基础设施。"
            "通过Pod、Service、Deployment等抽象概念，Kubernetes简化了微服务架构下的应用管理。"
            "Kubernetes的核心组件包括etcd分布式存储、kube-apiserver、kube-scheduler和kubelet，"
            "共同构成了一个声明式的、自愈的容器编排平台。"
        ),
        "domain": "technology",
        "expected_entities": {
            "must_find": [
                "Kubernetes", "Google", "Docker", "CNCF", "Pod", "etcd",
            ],
            "should_find": [
                "Borg", "kube-apiserver", "微服务",
            ],
            "must_not_find": _DEFAULT_MUST_NOT_FIND,
        },
        "expected_relations": {
            "must_find": [
                ("Kubernetes", "Google"),
                ("Kubernetes", "Docker"),
            ],
        },
    },

    # ------------------------------------------------------------------
    # 9. dialogue_tech  (NEW -- dialogue format, ~400 chars)
    # ------------------------------------------------------------------
    "dialogue_tech": {
        "text": (
            "A: 我们项目准备用Kubernetes做容器编排，你有什么建议？\n"
            "B: K8s确实不错，搭配Helm做包管理会很方便。你们用什么语言开发？\n"
            "A: 后端用Go，微服务架构，用gRPC做服务间通信。\n"
            "B: 那很适合K8s。数据库方面呢？\n"
            "A: 主库用PostgreSQL，缓存用Redis，搜索引擎用Elasticsearch。\n"
            "B: 建议用Prometheus做监控，Grafana做可视化。日志收集用ELK Stack。\n"
            "A: 好的，CI/CD呢？\n"
            "B: GitHub Actions或者GitLab CI都可以。镜像仓库用Harbor。"
        ),
        "domain": "technology",
        "expected_entities": {
            "must_find": [
                "Kubernetes", "Helm", "Go", "gRPC", "PostgreSQL",
                "Redis", "Elasticsearch",
            ],
            "should_find": [
                "Prometheus", "Grafana", "ELK", "GitHub Actions", "Harbor",
            ],
            "must_not_find": _DEFAULT_MUST_NOT_FIND,
        },
        "expected_relations": {
            "must_find": [
                ("Kubernetes", "Helm"),
                ("PostgreSQL", "Redis"),
            ],
        },
    },

    # ------------------------------------------------------------------
    # 10. ambiguous_names  (NEW -- same-name-different-entity, ~400 chars)
    # ------------------------------------------------------------------
    "ambiguous_names": {
        "text": (
            "张伟是北京大学计算机科学系的教授，研究方向是自然语言处理和深度学习。"
            "他2015年从斯坦福大学获得博士学位，师从Christopher Manning。"
            "张伟的研究团队开发了多个开源NLP工具包。\n\n"
            "另一位张伟是清华大学物理系的教授，主要研究量子光学和冷原子物理。"
            "他毕业于麻省理工学院（MIT），在Wolfgang Ketterle指导下完成了博士论文。"
            "这位张伟近年来致力于量子计算实验平台的搭建。\n\n"
            "还有一位张伟在上海交通大学医学院附属瑞金医院担任心内科主任医师，"
            "擅长冠心病介入治疗，发表SCI论文30余篇。"
        ),
        "domain": "mixed",
        "expected_entities": {
            "must_find": [
                "张伟", "北京大学", "斯坦福大学", "Christopher Manning",
                "清华大学", "MIT", "Wolfgang Ketterle", "上海交通大学",
                "瑞金医院",
            ],
            "should_find": [
                "自然语言处理", "深度学习", "量子光学", "冷原子物理",
                "量子计算", "冠心病",
            ],
            "must_not_find": _DEFAULT_MUST_NOT_FIND,
        },
        "expected_relations": {
            "must_find": [
                ("张伟", "北京大学"),
                ("张伟", "斯坦福大学"),
                ("张伟", "清华大学"),
            ],
        },
    },

    # ------------------------------------------------------------------
    # 11. english_long  (NEW -- pure English multi-window, ~800 chars)
    # ------------------------------------------------------------------
    "english_long": {
        "text": (
            "The Renaissance was a cultural movement that profoundly affected European intellectual life "
            "in the early modern period. Beginning in Italy, particularly in Florence, the Renaissance "
            "spread to the rest of Europe by the 16th century. Its influence was felt in art, architecture, "
            "philosophy, literature, music, science, technology, and politics.\n\n"
            "Leonardo da Vinci, born in 1452 in Vinci, Italy, is often described as the archetype of the "
            "Renaissance Man. His works include the Mona Lisa and The Last Supper, two of the most famous "
            "paintings in history. Leonardo was also a brilliant engineer and scientist, designing flying "
            "machines, tanks, and solar power concentrators centuries before they became reality.\n\n"
            "Michelangelo Buonarroti, another towering figure of the Renaissance, sculpted David and painted "
            "the ceiling of the Sistine Chapel. His rival, Raphael, painted The School of Athens, depicting "
            "Plato and Aristotle at the center. The Medici family, particularly Lorenzo de Medici (known as "
            "Lorenzo the Magnificent), were crucial patrons of Renaissance art in Florence.\n\n"
            "Niccolo Machiavelli wrote The Prince in 1513, a seminal work of political philosophy. "
            "Galileo Galilei, born in Pisa in 1564, made groundbreaking contributions to astronomy, physics, "
            "and the scientific method. His improvements to the telescope led to the discovery of Jupiter's "
            "four largest moons. The Renaissance laid the groundwork for the Scientific Revolution and the "
            "Enlightenment that followed."
        ),
        "domain": "history",
        "expected_entities": {
            "must_find": [
                "Renaissance", "Leonardo da Vinci", "Mona Lisa", "Michelangelo",
                "David", "Sistine Chapel", "Raphael", "Medici",
                "Machiavelli", "Galileo Galilei",
            ],
            "should_find": [
                "Florence", "Plato", "Aristotle", "Lorenzo de Medici",
                "The Prince", "Jupiter",
            ],
            "must_not_find": _DEFAULT_MUST_NOT_FIND,
        },
        "expected_relations": {
            "must_find": [
                ("Leonardo da Vinci", "Mona Lisa"),
                ("Michelangelo", "David"),
                ("Galileo Galilei", "telescope"),
            ],
        },
    },

    # ------------------------------------------------------------------
    # 12. code_mixed  (NEW -- code + Chinese mixed, ~400 chars)
    # ------------------------------------------------------------------
    "code_mixed": {
        "text": (
            "FastAPI是Python生态中最受欢迎的Web框架之一，由Sebastian Ramirez开发。"
            "它基于Starlette和Pydantic构建，支持异步编程（async/await）。"
            "安装FastAPI只需执行 pip install fastapi ，然后安装ASGI服务器 uvicorn 。"
            "FastAPI自动生成OpenAPI（Swagger）文档，开发者可以直接在浏览器中测试API。"
            "与Flask和Django相比，FastAPI的性能更接近Go语言的Gin框架和Node.js的Express框架。"
            "FastAPI支持依赖注入（Dependency Injection）、WebSocket、GraphQL等高级特性。"
            "它还内置了数据验证和序列化功能，通过Pydantic的BaseModel类来定义请求和响应模型。"
        ),
        "domain": "technology",
        "expected_entities": {
            "must_find": [
                "FastAPI", "Python", "Sebastian Ramirez", "Starlette",
                "Pydantic", "OpenAPI", "Swagger", "Flask", "Django",
            ],
            "should_find": [
                "uvicorn", "Gin", "Express", "GraphQL", "WebSocket",
            ],
            "must_not_find": _DEFAULT_MUST_NOT_FIND,
        },
        "expected_relations": {
            "must_find": [
                ("FastAPI", "Python"),
                ("FastAPI", "Pydantic"),
                ("FastAPI", "Starlette"),
            ],
        },
    },

    # ------------------------------------------------------------------
    # 13. concept_versioning  (NEW -- dense Chinese technical text, ~300 chars)
    # ------------------------------------------------------------------
    "concept_versioning": {
        "text": (
            "那按照我的理解是，处理一个episode，它抽取出来的概念将会面临两种情况。"
            "第一种通过对齐是发现图谱中已有存在可能是同个概念的数据，那么这个感念版本肯定就一定要更新一版了，"
            "至于新版本content是否要更新就是看llm判断。如果要沿用之前的，那就直接复制。"
            "如果是合并更新，那就按照git形式对已有的content结合当前episode提取出来的概念content做合并，"
            "赋予给新版本的这个概念。第二种情况就是对齐发现，这是一个全新的概念，那就直接存。"
            "这样的话，只要一个episode中抽取出来的概念，肯定会有一次更新版本，一一对应的。是吧？"
            "这样假如是阅读一本小说，按照500字切分小说长文，那第一个episode提到某个主角，"
            "第二个episode又提到，第三个还提到，那当处理完第三个episode后，主角概念是有3个版本的。"
            "而且每个版本都能连接到对应的各自的episode。当然有可能，主角的介绍属性并没有什么变化，"
            "所以content可能这三个版本都没有变化。当然如果每个episode都引入了主角新的信息，"
            "那每个版本都是增量更新了。"
        ),
        "domain": "technology",
        "expected_entities": {
            "must_find": [
                "episode", "概念抽取", "概念对齐", "概念版本", "图谱",
                "全新概念", "合并更新", "LLM判断", "版本溯源", "直接存储",
                "长文分块", "主角概念", "增量更新", "内容沿用", "git形式合并",
            ],
            "should_find": [
                "版本content属性", "一一对应规则", "500字切分", "小说阅读",
                "首版本", "版本管理颗粒度", "直接复制", "同个概念判定",
                "主角介绍属性", "已有content", "新版本",
            ],
            "must_not_find": _DEFAULT_MUST_NOT_FIND,
        },
        "expected_relations": {
            "must_find": [
                ("episode", "概念抽取"),
                ("概念抽取", "概念对齐"),
                ("概念对齐", "图谱"),
                ("概念对齐", "全新概念"),
                ("概念版本", "版本溯源"),
                ("合并更新", "git"),
                ("LLM判断", "内容沿用"),
                ("LLM判断", "合并更新"),
                ("全新概念", "直接存储"),
                ("长文分块", "主角"),
                ("主角概念", "增量更新"),
                ("概念版本", "episode"),
                ("长文分块", "小说"),
            ],
        },
    },
}
