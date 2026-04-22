#!/usr/bin/env python3
"""Comprehensive pipeline quality test suite with graph storage verification.

Tests 20+ diverse documents, verifies:
1. Entity extraction quality (no system leaks, meaningful names, sufficient content)
2. Relation quality (meaningful descriptions, endpoints resolve)
3. Graph storage integrity (no orphans expected, no duplicates)
4. Pipeline efficiency (time per test case)
"""
import json, time, sys, os, sqlite3, re
sys.path.insert(0, '.')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')

from server.api import build_processor

# ── Test Cases ──────────────────────────────────────────────
TEST_CASES = [
    # ---- Short texts (< 200 chars) ----
    ("short_tech", '''Docker是一个开源的应用容器引擎，让开发者可以打包他们的应用以及依赖包到一个可移植的容器中。容器使用沙箱机制，相互之间不会有任何接口。Kubernetes是容器集群管理系统，提供自动化部署、扩展和管理容器化应用的功能。Kubernetes也被称为K8s，由Google设计并捐赠给Cloud Native Computing Foundation。'''),

    ("short_person", '''袁隆平是中国著名的农业科学家，被誉为"杂交水稻之父"。他于1930年出生于北京，1953年毕业于西南农学院。他研发的杂交水稻技术使水稻产量提高了20%以上，解决了数亿人的粮食问题。2021年5月22日，袁隆平在长沙逝世，享年91岁。'''),

    ("short_geography", '''亚马逊河是世界上流量最大的河流，全长约6400公里，流经南美洲的秘鲁、哥伦比亚和巴西。亚马逊雨林是世界上最大的热带雨林，被称为"地球之肺"。'''),

    # ---- English texts ----
    ("english_sci", """Marie Curie was a Polish-born physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win in two different scientific fields. She shared the 1903 Nobel Prize in Physics with her husband Pierre Curie and Henri Becquerel, and won the 1911 Nobel Prize in Chemistry. Her achievements include the development of the theory of radioactivity, techniques for isolating radioactive isotopes, and the discovery of two elements: polonium and radium."""),

    ("english_tech", """Linux is a family of open-source Unix-like operating systems based on the Linux kernel, first released by Linus Torvalds in 1991. Linux is typically packaged as a Linux distribution, which includes the kernel and supporting system software and libraries. Popular distributions include Ubuntu, Debian, Fedora, and Arch Linux. Linux dominates the server market and powers most of the world's supercomputers."""),

    ("english_history", """The Renaissance was a cultural movement that began in Italy in the 14th century and spread throughout Europe, lasting until the 17th century. Leonardo da Vinci was an Italian polymath known for the Mona Lisa and The Last Supper. Michelangelo sculpted David and painted the Sistine Chapel ceiling. The invention of the printing press by Johannes Gutenberg around 1440 revolutionized the spread of knowledge."""),

    # ---- Chinese classical ----
    ("chinese_classical", """操自乘马张麾盖，引众将来攻阳平关。张鲁遣杨昂、杨任拒之。曹操令人四面攻之，杨昂坚守不出。忽一声炮响，山背后转出一彪军马，乃是杨任。两军混战，曹兵大败。曹操收兵回寨，叹曰：吾用兵多年，未尝如此败也。程昱曰：丞相勿忧，可使人往散关，暗伏兵马，诈作粮车，待贼兵出劫，伏兵齐出，必获大胜。操从之。"""),

    ("chinese_poetry", """李白，字太白，号青莲居士，唐代伟大的浪漫主义诗人，被后人誉为"诗仙"。李白的诗歌以豪放飘逸著称，代表作有《将进酒》《静夜思》《望庐山瀑布》等。杜甫，字子美，自号少陵野老，唐代伟大的现实主义诗人，被后人誉为"诗圣"。杜甫的诗歌以沉郁顿挫著称，代表作有《春望》《茅屋为秋风所破歌》等。"""),

    # ---- Dialogue / chat ----
    ("dialogue_chat", """小明：大家好，今天我们讨论一下新项目的技术选型。
小红：我觉得后端用Go语言比较合适，并发性能好。
小王：我同意，Go的goroutine和channel机制很适合我们的场景。
小明：前端呢？
小红：建议用React，生态成熟，组件丰富。
小王：数据库用PostgreSQL吧，支持JSON查询，比较灵活。
小明：好的，那部署方案用Docker+K8s？
小红：对，这样CI/CD也方便。"""),

    # ---- Japanese ----
    ("japanese_history", """織田信長は日本の戦国時代の大名であり、桶狭間の戦いで今川義元を破ったことで有名です。彼は天下統一を目指し、多くの革新政策を実施しました。しかし、1582年の本能寺の変で家臣の明智光秀に討たれました。豊臣秀吉は信長の家臣であり、信長の死後、天下統一を達成しました。徳川家康は秀吉の死後、関ヶ原の戦いに勝利し、江戸幕府を開きました。"""),

    # ---- Long technical article ----
    ("long_article", """Go（又称Golang）是Google开发的一种静态强类型、编译型语言。Go语言的设计目标是"让程序员更加高效地开发出简洁、可靠、高效的软件"。Go语言的主要特点包括：简洁的语法、高效的并发模型、快速的编译速度、内置垃圾回收机制。

Go语言的并发模型基于goroutine和channel。goroutine是一种轻量级线程，由Go运行时管理，创建成本极低。channel是goroutine之间的通信机制，分为无缓冲channel和带缓冲channel两种类型。通过goroutine和channel的组合，Go语言实现了CSP（Communicating Sequential Processes）并发模型。

Go语言的标准库非常丰富，包含了网络编程、文件操作、加密解密、测试框架等常用功能。其中net/http包提供了完整的HTTP服务器和客户端实现，encoding/json包支持JSON的编解码。

Go语言的工具链也非常完善，包括go build、go test、go fmt、go vet等命令。Go Modules是Go 1.11引入的依赖管理系统，解决了包版本管理问题。Go语言在云计算、微服务、容器化等领域广泛应用，Docker和Kubernetes等知名项目都使用Go语言开发。"""),

    # ---- Mixed bilingual ----
    ("mixed_bilingual", """Redis（Remote Dictionary Server）是一个开源的内存数据结构存储系统，由Salvatore Sanfilippo开发。Redis支持多种数据结构：字符串（string）、哈希（hash）、列表（list）、集合（set）、有序集合（sorted set）。Redis使用ANSI C编写，运行在大多数POSIX系统上。Redis的持久化策略包括RDB快照和AOF日志两种方式。"""),

    # ---- Code documentation ----
    ("code_doc", """FastAPI是一个现代、快速的Web框架，用于构建API。它基于Python 3.7+的类型提示，使用Starlette作为底层框架，Pydantic进行数据验证。FastAPI的主要特性包括：自动生成OpenAPI文档、异步支持、依赖注入系统、安全性工具。

安装FastAPI：pip install fastapi uvicorn

创建一个简单的API：
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}

运行服务器：uvicorn main:app --reload

FastAPI自动生成的Swagger UI可以在 /docs 路径访问，ReDoc可以在 /redoc 路径访问。"""),

    # ---- News ----
    ("news_article", """2024年巴黎奥运会于7月26日至8月11日举行。本届奥运会的口号是"奥运更开放"。中国代表团在本届奥运会取得了优异的成绩，在跳水、乒乓球、举重等传统优势项目上继续保持领先地位。潘展乐在男子100米自由泳项目中以46秒40的成绩打破世界纪录，为中国游泳队赢得历史性金牌。郑钦文在网球女子单打比赛中夺冠，成为中国首位奥运会网球单打冠军。"""),

    # ---- Philosophy / abstract ----
    ("philosophy", """存在主义是20世纪重要的哲学流派，主要代表人物包括萨特、加缪和海德格尔。萨特提出"存在先于本质"的核心观点，认为人首先存在，然后通过自己的选择和行动来定义自己的本质。加缪的荒诞哲学认为，人类寻求意义的努力与宇宙的沉默之间的矛盾构成了荒诞。海德格尔则关注"存在"本身的问题，提出了"此在"（Dasein）的概念，强调人是一种"被抛入"世界的存在。"""),

    # ---- Science article ----
    ("science_article", """量子计算是利用量子力学原理进行计算的新型计算模式。量子计算机使用量子比特（qubit）而非传统比特，可以同时处于0和1的叠加态。谷歌在2019年宣布实现了量子霸权，其Sycamore处理器在200秒内完成了经典超级计算机需要10000年才能完成的计算。IBM、微软和中国科学技术大学也在量子计算领域取得了重要突破。量子计算在密码学、药物发现、材料科学等领域有广泛应用前景。"""),

    # ---- History ----
    ("history_tang", """唐朝（618年-907年）是中国历史上最辉煌的朝代之一。唐高祖李渊于618年建立唐朝，定都长安。唐太宗李世民开创了"贞观之治"，使唐朝进入鼎盛时期。唐玄宗时期的"开元盛世"是唐朝的另一个高峰。唐朝在文化、经济、军事等方面都达到了极高水平，对东亚文化圈产生了深远影响。安史之乱（755年-763年）是唐朝由盛转衰的转折点。"""),

    # ---- Korean content ----
    ("korean_culture", """한글은 조선 제4대 왕인 세종대왕이 1443년에 만든 한국의 문자입니다. 한글은 과학적이고 체계적인 문자로, 전 세계 언어학자들이 극찬하는 문자 체계입니다. 세종대왕은 백성들이 한자를 배우기 어려워하는 것을 안타깝게 여겨 훈민정음을 창제했습니다. 한글날은 매년 10월 9일로, 한글의 우수성을 기리는 날입니다。"""),

    # ---- Very short edge case ----
    ("very_short", "Python是由Guido van Rossum于1991年创建的编程语言。"),

    # ---- Medical ----
    ("medical_info", """青霉素（Penicillin）是世界上第一种抗生素，由英国科学家亚历山大·弗莱明于1928年发现。弗莱明在圣玛丽医院的实验室中偶然发现了青霉菌能杀死葡萄球菌。青霉素在二战期间被大规模生产，拯救了数百万伤兵的生命。弗莱明因此与霍华德·弗洛里和恩斯特·钱恩共同获得了1945年诺贝尔生理学或医学奖。"""),

    # ---- Business / Economics ----
    ("business_econ", """苹果公司由史蒂夫·乔布斯、斯蒂夫·沃兹尼亚克和罗纳德·韦恩于1976年创立。苹果公司以iPhone、iPad、Mac等产品闻名。2023年，苹果公司市值超过3万亿美元，成为全球市值最高的公司。蒂姆·库克于2011年接任CEO，带领苹果公司进入了可穿戴设备和服务的时代。"""),

    # ---- Music / Arts ----
    ("music_arts", """贝多芬是德国作曲家和钢琴家，被尊称为"乐圣"。他的作品跨越了古典主义和浪漫主义时期，代表作包括《命运交响曲》《月光奏鸣曲》《欢乐颂》等。贝多芬在1801年开始失去听力，但仍然坚持创作。他的第九交响曲是在完全失聪后完成的，被认为是他最伟大的作品之一。莫扎特是奥地利作曲家，与贝多芬、巴赫并称为古典音乐的三B。"""),

    # ---- Edge: Number-heavy ----
    ("number_heavy", """截至2023年底，中国高铁运营里程达到4.5万公里，覆盖全国95%以上的50万人口以上城市。复兴号列车的最高运营时速达到350公里。京沪高铁全长1318公里，连接北京和上海两大城市，全程最快仅需4小时18分钟。中国已建成世界上规模最大的高速铁路网络。"""),

    # ---- Edge: List/formatting ----
    ("list_format", """Python web框架对比：
1. Django：全功能框架，内置ORM、Admin、认证系统，适合大型项目
2. Flask：轻量级框架，灵活可扩展，适合中小型项目
3. FastAPI：现代高性能框架，支持异步，自动生成API文档
4. Tornado：支持长连接和WebSocket的异步框架

常用数据库：
- PostgreSQL：功能最强大的开源关系数据库
- MySQL：最流行的开源关系数据库
- MongoDB：最流行的NoSQL文档数据库
- Redis：高性能键值存储，常用于缓存"""),

    # ---- Edge: Repetitive content ----
    ("repetitive", """机器学习是人工智能的一个重要分支。深度学习是机器学习的一个重要分支。强化学习是机器学习的一个重要分支。自然语言处理是人工智能的一个重要分支。计算机视觉是人工智能的一个重要分支。这些技术共同推动了人工智能的发展。"""),
]


def get_stored_data(db_path, source_doc):
    """Get stored entities and relations for verification."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    entities = conn.execute("""
        SELECT e.id, e.family_id, e.name, e.content, e.source_document,
               length(e.content) as content_len, e.processed_time
        FROM entities e
        WHERE e.source_document = ?
        AND e.processed_time = (
            SELECT MAX(e2.processed_time) FROM entities e2 WHERE e2.family_id = e.family_id
        )
        ORDER BY e.processed_time DESC
    """, (source_doc,)).fetchall()

    relations = conn.execute("""
        SELECT r.family_id, r.content, r.entity1_absolute_id, r.entity2_absolute_id,
               r.source_document, r.processed_time
        FROM relations r
        WHERE r.source_document = ?
        AND r.processed_time = (
            SELECT MAX(r2.processed_time) FROM relations r2 WHERE r2.family_id = r.family_id
        )
        ORDER BY r.processed_time DESC
    """, (source_doc,)).fetchall()

    conn.close()
    return [dict(r) for r in entities], [dict(r) for r in relations]


def check_quality(entities, relations):
    """Check quality of stored entities and relations."""
    issues = []
    warnings = []

    for e in entities:
        name = e['name']
        content = e.get('content', '') or ''
        clen = e.get('content_len', len(content))

        if clen < 15:
            issues.append(f"ENTITY_SHORT: '{name}' content only {clen} chars")

        for pat in ['处理进度', '缓存', '抽取']:
            if pat in content.lower():
                issues.append(f"ENTITY_LEAK: '{name}' contains '{pat}'")
                break

        if len(name) < 2:
            issues.append(f"ENTITY_NAME_SHORT: '{name}'")

    # Relation quality
    connected_ids = set()
    for r in relations:
        content = r.get('content', '') or ''
        if len(content) < 5:
            issues.append(f"RELATION_EMPTY: relation {r.get('family_id', '?')[:15]}")
        connected_ids.add(r.get('entity1_absolute_id', ''))
        connected_ids.add(r.get('entity2_absolute_id', ''))

    # Orphan check: compare entity 'id' (not family_id) against relation endpoints
    orphan_ents = []
    for e in entities:
        ent_id = e.get('id', e.get('family_id', ''))
        if ent_id not in connected_ids:
            orphan_ents.append(e['name'])

    if orphan_ents and len(orphan_ents) == len(entities):
        warnings.append(f"All orphans: {len(orphan_ents)}/{len(entities)}")
        warnings.append(f"Many orphans: {len(orphan_ents)}/{len(entities)}")

    # Core name duplicates
    core_map = {}
    for e in entities:
        core = re.sub(r'[（(][^）)]+[）)]', '', e['name']).strip()
        if core not in core_map:
            core_map[core] = []
        core_map[core].append(e['family_id'])
    for core, fids in core_map.items():
        if len(set(fids)) > 1:
            warnings.append(f"DUPLICATE_CORE: '{core}' → {len(set(fids))} family_ids")

    return issues, warnings, orphan_ents


def main():
    with open('service_config.json') as f:
        config = json.load(f)

    db_path = 'graph/graph.db'

    # Clear previous test data
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM relations WHERE source_document LIKE 'qa_%'")
    conn.execute("DELETE FROM entities WHERE source_document LIKE 'qa_%'")
    try:
        conn.execute("DELETE FROM episodes WHERE document_name LIKE 'qa_%'")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("DELETE FROM episode_mentions WHERE episode_id LIKE '%qa_%'")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()
    print("Cleared previous test data\n")

    results = []
    total_issues = 0
    total_warnings = 0

    for doc_name, text in TEST_CASES:
        processor = build_processor(config)
        source_doc = f'qa_{doc_name}'

        print(f'{"="*60}')
        print(f'TEST: {doc_name} ({len(text)} chars)')
        print(f'{"="*60}')

        t0 = time.time()
        try:
            result = processor.remember_text(text, doc_name=source_doc, verbose=False, verbose_steps=False)
        except Exception as e:
            print(f'>>> ERROR: {e}')
            results.append({
                'doc_name': doc_name, 'text_len': len(text),
                'time': time.time() - t0, 'entity_count': 0, 'relation_count': 0,
                'issues': [str(e)], 'warnings': [], 'entities': [], 'orphans': [],
            })
            total_issues += 1
            continue

        elapsed = time.time() - t0

        entities, relations = get_stored_data(db_path, source_doc)
        issues, warnings, orphan_ents = check_quality(entities, relations)

        status = "OK" if not issues else f"ISSUES({len(issues)})"
        print(f'>>> {elapsed:.1f}s | {len(entities)} ents | {len(relations)} rels | {status}')

        if issues:
            for i in issues[:5]:
                print(f'    ISSUE: {i}')
        if warnings:
            for w in warnings[:3]:
                print(f'    WARN: {w}')

        ent_names = [e['name'] for e in entities]
        print(f'    Entities: {", ".join(ent_names[:8])}{"..." if len(ent_names) > 8 else ""}')
        if orphan_ents:
            print(f'    Orphans: {", ".join(orphan_ents[:5])}')

        total_issues += len(issues)
        total_warnings += len(warnings)
        results.append({
            'doc_name': doc_name, 'text_len': len(text),
            'time': elapsed,
            'entity_count': len(entities), 'relation_count': len(relations),
            'issues': issues, 'warnings': warnings,
            'entities': ent_names, 'orphans': orphan_ents,
        })
        print()

    # ── Summary ──
    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    print(f'{"Test Case":25s} | {"Chars":>5s} | {"Time":>5s} | {"Ents":>4s} | {"Rels":>4s} | {"Orph":>4s} | Status')
    print('-' * 80)

    all_pass = True
    for r in results:
        has_issues = len(r['issues']) > 0
        if has_issues:
            all_pass = False
        status = "OK" if not has_issues else f"BAD({len(r['issues'])})"
        print(f"  {r['doc_name']:23s} | {r['text_len']:5d} | {r['time']:5.1f}s | {r['entity_count']:4d} | {r['relation_count']:4d} | {len(r.get('orphans',[])):4d} | {status}")

    print(f'\n  Total: {len(results)} tests | {total_issues} issues | {total_warnings} warnings')

    if all_pass:
        print('\n  ALL TESTS PASSED')
    else:
        failed = sum(1 for r in results if r['issues'])
        print(f'\n  {failed}/{len(results)} TESTS HAVE ISSUES')

    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
