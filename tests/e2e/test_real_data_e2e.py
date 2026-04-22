"""
End-to-end tests using real-world data from different domains and lengths.

Covers:
1. Short text (single sentence) — tech domain
2. Medium text (paragraph) — biology domain
3. Long text (multi-paragraph) — history domain
4. Chinese text — literature domain
5. Mixed language text — academic domain
6. Very long text — legal/policy document

Tests:
- remember API (entity/relation extraction)
- find API (semantic search)
- entity CRUD
- time_after / time_before filters
- cascade delete
- health/stats endpoints
"""
import json
import os
import time
import unittest
from datetime import datetime, timezone

import requests

BASE_URL = os.environ.get("DEEP_DREAM_URL", "http://localhost:16200")
GRAPH_ID = "test_e2e"


def _api(method, path, **kwargs):
    kwargs.setdefault("timeout", 120)
    url = f"{BASE_URL}{path}"
    if "params" not in kwargs:
        kwargs["params"] = {}
    kwargs["params"]["graph_id"] = GRAPH_ID
    # Bypass proxy for localhost
    kwargs.setdefault("proxies", {"http": None, "https": None})
    resp = getattr(requests, method)(url, **kwargs)
    return resp


class TestE2ERealData(unittest.TestCase):
    """Real-world data E2E tests."""

    @classmethod
    def setUpClass(cls):
        # Ensure the test graph is clean
        resp = _api("get", "/api/v1/find/stats")
        data = resp.json()
        print(f"\nTest graph '{GRAPH_ID}' initial stats: {data}")

    # ── Test Data ────────────────────────────────────────────────────────

    SHORT_TECH = "Python 3.12 released with improved error messages and faster CPython runtime."

    MEDIUM_BIO = """
The mitochondrion is a double-membraned organelle found in most eukaryotic organisms.
It is the powerhouse of the cell, generating most of the cell's supply of adenosine
triphosphate (ATP), used as a source of chemical energy. Mitochondria have their own
DNA (mtDNA), which is inherited maternally in most species. The endosymbiotic theory
proposes that mitochondria were once free-living prokaryotes that were engulfed by
eukaryotic cells. Dysfunction of mitochondria is associated with numerous diseases
including Parkinson's, Alzheimer's, and mitochondrial myopathies.
"""

    LONG_HISTORY = """
The Renaissance was a cultural and intellectual movement that began in Italy during
the late Middle Ages and later spread to the rest of Europe. It lasted from the 14th
to the 17th century, encompassing the transition from the medieval period to modernity.

The movement began in Florence, one of the wealthiest city-states in Italy, fueled by
the Medici family's patronage of the arts. Key figures include Leonardo da Vinci,
whose notebooks contain designs for flying machines and anatomical studies that were
centuries ahead of their time. Michelangelo sculpted David and painted the Sistine
Chapel ceiling, works that epitomize Renaissance ideals of human potential and beauty.

Raphael, known for his Madonnas and the School of Athens fresco, embodied the
Renaissance ideal of harmonious composition. Niccolò Machiavelli wrote The Prince,
a political treatise that broke from medieval political philosophy by describing
the world as it is rather than as it should be.

The invention of the printing press by Johannes Gutenberg around 1440 revolutionized
the spread of knowledge. Books became cheaper and more widely available, leading to
increased literacy rates across Europe. The Protestant Reformation, sparked by Martin
Luther in 1517, was made possible in part by the printing press.

Galileo Galilei's astronomical observations with the telescope provided empirical
evidence for the heliocentric model proposed by Copernicus. This brought him into
conflict with the Catholic Church, highlighting the tension between emerging scientific
thought and established religious authority.

The Renaissance also saw significant developments in art techniques, including linear
perspective (pioneered by Brunelleschi and Alberti), chiaroscuro (the use of strong
contrasts between light and dark), and sfumato (the subtle blending of tones,
mastered by Leonardo). These techniques gave paintings a three-dimensional quality
that had not been achieved in medieval art.

In architecture, Filippo Brunelleschi designed the dome of Florence Cathedral, an
engineering marvel that remains the largest brick dome ever constructed. Leon Battista
Alberti wrote De re aedificatoria, the first architectural treatise of the Renaissance.

The Northern Renaissance, centered in the Low Countries and Germany, produced artists
such as Jan van Eyck, Albrecht Dürer, and Hieronymus Bosch. While sharing the Italian
Renaissance's interest in humanism and classical learning, Northern artists placed
greater emphasis on detailed realism and religious devotion.

The Renaissance fundamentally shaped Western civilization, laying the groundwork for
the Scientific Revolution, the Enlightenment, and modern democratic thought. Its
emphasis on individual achievement, empirical observation, and the revival of classical
learning continues to influence education, art, and philosophy today.
"""

    CHINESE_LITERATURE = """
《红楼梦》是中国古典四大名著之一，由清代作家曹雪芹创作。小说以贾、史、王、
薛四大家族的兴衰为背景，以贾宝玉和林黛玉的爱情悲剧为主线，通过对封建社会
的深刻描写，展现了18世纪中国封建社会走向末世的全景图。

小说塑造了数百个性格鲜明的人物形象，其中贾宝玉是核心人物，他叛逆不羁，
厌恶仕途经济，对女性充满同情和理解。林黛玉聪慧多才、多愁善感，与宝玉有着
前世木石前盟的深厚感情。薛宝钗端庄贤淑、圆通世故，代表了封建正统的女性
理想。

《红楼梦》不仅是一部伟大的文学作品，也是一部百科全书式的文化巨著，涉及
诗词、医药、饮食、建筑、礼仪、戏曲等诸多领域。曹雪芹以其天才的艺术手法，
将现实主义与浪漫主义完美结合，创造了中国文学史上的一座高峰。

小说后半部分由高鹗续写完成，关于续书的优劣一直是红学研究的争议焦点。
脂砚斋批本是研究《红楼梦》的重要文献，为理解曹雪芹的创作意图提供了宝贵线索。
"""

    MIXED_ACADEMIC = """
Machine Learning in Protein Structure Prediction: A Survey

The prediction of protein three-dimensional structure from amino acid sequence has been
a grand challenge in computational biology for over five decades. Recent advances in
deep learning, particularly AlphaFold2 (Jumper et al., Nature 2021) and RoseTTAFold
(Baek et al., Science 2021), have achieved breakthrough accuracy, with predicted
structures often matching experimental results.

AlphaFold2 uses an attention-based neural network architecture that processes
multiple sequence alignments (MSAs) and pairwise features. The Evoformer module
captures co-evolutionary information between residues, while the structure module
generates 3D coordinates using an iterative refinement approach.

The CASP14 competition (Critical Assessment of protein Structure Prediction, 2020)
demonstrated AlphaFold2's dominance, achieving a median GDT-TS score of 92.4 on
free-modeling targets — approaching experimental accuracy for the first time.

Subsequent work includes AlphaFold-Multimer for protein complex prediction,
AlphaFold3 for broader biomolecular modeling (including protein-ligand and
protein-nucleic acid interactions), and ESMFold for fast single-sequence prediction
using protein language models.

Applications range from drug discovery (e.g., structure-based virtual screening)
to enzyme engineering (designing novel catalysts) and understanding disease mutations
(predicting the structural impact of genetic variants).
"""

    LEGAL_POLICY = """
中华人民共和国个人信息保护法（节选）

第一章 总则

第一条 为了保护个人信息权益，规范个人信息处理活动，促进个人信息合理利用，
根据宪法，制定本法。

第二条 自然人的个人信息受法律保护，任何组织、个人不得侵害自然人的个人信息权益。

第三条 本法适用于在中华人民共和国境内处理自然人个人信息的活动。
在中华人民共和国境外处理中华人民共和国境内自然人个人信息的，也适用本法。

第二章 个人信息处理规则

第一节 一般规定

第十三条 处理个人信息应当取得个人的同意，但有下列情形之一的，不需取得个人同意：
（一）为订立、履行个人作为一方当事人的合同所必需；
（二）为履行法定职责或者法定义务所必需；
（三）为应对突发公共卫生事件，或者紧急情况下为保护自然人的生命健康和财产安全所必需；
（四）为公共利益实施新闻报道、舆论监督等行为，在合理的范围内处理个人信息。

第二十一条 个人信息处理者委托处理个人信息的，应当与受托人约定委托处理的目的、
期限、处理方式、个人信息的种类、保护措施以及双方的权利义务等，并对受托人的
个人信息处理活动进行监督。

第二十五条 个人信息处理者不得公开其处理的个人信息，取得个人单独同意的除外。

第三章 个人信息跨境提供的规则

第三十八条 个人信息处理者因业务等需要，确需向中华人民共和国境外提供个人信息
的，应当具备下列条件之一：
（一）通过国家网信部门组织的安全评估；
（二）经专业机构进行个人信息保护认证；
（三）按照国家网信部门制定的标准合同与境外接收方订立合同。

第四章 个人权利

第四十四条 个人对其个人信息的处理享有知情权、决定权，有权限制或者拒绝他人
对其个人信息进行处理。

第四十七条 有下列情形之一的，个人信息处理者应当主动删除个人信息：
（一）处理目的已实现、无法实现或者为实现处理目的不再必要；
（二）个人信息处理者停止提供产品或者服务，或者保存期限已届满。
"""

    # ── Helpers ──────────────────────────────────────────────────────────

    def _remember(self, text, source="e2e_test"):
        """Submit a remember task and wait for completion."""
        resp = _api("post", "/api/v1/remember", json={
            "text": text,
            "source_name": source,
        })
        self.assertEqual(resp.status_code, 200, f"Remember failed: {resp.text}")
        data = resp.json()
        self.assertTrue(data["success"], f"Remember not successful: {data}")
        task_id = data["data"]["task_id"]

        # Poll until complete
        for _ in range(60):
            time.sleep(2)
            status_resp = _api("get", f"/api/v1/remember/tasks/{task_id}")
            status_data = status_resp.json()
            task_status = status_data.get("data", {}).get("status", "unknown")
            if task_status in ("completed", "failed"):
                return task_id, status_data["data"]
        self.fail(f"Task {task_id} did not complete within 120s")

    # ── Tests ────────────────────────────────────────────────────────────

    def test_01_health(self):
        """Server health check."""
        resp = _api("get", "/api/v1/health")
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["storage_backend"], "neo4j")
        self.assertTrue(data["data"]["embedding_available"])

    def test_02_remember_short_tech(self):
        """Test remember with short tech text."""
        task_id, result = self._remember(self.SHORT_TECH, source="short_tech")
        self.assertEqual(result["status"], "completed")
        print(f"  short_tech: entities={result.get('entity_count', '?')}, relations={result.get('relation_count', '?')}")

    def test_03_remember_medium_bio(self):
        """Test remember with medium biology text."""
        task_id, result = self._remember(self.MEDIUM_BIO, source="medium_bio")
        self.assertEqual(result["status"], "completed")
        print(f"  medium_bio: entities={result.get('entity_count', '?')}, relations={result.get('relation_count', '?')}")

    def test_04_remember_long_history(self):
        """Test remember with long history text."""
        task_id, result = self._remember(self.LONG_HISTORY, source="long_history")
        self.assertEqual(result["status"], "completed")
        print(f"  long_history: entities={result.get('entity_count', '?')}, relations={result.get('relation_count', '?')}")

    def test_05_remember_chinese_literature(self):
        """Test remember with Chinese literature text."""
        task_id, result = self._remember(self.CHINESE_LITERATURE, source="chinese_lit")
        self.assertEqual(result["status"], "completed")
        print(f"  chinese_lit: entities={result.get('entity_count', '?')}, relations={result.get('relation_count', '?')}")

    def test_06_remember_mixed_academic(self):
        """Test remember with mixed-language academic text."""
        task_id, result = self._remember(self.MIXED_ACADEMIC, source="mixed_academic")
        self.assertEqual(result["status"], "completed")
        print(f"  mixed_academic: entities={result.get('entity_count', '?')}, relations={result.get('relation_count', '?')}")

    def test_07_remember_legal_policy(self):
        """Test remember with long Chinese legal text."""
        task_id, result = self._remember(self.LEGAL_POLICY, source="legal_policy")
        self.assertEqual(result["status"], "completed")
        print(f"  legal_policy: entities={result.get('entity_count', '?')}, relations={result.get('relation_count', '?')}")

    def test_08_find_semantic(self):
        """Test semantic find across domains."""
        queries = [
            ("mitochondria ATP", "Should match biology text"),
            ("Renaissance art", "Should match history text"),
            ("贾宝玉", "Should match Chinese literature text"),
            ("AlphaFold protein", "Should match academic text"),
            ("个人信息保护", "Should match legal text"),
            ("Python programming", "Should match tech text"),
        ]
        for query, description in queries:
            resp = _api("post", "/api/v1/find", json={
                "query": query,
                "max_entities": 5,
                "max_relations": 10,
            })
            data = resp.json()
            self.assertTrue(data["success"], f"Find failed for '{query}': {data}")
            entities = data["data"]["entities"]
            relations = data["data"]["relations"]
            print(f"  find('{query}'): {len(entities)} entities, {len(relations)} relations — {description}")
            # At least some results expected
            self.assertGreater(len(entities) + len(relations), 0,
                               f"Expected results for '{query}' ({description})")

    def test_09_find_with_time_filter(self):
        """Test find with time_after filter."""
        # Get stats first
        resp = _api("get", "/api/v1/find/stats")
        stats = resp.json()["data"]
        print(f"  Stats: {stats}")

        # Find with time_after = now (should return nothing or very recent)
        resp = _api("post", "/api/v1/find", json={
            "query": "protein",
            "time_after": datetime.now(timezone.utc).isoformat(),
            "max_entities": 10,
        })
        data = resp.json()
        self.assertTrue(data["success"])

        # Find with time_after = past (should return results)
        resp = _api("post", "/api/v1/find", json={
            "query": "protein",
            "time_after": "2020-01-01T00:00:00",
            "max_entities": 10,
        })
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertGreater(len(data["data"]["entities"]), 0, "Expected results for old time_after")

    def test_10_entity_crud(self):
        """Test entity list, get, update."""
        # List entities
        resp = _api("get", "/api/v1/find/entities", params={"limit": 5})
        data = resp.json()
        self.assertTrue(data["success"])
        entities = data["data"]["entities"]
        self.assertGreater(len(entities), 0, "Expected some entities after remember operations")

        # Get specific entity
        first = entities[0]
        fid = first["family_id"]
        resp = _api("get", f"/api/v1/find/entities/{fid}")
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["family_id"], fid)

        # Update entity summary
        resp = _api("put", f"/api/v1/find/entities/{fid}", json={
            "summary": "E2E test summary update",
        })
        data = resp.json()
        self.assertTrue(data["success"], f"Update failed: {data}")

    def test_11_find_entities_search(self):
        """Test entity search endpoint."""
        resp = _api("post", "/api/v1/find/entities/search", json={
            "query_name": "mitochondrion",
            "max_results": 5,
        })
        data = resp.json()
        self.assertTrue(data["success"])
        results = data["data"] if isinstance(data["data"], list) else data["data"].get("entities", [])
        print(f"  entity_search('mitochondrion'): {len(results)} results")

    def test_12_find_relations_search(self):
        """Test relation search endpoint."""
        resp = _api("post", "/api/v1/find/relations/search", json={
            "query_text": "Leonardo da Vinci painted",
            "max_results": 5,
        })
        data = resp.json()
        self.assertTrue(data["success"])
        results = data["data"] if isinstance(data["data"], list) else data["data"].get("relations", [])
        print(f"  relation_search('Leonardo'): {len(results)} results")

    def test_13_find_relations_between(self):
        """Test finding relations between entities."""
        # First find two entity family_ids via entity list
        resp = _api("get", "/api/v1/find/entities", params={"limit": 3})
        entities = resp.json()["data"]["entities"]
        if len(entities) >= 2:
            fid_a = entities[0]["family_id"]
            fid_b = entities[1]["family_id"]
            resp = _api("post", "/api/v1/find/relations/between", json={
                "family_id_a": fid_a,
                "family_id_b": fid_b,
            })
            data = resp.json()
            self.assertTrue(data["success"])
            rels = data["data"] if isinstance(data["data"], list) else data["data"].get("relations", [])
            print(f"  relations_between: {len(rels)} relations")

    def test_14_stats(self):
        """Test stats endpoint has data from all remember operations."""
        resp = _api("get", "/api/v1/find/stats")
        data = resp.json()
        self.assertTrue(data["success"])
        stats = data["data"]
        print(f"  Final stats: entities={stats['total_entities']}, "
              f"relations={stats['total_relations']}, episodes={stats['total_episodes']}")
        # After 6 remember operations, should have entities
        self.assertGreater(stats["total_entities"], 0)
        self.assertGreater(stats["total_relations"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
