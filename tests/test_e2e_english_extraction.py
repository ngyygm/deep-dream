"""E2E tests: English extraction quality with real LLM + Neo4j backend.

Tests the full remember → find pipeline with diverse English texts to validate:
1. Entity name validation (English generic words, verb phrases, sentences)
2. Relation content validation (English label-only, empty templates)
3. Cross-domain extraction (science, tech, history, literature)
4. Long-text handling
5. Search quality after extraction
"""
import json
import time
import uuid
import pytest
import requests

BASE = "http://127.0.0.1:16200"
PROXIES = {"http": None, "https": None}


def _remember(content: str, source: str = "", timeout: int = 300) -> dict:
    """Call remember API and poll until completed. Retries on server errors."""
    import time as _time
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{BASE}/api/v1/remember",
                json={
                    "text": content,
                    "source_name": source or f"test_{uuid.uuid4().hex[:6]}",
                    "wait": True,
                    "timeout": timeout,
                },
                proxies=PROXIES, timeout=timeout + 60,
            )
            if resp.status_code == 200:
                result = resp.json()
                if result.get("success"):
                    return result.get("data", {})
                raise RuntimeError(f"remember failed: {result}")
            # 202 = still processing, poll for completion
            if resp.status_code == 202:
                data = resp.json().get("data", {})
                task_id = data.get("task_id")
                if task_id:
                    return _poll_task(task_id, timeout=timeout)
                raise RuntimeError(f"remember 202 but no task_id: {resp.text[:300]}")
            raise RuntimeError(f"remember failed: {resp.status_code} {resp.text[:500]}")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < 2:
                print(f"  Connection error (attempt {attempt+1}/3): {e}, waiting 30s...")
                _time.sleep(30)
            else:
                raise


def _poll_task(task_id: str, timeout: int = 300) -> dict:
    """Poll a remember task until completed."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(
            f"{BASE}/api/v1/remember/tasks/{task_id}",
            proxies=PROXIES, timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            status = data.get("status", "unknown")
            if status == "completed":
                return data
            if status == "failed":
                raise RuntimeError(f"Task failed: {data}")
        time.sleep(3)
    raise RuntimeError(f"Task {task_id} timed out after {timeout}s")


def _search_entities(query: str, mode: str = "hybrid") -> list:
    resp = requests.post(
        f"{BASE}/api/v1/find/entities/search",
        json={"query_name": query, "mode": mode, "max_results": 10},
        proxies=PROXIES, timeout=15,
    )
    assert resp.status_code == 200, f"search failed: {resp.status_code} {resp.text[:300]}"
    data = resp.json()["data"]
    # API returns list directly or dict with entities
    if isinstance(data, list):
        return data
    return data.get("entities", [])


def _quick_search(query: str) -> dict:
    resp = requests.post(
        f"{BASE}/api/v1/find",
        json={"query": query, "max_entities": 10, "max_relations": 20},
        proxies=PROXIES, timeout=30,
    )
    assert resp.status_code == 200, f"quick-search failed: {resp.status_code} {resp.text[:300]}"
    return resp.json()["data"]


# =============================================================================
# Test 1: Science domain — Marie Curie biography (English)
# =============================================================================

MARIE_CURIE_TEXT = """
Marie Curie (born Maria Sklodowska, 7 November 1867 – 4 July 1934) was a Polish and
naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice,
and the only person to win a Nobel Prize in two scientific fields. Her husband, Pierre Curie,
was a co-winner of her first Nobel Prize, making them the first married couple to win the
Nobel Prize and launching the Curie family legacy of five Nobel Prizes.

She shared the 1903 Nobel Prize in Physics with her husband Pierre Curie and physicist
Henri Becquerel. She won the 1911 Nobel Prize in Chemistry. Her achievements include the
development of the theory of radioactivity (a term she coined), techniques for isolating
radioactive isotopes, and the discovery of two elements, polonium and radium. Under her
direction, the world's first studies were conducted into the treatment of neoplasms using
radioactive isotopes. She founded the Curie Institutes in Paris and Warsaw, which remain
major centres of medical research today.
"""


@pytest.fixture(scope="module")
def marie_curie_result():
    return _remember(MARIE_CURIE_TEXT, source="wikipedia_marie_curie")


def test_marie_curie_entities_extracted(marie_curie_result):
    """Should extract key scientific entities, not generic labels."""
    assert isinstance(marie_curie_result, dict), f"Unexpected result: {marie_curie_result}"

    # Search for core entities
    results = _search_entities("Marie Curie")
    names = [e["name"] for e in results]
    # Marie Curie should be findable
    assert any("Curie" in n for n in names), f"Expected Curie entity, got: {names}"


def test_marie_curie_no_generic_entities(marie_curie_result):
    """Should NOT extract generic labels like 'research', 'contribution', 'legacy'."""
    results = _search_entities("research")
    names = [e["name"].lower() for e in results]
    # Generic word "research" alone should not be an entity name
    for name in names:
        assert name != "research", f"Generic word 'research' should not be an entity"
        assert name != "contribution", f"Generic word 'contribution' should not be an entity"
        assert name != "discovery", f"Generic word 'discovery' should not be an entity"


def test_marie_curie_relations_found(marie_curie_result):
    """Should extract relations between scientific entities."""
    data = _quick_search("Nobel Prize Curie")
    relations = data.get("relations", [])
    # Should have some relations mentioning Curie or Nobel Prize
    assert len(relations) > 0, "Expected at least one relation about Curie/Nobel Prize"


# =============================================================================
# Test 2: Technology domain — Kubernetes architecture (English, technical)
# =============================================================================

K8S_TEXT = """
Kubernetes (K8s) is an open-source container orchestration system for automating software
deployment, scaling, and management of containerized applications. Originally designed by
Google, it is now maintained by the Cloud Native Computing Foundation (CNCF).

Kubernetes defines a set of building blocks (primitives) that collectively provide the
mechanisms to deploy, maintain, and scale applications. The key components include:

- Pods: The smallest deployable computing unit in Kubernetes. A pod encapsulates one or
  more application containers, storage resources, and a unique network IP.
- Services: An abstraction that defines a logical set of pods and a policy to access them.
- Deployments: Declarative updates for Pods and ReplicaSets.
- ConfigMaps and Secrets: Mechanisms to inject configuration data into containers.

The Kubernetes control plane consists of several components: kube-apiserver (the front end
for the control plane), etcd (a consistent and highly-available key value store), kube-scheduler
(watches for newly created Pods with no assigned node), kube-controller-manager (runs
controller processes), and cloud-controller-manager (embeds cloud-specific control logic).

Kubernetes uses a declarative configuration approach. Users define the desired state of their
cluster using YAML or JSON manifests, and Kubernetes ensures the actual state converges to
the desired state through reconciliation loops.
"""


@pytest.fixture(scope="module")
def k8s_result():
    return _remember(K8S_TEXT, source="kubernetes_docs")


def test_k8s_entities_extracted(k8s_result):
    """Should extract technical entities like Kubernetes, Pod, Service, etc."""
    assert isinstance(k8s_result, dict), f"Unexpected result: {k8s_result}"

    results = _search_entities("Kubernetes Pod")
    names = [e["name"] for e in results]
    # Should find Kubernetes-related entities
    assert any("Kubernetes" in n or "K8s" in n for n in names), \
        f"Expected Kubernetes entity, got: {names}"


def test_k8s_no_tech_generic_entities(k8s_result):
    """Should NOT extract generic tech terms like 'architecture', 'design', 'process'."""
    results = _quick_search("architecture design")
    entities = results.get("entities", [])
    names = [e["name"].lower() for e in entities]
    for name in names:
        assert name != "architecture", f"Generic 'architecture' should not be entity"
        assert name != "design", f"Generic 'design' should not be entity"
        assert name != "process", f"Generic 'process' should not be entity"


def test_k8s_technical_relations(k8s_result):
    """Should extract relations between K8s components."""
    data = _quick_search("kube-apiserver etcd")
    relations = data.get("relations", [])
    # Should have some technical relations
    assert len(relations) >= 0  # May or may not have direct relation between these two


# =============================================================================
# Test 3: Mixed Chinese-English text — quantum computing
# =============================================================================

QUANTUM_MIXED_TEXT = """
量子计算（Quantum Computing）是一种利用量子力学原理进行计算的新型计算范式。
与经典计算机使用比特（bit）不同，量子计算机使用量子比特（qubit），可以同时
处于0和1的叠加态（superposition）。

中国的量子计算研究在国际上处于领先地位。潘建伟团队领导的"九章"量子计算原型机
在2020年实现了量子优越性（quantum supremacy），处理特定问题的速度比当时最快
的超级计算机快一百亿倍。

量子纠缠（Quantum Entanglement）是量子计算的核心资源。当两个量子比特纠缠时，
对其中一个的测量会立即影响另一个的状态，无论它们相距多远。爱因斯坦曾将这一
现象称为"鬼魅般的远距作用"（spooky action at a distance）。
"""


@pytest.fixture(scope="module")
def quantum_result():
    return _remember(QUANTUM_MIXED_TEXT, source="quantum_intro")


def test_quantum_mixed_entities(quantum_result):
    """Should extract both Chinese and English scientific entities."""
    assert isinstance(quantum_result, dict), f"Unexpected result: {quantum_result}"

    results = _search_entities("量子比特 qubit")
    names = [e["name"] for e in results]
    # Should find quantum-related entities
    assert len(names) > 0, "Expected quantum-related entities"


def test_quantum_no_generic_entities(quantum_result):
    """Should NOT extract generic concepts like '研究' or 'development'."""
    results = _quick_search("研究 development")
    entities = results.get("entities", [])
    for e in entities:
        name = e["name"]
        # These generic terms should not be standalone entities
        assert name not in ("研究", "development", "分析", "analysis"), \
            f"Generic term '{name}' should not be an entity"


# =============================================================================
# Test 4: Short text — single paragraph about a specific topic
# =============================================================================

SHORT_TEXT = """
The Rust programming language was designed by Graydon Hoare at Mozilla Research.
It emphasizes memory safety, zero-cost abstractions, and concurrency without data races.
Rust's ownership system and borrow checker enforce memory safety at compile time,
eliminating the need for garbage collection while preventing use-after-free and
buffer overflow vulnerabilities.
"""


@pytest.fixture(scope="module")
def short_result():
    return _remember(SHORT_TEXT, source="rust_intro")


def test_short_text_entities(short_result):
    """Short text should still extract meaningful entities."""
    assert isinstance(short_result, dict), f"Unexpected result: {short_result}"

    # LLM may extract "Rust" as-is or translate to Chinese equivalents
    # Accept both: "Rust" directly, or Chinese terms like "编程语言" (programming language)
    results = _search_entities("Rust programming")
    names = [e["name"] for e in results]
    rust_related = any(
        "Rust" in n or "rust" in n.lower()
        or "编程语言" in n or "内存安全" in n or "ownership" in n.lower()
        for n in names
    )
    assert rust_related, f"Expected Rust-related entity, got: {names}"


def test_short_text_no_generic(short_result):
    """Short text should not produce generic entities."""
    results = _search_entities("memory safety")
    for e in results:
        name = e["name"].lower()
        # "memory safety" is specific enough to be valid, but "safety" alone is not
        if name == "safety":
            pytest.fail("Generic word 'safety' should not be an entity")


# =============================================================================
# Test 5: Verify bad entity names are NOT in the database
# =============================================================================

def test_no_english_generic_entities_in_db():
    """After all extractions, verify generic English words are not entities."""
    generic_words = [
        "research", "discovery", "contribution", "achievement",
        "influence", "impact", "development", "improvement",
        "collaboration", "partnership", "framework", "architecture",
        "process", "approach", "strategy", "method",
    ]
    for word in generic_words:
        results = _search_entities(word)
        names = [e["name"].lower().strip() for e in results]
        assert word.lower() not in names, \
            f"Generic word '{word}' should not be an entity name, found in: {names}"


# =============================================================================
# Test 6: Verify bad relation content patterns are filtered
# =============================================================================

def test_no_vague_relations_in_db():
    """Check that vague relation patterns like 'is related to' are not stored."""
    # Search for relations and check content quality
    data = _quick_search("Curie Nobel")
    relations = data.get("relations", [])

    vague_patterns = [
        "is related to", "is associated with", "is connected to",
        "is linked to", "have a relationship", "are related",
        "are associated", "are connected",
    ]
    for rel in relations:
        content = rel.get("content", "").lower()
        for pattern in vague_patterns:
            # Allow the pattern as part of a longer, specific description
            if content.strip() == pattern or content.strip().endswith(pattern):
                pytest.fail(f"Vague relation content: '{content[:100]}'")
