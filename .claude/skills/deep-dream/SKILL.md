---
name: deep-dream
description: >
  Use when working with the local Deep-Dream memory system: querying remembered
  documents, episodes, concepts, relations, provenance, semantic recall, graph
  traversal, Obsidian/Markdown vault indexes, or writing/repairing memories.
  Treat local files as the source of truth and use Deep-Dream's SQLite/API graph
  only when document search needs episode/concept/relation support.
---

# Deep-Dream Document-First Steward

Deep-Dream is a local memory system with three evidence levels:

```text
Memory library / graph folder
  -> Document file
  -> Episode span
  -> Concept/relation graph
```

Use Deep-Dream as a document-first local memory steward. Raw documents are the
source of truth; episodes and concepts are enhancement layers for alignment,
semantic recall, graph traversal, and provenance.

## Explicit Invocation Minimum

If the user explicitly names `$deep-dream`, links this skill, or says they are
testing Deep-Dream, do not answer from general knowledge alone.

Minimum required workflow:

1. Run `deep-dream doctor` or `python -m core.cli doctor` from the likely
   Deep-Dream project/install directory.
2. If the CLI is not available, try the local API health/graphs endpoints.
3. For content questions, run `deep-dream docs search "<query>"` for direct
   document evidence. For abstract/fuzzy questions, think of 3-8 likely
   related terms from the user's wording and domain context, then pass them to
   `deep-dream explore "<query>" --terms "term1,term2,term3"`.
4. If documents are hit, inspect the reported readable path and line evidence.
5. If the task needs alignment, semantic recall, cross-document recurrence,
   relation evidence, or graph neighbors, enter the episode/concept layer.
6. In the final answer, state which layers were actually used: raw files,
   episodes, concepts, semantic search, relations, SQL, API, or fallback.

Never silently skip Deep-Dream after this skill is invoked. If discovery fails,
report the failure and ask for a config path, storage root, graph id, or document
root.

## Operating Rules

- Start with readable local documents whenever possible: use `deep-dream docs search`, `deep-dream docs grep`, `rg`, file search, and direct reading first.
- Use episodes when a document hit needs source span, line/offset mapping, or local context.
- Use concepts/relations when raw file search is insufficient: fuzzy recall, alias alignment, cross-document links, entity disambiguation, event chains, relation evidence, provenance, or repair.
- Treat concepts and relation summaries as candidates. Verify important claims against `source_text` or the original file span before answering.
- Prefer document paths and line/span evidence in final answers. Do not cite a concept summary as final evidence when raw text is available.
- Do not infer an event from co-occurrence alone. Two concepts in one episode may indicate mention, narration, contrast, or indirect context.
- Prefer the `deep-dream` CLI for agent work. Use read-only SQL for inspection. Use CLI/API service paths for writes, indexing, deletion, repair, and task control.
- Never assume the graph id is `default`. Discover graphs first unless the user specified one.

## Mental Model

Documents are the durable source. They may be managed by Deep-Dream or external files indexed in place.

Episodes are evidence-bearing spans. They bridge a document location to extracted memory content, mentions, and asserted relations.

Concepts are cross-episode semantic objects: entities, relations, document nodes, or episode nodes. They help locate and align evidence, but they are not evidence by themselves.

## Quick Workflow

1. Identify the likely graph and document set.
2. Run `deep-dream doctor` when system state is unknown.
3. Search files directly with `deep-dream docs search`, `deep-dream docs grep`, `rg`, or read known paths.
4. If raw text answers the task, answer from raw text.
5. If the task needs semantic expansion, query episodes/concepts/relations.
6. Map graph candidates back to document paths and spans.
7. Verify against raw text or `v_episodes.source_text`.
8. If graph coverage is incomplete, inspect document integrity before trusting misses.

Use semantic search for themes, fuzzy descriptions, and "related but not exact wording" tasks. Use SQL and file search for exact names, counts, dates, quotations, and event ordering. For abstract Chinese concepts, the agent must generate context-specific expansion terms before deciding there are no raw document hits. Do not rely on hardcoded domain expansions.

## Preferred CLI

Use the CLI first when available:

```powershell
deep-dream doctor
deep-dream graph list
deep-dream docs roots
deep-dream docs list --graph TARGET_GRAPH
deep-dream docs search "keyword" --graph TARGET_GRAPH
deep-dream docs grep "regex" --graph TARGET_GRAPH
deep-dream docs map "C:\path\to\file.md" --graph TARGET_GRAPH
deep-dream episode from-file "C:\path\to\file.md" --line 120 --graph TARGET_GRAPH
deep-dream episode concepts EPISODE_VERSION_ID --graph TARGET_GRAPH
deep-dream concept search "query" --semantic --graph TARGET_GRAPH
deep-dream concept trace FAMILY_ID --graph TARGET_GRAPH
deep-dream concept neighbors FAMILY_ID --depth 2 --graph TARGET_GRAPH
deep-dream relation evidence CONCEPT_A CONCEPT_B --graph TARGET_GRAPH
deep-dream explore "abstract question or theme" --terms "agent-generated,related,terms" --graph TARGET_GRAPH
deep-dream sql --query "SELECT * FROM v_document_files LIMIT 20" --graph TARGET_GRAPH
```

If the command is not installed, try `python -m core.cli ...` from the
Deep-Dream repository or installed package directory.

## API Helper

Default base URL:

```python
import json
import urllib.request

BASE = "http://127.0.0.1:16200/api/v1"

def api(method, path, body=None):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
```

Discover graphs:

```python
for graph in api("GET", "/graphs")["data"]["graphs_info"]:
    print(
        graph["graph_id"],
        graph.get("name", ""),
        graph.get("document_count"),
        graph.get("entity_count"),
    )
```

Run read-only SQL:

```python
rows = api("POST", "/agent/sql", {
    "graph_id": "TARGET_GRAPH",
    "sql": "SELECT * FROM v_document_files ORDER BY processed_time DESC LIMIT 20",
    "params": {},
    "limit": 200,
})["data"]["rows"]
```

Allowed Agent SQL: `SELECT`, `WITH`, and `EXPLAIN QUERY PLAN SELECT/WITH`.

Read the full endpoint list only when needed: `references/api-reference.md`.

## Find Files First

Map graph documents to readable paths:

```sql
SELECT document_version_id, title, source_mode, read_path,
       absolute_path, managed_path, snapshot_path,
       byte_size, char_count, line_count,
       complete_windows, total_windows, missing_windows
FROM v_document_files
ORDER BY processed_time DESC
LIMIT 100;
```

Path selection:

- `source_mode = 'external'`: read `absolute_path`.
- `source_mode = 'managed'`: read the graph-local `managed_path`, resolved under the graph directory.
- If the readable file is missing, fall back to `snapshot_path`.
- If the user gives a local path, map it back to Deep-Dream with the "File path -> document" query below.

Use local search after discovering paths:

```powershell
rg "keyword|alias|exact phrase" "C:\path\to\documents"
```

## Use Graph When Needed

Use graph queries for:

- Alias/entity alignment: "Are X, Y, and Z the same person/project/concept?"
- Cross-document recurrence: "Where does this idea appear?"
- Evidence trails: "Which episodes support this relation?"
- Fuzzy recall: "Find scenes like..."
- Relationship expansion: "What is connected to this concept?"
- Coverage/debugging: "Why did search miss this?"

Stable views:

- `v_document_files`: document ids, readable paths, file integrity, source mode.
- `v_documents`: document versions and metadata.
- `v_episodes`: episode content, `source_text`, offsets, line ranges, source path.
- `v_mentions`: episode-to-concept mentions.
- `v_latest_concept`: latest concept version per family.
- `v_relation_edges`: relation concepts rendered as entity-to-entity edges with evidence.
- `v_concept_documents`: concept occurrence by document.
- `v_document_stats`: per-document counts.

Core edge meanings:

- `HAS_EPISODE`: document version -> episode version.
- `MENTIONS`: episode -> entity concept.
- `ASSERTS`: episode -> relation concept.
- `CONNECTS`: relation concept -> entity endpoint.
- `DOCUMENT_LINK`: explicit Markdown/Obsidian document link.

## Query Recipes

File path -> document:

```sql
SELECT *
FROM v_document_files
WHERE absolute_path = :path
   OR managed_path = :path
   OR snapshot_path = :path
   OR read_path = :path
LIMIT 5;
```

Document -> episodes with source text:

```sql
SELECT version_id, heading_path, start_offset, end_offset,
       line_start, line_end, source_path, source_text
FROM v_episodes
WHERE document_version_id = :doc_id
ORDER BY start_offset
LIMIT 100;
```

Episode -> concepts:

```sql
SELECT m.target_family_id, m.target_name, lc.role, lc.content, m.provenance
FROM v_mentions m
JOIN v_latest_concept lc
  ON lc.graph_id = m.graph_id
 AND lc.family_id = m.target_family_id
WHERE m.episode_version_id = :episode_id
ORDER BY m.target_name
LIMIT 100;
```

Concept -> source file spans:

```sql
SELECT d.title, d.read_path, ep.version_id AS episode_id,
       ep.start_offset, ep.end_offset, ep.line_start, ep.line_end,
       ep.source_text, m.provenance
FROM v_mentions m
JOIN v_episodes ep ON ep.version_id = m.episode_version_id
JOIN v_document_files d ON d.document_version_id = ep.document_version_id
WHERE m.target_family_id = :family_id
ORDER BY d.processed_time, ep.start_offset
LIMIT 50;
```

Two concepts -> relation evidence:

```sql
SELECT re.relation_family_id, re.relation_version_id,
       re.relation_name, re.relation_content,
       re.entity1_name, re.entity2_name,
       d.title, d.read_path, ep.line_start, ep.line_end, ep.source_text
FROM v_relation_edges re
JOIN v_episodes ep ON ep.version_id = re.episode_version_id
JOIN v_document_files d ON d.document_version_id = re.document_version_id
WHERE (re.entity1_family_id = :a AND re.entity2_family_id = :b)
   OR (re.entity1_family_id = :b AND re.entity2_family_id = :a)
ORDER BY d.title, ep.start_offset
LIMIT 50;
```

Concepts appearing across a target document set:

```sql
WITH target_docs AS (
  SELECT document_version_id, title
  FROM v_document_files
  WHERE title LIKE :title_pattern
),
hits AS (
  SELECT m.target_family_id AS family_id, lc.name,
         COUNT(DISTINCT ep.document_version_id) AS doc_count,
         COUNT(*) AS mention_count,
         GROUP_CONCAT(DISTINCT d.title) AS documents
  FROM v_mentions m
  JOIN v_episodes ep ON ep.version_id = m.episode_version_id
  JOIN target_docs d ON d.document_version_id = ep.document_version_id
  JOIN v_latest_concept lc
    ON lc.family_id = m.target_family_id
   AND lc.role = 'entity'
  GROUP BY m.target_family_id
)
SELECT *
FROM hits
WHERE doc_count = (SELECT COUNT(*) FROM target_docs)
ORDER BY mention_count DESC
LIMIT 50;
```

Document integrity:

```sql
SELECT title, read_path, complete_windows, total_windows, missing_windows
FROM v_document_files
WHERE title LIKE :title
ORDER BY processed_time DESC;
```

If `missing_windows > 0`, graph extraction may be incomplete. Prefer raw files and consider repair.

## Semantic Recall

Use semantic search as candidate recall, then verify:

```powershell
deep-dream explore "abstract theme, scene, or fuzzy description" --graph TARGET_GRAPH
deep-dream concept search "abstract theme" --semantic --graph TARGET_GRAPH
```

API fallback:

```python
hits = api("POST", "/agent/semantic-search", {
    "graph_id": "TARGET_GRAPH",
    "query": "abstract theme, scene, or fuzzy description",
    "role": "entity",
    "top_k": 30,
    "threshold": 0.2,
})["data"]
```

Good mixed strategy:

1. Expand abstract queries into 3-8 related terms based on the user's actual question and current domain context.
2. Exact file search finds known names and phrases.
3. Semantic search finds nearby themes or unnamed scenes.
4. Episode/concept SQL maps candidates to documents and source spans.
5. Neighbor/relation expansion finds cross-document associations.
6. Raw file reading verifies final claims.

When `explore` returns `evidence_cards`, prefer them over long raw
`source_text` blocks for answer drafting. Use full source text only when a card
is ambiguous or the user asks for exact context.

## Writes, Indexing, And Repair

Use CLI/API service paths, not direct DB writes:

- `deep-dream remember --file path.md`
- `deep-dream remember --text "..."`
- `deep-dream vault index C:\path\to\vault`

- `POST /api/v1/remember`: ingest text or file.
- `POST /api/v1/vaults/index`: index an Obsidian/Markdown vault in place.
- `GET /api/v1/remember/tasks`: inspect ingestion tasks.
- `GET /api/v1/documents`: list indexed documents.
- `GET /api/v1/documents/<id>/content`: read indexed Markdown content.
- `POST /api/v1/documents/<id>/repair?graph_id=<graph_id>`: repair incomplete extraction.
- `DELETE /api/v1/documents/<id>`: delete a document version.

When indexing external Markdown/vault files, preserve their local paths. Deep-Dream should map to the files rather than forcing all notes into one managed folder.

## Answering Standard

When answering from Deep-Dream:

- State whether the answer came from raw files, graph evidence, or both.
- State whether `doctor`, `docs search`, `explore`, semantic search, SQL, or API was used.
- Include document titles/paths and line ranges when available.
- Separate verified claims from graph-suggested candidates.
- Mention coverage limitations if document integrity is incomplete or only semantic candidates were found.
