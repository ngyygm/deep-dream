#!/usr/bin/env python3
"""GAP-D: entity alignment vs ground-truth reconciliation (offline, no LLM).

Ground truth (v2, injective-name matching):
  The MAB AR-000 haystack is built from SQuAD/Wikipedia articles; the
  benchmark ships a DBpedia URI pool (entity2id.json, 31,161 entities) whose
  URI slugs are canonical entity names. A library mention's ground-truth
  identity is asserted ONLY when its surface form's content-word set equals
  exactly ONE pool entity's name (parenthetical stripped). Ambiguous names
  ("Victoria" -> film/state/person) and non-pool surfaces are excluded and
  reported as coverage, never guessed. An earlier draft tried to resolve the
  subject of each "Document N:" block via Wikipedia-abstract overlap; that
  premise is wrong (blocks are mid-article paragraphs, not lead sections) and
  the code path was removed.

Metrics per frozen library (Base v1 = commit 73d9c5e vs Align+ v2 = 8fa9651),
on the injective subset:
  fragmentation   a true entity whose mentions land in >=2 terminal families
                  (terminal = entity_redirects chain followed to fixpoint)
  over-merge      a terminal family holding mentions of >=2 true entities
  cross-name frag fragmented entity whose families carry >1 distinct
                  canonical name (what the paper's duplicate-family proxy saw)

Outputs: gapd_results.json + a markdown table on stdout.
Usage (repo root): python3 research/paper/rebuttal/gapd_alignment_audit.py
"""
import json
import re
import sqlite3
import urllib.parse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "research/.benchmark_runs"
E2I = ROOT / "research/.benchmark_data/memoryagentbench/entity2id.json"
OUT = ROOT / "research/paper/rebuttal/gapd_results.json"

ENGINES = {
    "Base(73d9c5e)": RUNS / "memoryagentbench-kimik3-sample-v1",
    "Align+(8fa9651)": RUNS / "memoryagentbench-kimik3-sample-v2",
}

_punct = re.compile(r"[()\"'“”‘’.,;:!?—–\-]")
STOP = {"of", "de", "du", "the", "and", "la", "le", "von", "van", "der", "den",
        "di", "da", "do", "dos", "das", "a", "an", "in", "on", "at", "to"}


def norm(s: str) -> str:
    s = urllib.parse.unquote(s)
    s = s.replace("_", " ").replace("%20", " ").replace("+", " ")
    s = _punct.sub(" ", s.lower())
    return " ".join(s.split())


def content_words(s: str) -> frozenset[str]:
    return frozenset(w for w in norm(s).split() if w not in STOP)


def slug_of(uri: str) -> str:
    return uri.strip("<>").rsplit("/resource/", 1)[-1]


def build_injective_pool(entity2id: dict) -> dict[frozenset[str], str]:
    """content-word set -> URI, kept only when the set maps to exactly one URI.

    Both the full slug and its parenthetical-stripped base register as names
    of the same URI ("Victoria_(Australia)" also answers to "Victoria"), but a
    word set claimed by two different URIs is ambiguous and dropped.
    """
    claims: dict[frozenset[str], set[str]] = defaultdict(set)
    for uri in entity2id:
        base = slug_of(uri)
        for variant in {base, base.split("(")[0]}:
            cw = content_words(variant)
            if len(cw) >= 1:
                claims[cw].add(uri)
    return {cw: next(iter(uris)) for cw, uris in claims.items() if len(uris) == 1}


def audit_library(lib_dir: Path, pool: dict[frozenset[str], str],
                  ambiguous_names: set[frozenset[str]]) -> dict:
    conn = sqlite3.connect(f"file:{lib_dir / 'library.db'}?mode=ro", uri=True)
    # terminal family after redirect chains
    red = dict(conn.execute(
        "SELECT source_family_id, target_family_id FROM entity_redirects"))

    def resolve_f(f, seen=()):
        if f in term:
            return term[f]
        if f in seen:
            return f
        term[f] = resolve_f(red[f], (*seen, f)) if f in red else f
        return term[f]

    term: dict[str, str] = {}
    fams = {r[0] for r in conn.execute("SELECT entity_family_id FROM entity_families")}
    fams |= set(red) | set(red.values())
    for f in fams:
        resolve_f(f)
    fnames: dict[str, set] = defaultdict(set)
    for fid, name in conn.execute(
            "SELECT entity_family_id, canonical_name FROM entity_families"):
        fnames[fid].add(norm(name))

    mentions = list(conn.execute(
        "SELECT entity_family_id, episode_id, surface_text, start_offset "
        "FROM entity_mentions"))
    conn.close()

    subj_fams: dict[str, set] = defaultdict(set)   # true entity -> terminal families
    fam_subjs: dict[str, set] = defaultdict(set)   # terminal family -> true entities
    # name-level view over ALL mentions (no pool needed): surface name ->
    # terminal families + whether every surface occurrence is entity-cased.
    name_fams: dict[frozenset[str], set] = defaultdict(set)
    name_entity_cased: dict[frozenset[str], bool] = defaultdict(bool)
    n_ambig = n_unmatched = n_resolved = 0
    for fid, _eid, surface, _off in mentions:
        cw = content_words(surface or "")
        if not cw:
            n_unmatched += 1
            continue
        tf = term.get(fid, fid)
        name_fams[cw].add(tf)
        # entity-likeness: capitalized first alphabetic token (or all-caps),
        # as written in the source; lowercase common nouns drop out.
        alpha = [t for t in (surface or "").split() if t[:1].isalpha()]
        if alpha and all(t[:1].isupper() or t.isupper() for t in alpha):
            name_entity_cased[cw] = True
        uri = pool.get(cw)
        if uri is None:
            n_ambig += 1
            continue
        n_resolved += 1
        subj_fams[uri].add(tf)
        fam_subjs[tf].add(uri)

    fragmented = {u: fs for u, fs in subj_fams.items() if len(fs) >= 2}
    crossname = 0
    for u, fs in fragmented.items():
        names = set()
        for f in fs:
            names |= fnames.get(f, set())
        if len(names) > 1:
            crossname += 1
    overmerged = {f: ss for f, ss in fam_subjs.items() if len(ss) >= 2}
    # name-level fragmentation stratified by pool ambiguity. Unambiguous
    # entity names (pool single-claim) that span >=2 terminal families are
    # fragmentation ERRORS; ambiguous names may span families legitimately
    # (disambiguation) and are reported separately.
    name_split = {"unambiguous": 0, "ambiguous": 0, "unpooled_entity": 0,
                  "unpooled_other": 0}
    name_total = dict(name_split)
    for cw, fams in name_fams.items():
        if cw in pool:
            bucket = "unambiguous"
        elif cw in ambiguous_names:
            bucket = "ambiguous"
        elif name_entity_cased.get(cw):
            bucket = "unpooled_entity"
        else:
            bucket = "unpooled_other"
        name_total[bucket] += 1
        if len(fams) >= 2:
            name_split[bucket] += 1
    return {
        "mentions_total": len(mentions),
        "resolved_mentions": n_resolved,
        "ambiguous_or_unmatched": n_ambig + n_unmatched,
        "true_entities": len(subj_fams),
        "fragmented": len(fragmented),
        "fragmented_crossname": crossname,
        "overmerged_families": len(overmerged),
        "overmerged_pairs": sum(len(s) for s in overmerged.values()),
        "coverage_pct": round(100 * n_resolved / max(1, len(mentions)), 1),
        "name_total": name_total,
        "name_split": name_split,
    }


def main():
    entity2id = json.loads(E2I.read_text())
    claims: dict[frozenset[str], set[str]] = defaultdict(set)
    for uri in entity2id:
        base = slug_of(uri)
        for variant in {base, base.split("(")[0]}:
            cw = content_words(variant)
            if cw:
                claims[cw].add(uri)
    pool = {cw: next(iter(us)) for cw, us in claims.items() if len(us) == 1}
    ambiguous = {cw for cw, us in claims.items() if len(us) > 1}
    print(f"injective name sets: {len(pool)}  ambiguous: {len(ambiguous)} "
          f"(from {len(entity2id)} URIs)")

    results = {e: {} for e in ENGINES}
    for engine, run_dir in ENGINES.items():
        for lib in sorted((run_dir / "libraries").iterdir()):
            if not (lib.name.startswith("mab-ar-") and (lib / "library.db").exists()):
                continue
            r = audit_library(lib, pool, ambiguous)
            results[engine][lib.name] = r
            print(f"{engine:18s} {lib.name:26s} mentions={r['mentions_total']:5d} "
                  f"resolved={r['resolved_mentions']:5d}({r['coverage_pct']}%) "
                  f"true={r['true_entities']:4d} frag={r['fragmented']:3d}"
                  f"(x={r['fragmented_crossname']:2d}) ovm={r['overmerged_families']:3d}",
                  flush=True)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    keys = ("mentions_total", "resolved_mentions", "true_entities", "fragmented",
            "fragmented_crossname", "overmerged_families", "overmerged_pairs")
    for engine, libs in results.items():
        tot = {k: sum(v[k] for v in libs.values()) for k in keys}
        names_tot = {b: sum(v["name_total"][b] for v in libs.values())
                     for b in ("unambiguous", "ambiguous", "unpooled_entity", "unpooled_other")}
        names_split = {b: sum(v["name_split"][b] for v in libs.values())
                       for b in names_tot}
        frag_pct = 100 * tot["fragmented"] / max(1, tot["true_entities"])
        cov = 100 * tot["resolved_mentions"] / max(1, tot["mentions_total"])
        ns = names_split
        nt = names_tot
        print(f"\n## {engine}  ({len(libs)} AR libraries, pool coverage {cov:.1f}%)")
        print("| mentions | injective-resolved | true entities | fragmented | "
              "cross-name frag | over-merged fams (pairs) |")
        print("|---|---|---|---|---|---|")
        print(f"| {tot['mentions_total']} | {tot['resolved_mentions']} | "
              f"{tot['true_entities']} | {tot['fragmented']} ({frag_pct:.1f}%) | "
              f"{tot['fragmented_crossname']} | "
              f"{tot['overmerged_families']} ({tot['overmerged_pairs']}) |")
        print("\nname-level fragmentation (same surface name spanning >=2 terminal "
              "families), stratified:")
        print("| name bucket | names | split | split % |")
        print("|---|---|---|---|")
        for b, label in (("unambiguous", "pool-unambiguous entity name"),
                         ("ambiguous", "pool-ambiguous name (may be legit disambiguation)"),
                         ("unpooled_entity", "entity-cased name not in pool"),
                         ("unpooled_other", "non-entity-cased (lowercase) name")):
            pct = 100 * ns[b] / max(1, nt[b])
            print(f"| {label} | {nt[b]} | {ns[b]} | {pct:.1f}% |")


if __name__ == "__main__":
    main()
