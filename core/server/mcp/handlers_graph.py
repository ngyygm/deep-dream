#!/usr/bin/env python3
"""
Graph management and system dashboard handlers for Deep Dream MCP Server.

Handles: list_graphs, create_graph, delete_graph, switch_graph, get_active_graph,
         system_dashboard, system_overview, system_graphs, system_tasks,
         system_logs, system_access_stats, list_docs, get_doc_content,
         get_entity_neighbors (Neo4j), communities.
"""

from .transport import (
    _get, _post, _delete,
    _active_graph_id, _DEFAULT_GRAPH_ID,
)
from .response_format import (
    _result, _hint, _inner,
    _compact_entity, _compact_relation, _compact_list,
)
from .dispatch_helpers import _arg, _req


# ── Graphs ────────────────────────────────────────────────────────────────

def list_graphs(args):
    data, code = _get("/api/v1/graphs")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        graphs_info = inner.get("graphs_info", [])
        graphs = inner.get("graphs", [])
        if isinstance(graphs_info, list) and graphs_info:
            active = _active_graph_id
            parts = []
            for g in graphs_info:
                if not isinstance(g, dict):
                    continue
                gid = g.get("graph_id", "?")
                marker = " (active)" if gid == active else ""
                name = g.get("name", "")
                ec = g.get("entity_count", "?")
                rc = g.get("relation_count", "?")
                label = f"'{gid}'{marker}"
                if name:
                    label += f" [{name}]"
                label += f" ({ec}E/{rc}R)"
                parts.append(label)
            hint = f"\n→ {len(parts)} graph(s): {', '.join(parts)}. Use switch_graph(graph_id='...') to change active graph."
            _hint(data, hint)
        elif isinstance(graphs, list) and graphs:
            active = _active_graph_id
            graph_list = ", ".join(f"'{g}'{' (active)' if g == active else ''}" for g in graphs)
            _hint(data, f"\n→ {len(graphs)} graph(s): {graph_list}. Use switch_graph(graph_id='...') to change active graph.")
    return _result(data, code)


def create_graph(args):
    body = {"graph_id": args["graph_id"]}
    if _arg(args, "name"):
        body["name"] = args["name"]
    if _arg(args, "description"):
        body["description"] = args["description"]
    data, code = _post("/api/v1/graphs", body)
    if code < 400:
        gid = args["graph_id"]
        hint = f"\n→ Graph '{gid}' created. Use switch_graph(graph_id='{gid}') to start using it."
        if isinstance(data, dict):
            _hint(data, hint)
    return _result(data, code)


def delete_graph(args):
    gid = _req(args, "graph_id")
    data, code = _delete(f"/api/v1/graphs/{gid}")
    if code < 400:
        # If deleted graph was active, switch back to default
        global _active_graph_id
        was_active = _active_graph_id == gid
        if was_active:
            _active_graph_id = _DEFAULT_GRAPH_ID
        hint = f"\n→ Graph '{gid}' deleted permanently."
        if was_active:
            hint += f" Active graph reset to '{_active_graph_id}'."
        else:
            hint += f" Active graph remains '{_active_graph_id}'."
        if isinstance(data, dict):
            _hint(data, hint)
    return _result(data, code)


def switch_graph(args):
    """Switch the active graph for all subsequent tool calls."""
    global _active_graph_id
    gid = _req(args, "graph_id")
    # Verify graph exists by listing
    data, code = _get("/api/v1/graphs")
    if code >= 400:
        return _result(data, code)
    inner = _inner(data) if isinstance(data, dict) else {}
    graphs = inner.get("graphs", [])
    if gid not in graphs:
        available = ", ".join(f"'{g}'" for g in graphs) if graphs else "none"
        return _result({"error": f"Graph '{gid}' does not exist. Available: {available}. Use create_graph first."}, 404)
    old = _active_graph_id
    _active_graph_id = gid
    # Get summary of the new graph (pass graph_id explicitly since _current_call_graph_id may be stale)
    summary_data, _ = _get("/api/v1/find/graph-summary", graph_id=gid)
    summary_inner = _inner(summary_data) if isinstance(summary_data, dict) else {}
    entity_count = summary_inner.get("entity_count", "?")
    relation_count = summary_inner.get("relation_count", "?")
    return {"content": [{"type": "text", "text": f"Switched active graph: '{old}' → '{gid}'\nGraph '{gid}': {entity_count} entities, {relation_count} relations."}]}


def get_active_graph(args):
    """Get the currently active graph ID and its summary."""
    gid = _active_graph_id
    summary_data, _ = _get("/api/v1/find/graph-summary", graph_id=gid)
    summary_inner = _inner(summary_data) if isinstance(summary_data, dict) else {}
    entity_count = summary_inner.get("entity_count", "?")
    relation_count = summary_inner.get("relation_count", "?")
    return {"content": [{"type": "text", "text": f"Active graph: '{gid}' (env default: '{_DEFAULT_GRAPH_ID}')\nEntities: {entity_count}, Relations: {relation_count}"}]}


# ── System ────────────────────────────────────────────────────────────────

def system_dashboard(args):
    return _result(*_get("/api/v1/system/dashboard"))


def system_overview(args):
    return _result(*_get("/api/v1/system/overview"))


def system_graphs(args):
    return _result(*_get("/api/v1/system/graphs"))


def system_tasks(args):
    return _result(*_get("/api/v1/system/tasks"))


def system_logs(args):
    qp = {}
    if _arg(args, "level"):
        qp["level"] = args["level"]
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    return _result(*_get("/api/v1/system/logs", **qp))


def system_access_stats(args):
    return _result(*_get("/api/v1/system/access-stats"))


# ── Docs ──────────────────────────────────────────────────────────────────

def list_docs(args):
    data, code = _get("/api/v1/docs")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        docs = inner.get("docs", inner.get("documents", []))
        if isinstance(docs, list) and docs:
            _hint(data, f"\n→ {len(docs)} documents. Use get_doc_content(filename='...') to read a specific document.")
        elif isinstance(docs, list) and not docs:
            _hint(data, "\n→ No documents stored. Documents are created automatically when using remember with a source_document parameter.")
    return _result(data, code)


def get_doc_content(args):
    return _result(*_get(f"/api/v1/docs/{args['filename']}"))


# ── Neo4j ─────────────────────────────────────────────────────────────────

def get_entity_neighbors(args):
    qp = {}
    if _arg(args, "direction"):
        qp["direction"] = args["direction"]
    if _arg(args, "limit"):
        qp["limit"] = str(args["limit"])
    data, code = _get(f"/api/v1/find/entities/{args['uuid']}/neighbors", **qp)
    if code < 400 and isinstance(data, dict):
        _hint(data, "\n→ Neo4j neighbors. For family_id-based access, use traverse_graph or entity_profile instead.")
    return _result(data, code)


# ── Communities ───────────────────────────────────────────────────────────

def detect_communities(args):
    body = {}
    if _arg(args, "algorithm"):
        body["algorithm"] = args["algorithm"]
    if _arg(args, "resolution"):
        body["resolution"] = float(args["resolution"])
    data, code = _post("/api/v1/communities/detect", body)
    if code < 400:
        hint = "\n→ Detection complete. Use list_communities to see all communities, then get_community(cid='...') to inspect members."
        if isinstance(data, dict):
            _hint(data, hint)
    return _result(data, code)


def list_communities(args):
    data, code = _get("/api/v1/communities")
    if code < 400 and isinstance(data, dict):
        inner = _inner(data)
        comms = inner.get("communities", [])
        if comms and isinstance(comms, list):
            # Compact communities: keep essential fields only
            for i, c in enumerate(comms):
                if isinstance(c, dict):
                    comms[i] = {k: v for k, v in c.items() if k in ("community_id", "cid", "name", "member_count", "internal_relation_count", "summary")}
            if comms:
                hint = f"\n→ {len(comms)} communities detected. Use get_community(cid='...') to inspect members."
                _hint(data, hint)
    return _result(data, code)


def get_community(args):
    cid = args["cid"]
    data, code = _get(f"/api/v1/communities/{cid}")
    if code < 400 and isinstance(data, dict):
        hint = f"\n→ Use get_community_graph(cid='{cid}') for subgraph visualization."
        _hint(data, hint)
    return _result(data, code)


def get_community_graph(args):
    data, code = _get(f"/api/v1/communities/{args['cid']}/graph")
    if code < 400:
        data = _compact_list(data, _compact_entity, "entities")
        data = _compact_list(data, _compact_relation, "relations")
        inner = _inner(data)
        entities = inner.get("entities", [])
        relations = inner.get("relations", [])
        if isinstance(entities, list) and isinstance(relations, list):
            _hint(data, f"\n→ Community subgraph: {len(entities)} entities, {len(relations)} relations.")
    return _result(data, code)


def clear_communities(args):
    data, code = _delete("/api/v1/communities")
    if code < 400:
        _hint(data, "\n→ Community labels cleared. Entities and relations are NOT deleted. Run detect_communities to rebuild.")
    return _result(data, code)


def register_handlers(tool_map):
    tool_map["list_graphs"] = list_graphs
    tool_map["create_graph"] = create_graph
    tool_map["delete_graph"] = delete_graph
    tool_map["switch_graph"] = switch_graph
    tool_map["get_active_graph"] = get_active_graph
    tool_map["system_dashboard"] = system_dashboard
    tool_map["system_overview"] = system_overview
    tool_map["system_graphs"] = system_graphs
    tool_map["system_tasks"] = system_tasks
    tool_map["system_logs"] = system_logs
    tool_map["system_access_stats"] = system_access_stats
    tool_map["list_docs"] = list_docs
    tool_map["get_doc_content"] = get_doc_content
    tool_map["get_entity_neighbors"] = get_entity_neighbors
    tool_map["detect_communities"] = detect_communities
    tool_map["list_communities"] = list_communities
    tool_map["get_community"] = get_community
    tool_map["get_community_graph"] = get_community_graph
    tool_map["clear_communities"] = clear_communities
