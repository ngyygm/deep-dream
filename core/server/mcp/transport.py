#!/usr/bin/env python3
"""
Transport layer for Deep Dream MCP Server.

Protocol I/O (stdio with Content-Length / NDJSON auto-detection) and HTTP proxy
helpers that forward requests to the upstream Deep Dream REST API.
"""

import json
import os
import sys
import contextlib
import httpx
from datetime import datetime

# ── Stdio setup ───────────────────────────────────────────────────────────
sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', buffering=0)
sys.stdin = os.fdopen(sys.stdin.fileno(), 'rb', buffering=0)

DEBUG_LOG = "/tmp/deep-dream-mcp-debug.log"
BASE_URL = os.environ.get("DEEP_DREAM_BASE_URL", "http://localhost:16200")
_DEFAULT_GRAPH_ID = os.environ.get("DEEP_DREAM_GRAPH_ID", "default")
_active_graph_id = _DEFAULT_GRAPH_ID  # runtime switchable

_use_ndjson = False


# ── Protocol I/O ──────────────────────────────────────────────────────────

def debug_log(msg):
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(f"{datetime.now().isoformat()} {msg}\n")
    except Exception:
        pass


def send_response(resp):
    global _use_ndjson
    data = json.dumps(resp, ensure_ascii=False, separators=(',', ':')).encode()
    if _use_ndjson:
        sys.stdout.write(data + b'\n')
    else:
        sys.stdout.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
    sys.stdout.flush()


def read_message():
    global _use_ndjson
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.decode().rstrip('\r\n')
    if line.lower().startswith("content-length:"):
        n = int(line.split(':', 1)[1].strip())
        while True:
            h = sys.stdin.readline()
            if not h:
                return None
            if h.decode().rstrip('\r\n') == '':
                break
        body = sys.stdin.read(n)
        return json.loads(body.decode())
    elif line.startswith('{') or line.startswith('['):
        _use_ndjson = True
        return json.loads(line)
    return None


# ── HTTP helpers ──────────────────────────────────────────────────────────

_client = httpx.Client(timeout=60.0)


def _resolve_graph_id(args):
    """Resolve graph_id for a tool call.

    Priority: per-call args['graph_id'] > _active_graph_id (set by switch_graph) > env default.
    """
    gid = args.get("graph_id")
    if gid and isinstance(gid, str) and gid.strip():
        return gid.strip()
    return _active_graph_id


# ── Per-call graph context ────────────────────────────────────────────────
# When a tool is dispatched, we set _current_call_graph_id so that _url() can
# pick it up without every handler needing to pass graph_id explicitly.

_current_call_graph_id = _DEFAULT_GRAPH_ID


@contextlib.contextmanager
def _graph_context(args):
    """Context manager: set per-call graph_id for the duration of a tool dispatch."""
    global _current_call_graph_id
    old = _current_call_graph_id
    _current_call_graph_id = _resolve_graph_id(args)
    try:
        yield _current_call_graph_id
    finally:
        _current_call_graph_id = old


def _url(path, graph_id=None, **qp):
    """Build URL with graph_id query param.

    Priority: explicit graph_id > per-call context > active graph > env default.
    """
    gid = graph_id or _current_call_graph_id or _active_graph_id
    sep = '&' if '?' in path else '?'
    params = [f"graph_id={gid}"]
    for k, v in qp.items():
        if v is not None:
            params.append(f"{k}={v}")
    return f"{BASE_URL}{path}{sep}{'&'.join(params)}"


def _get(path, **qp):
    r = _client.get(_url(path, **qp))
    return r.json(), r.status_code


def _post(path, body=None, **qp):
    r = _client.post(_url(path, **qp), json=body or {})
    return r.json(), r.status_code


def _put(path, body=None, **qp):
    r = _client.put(_url(path, **qp), json=body or {})
    return r.json(), r.status_code


def _delete(path, body=None, **qp):
    kw = {}
    if body:
        kw["json"] = body
    r = _client.delete(_url(path, **qp), **kw)
    return r.json(), r.status_code
