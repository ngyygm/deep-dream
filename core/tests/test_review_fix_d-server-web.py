"""d-server-web 组 code review 修复的回归测试。

f10-cors-dead-allowlist：
  _cors_preflight 之前对一切非严格同源（scheme/host/port 全等）的 Origin 直接 403，
  使同批加入的 _is_allowed_origin（localhost 系任意端口）沦为死代码，
  也击穿了 api.py 顶部注释承诺的 Vite 跨端口开发流。
  修复后：同源或 allowlist 命中放行，其余 403；
  _reject_cross_site_mutations 对跨站变更请求的拒绝语义保持不变。
"""

from core.server.api import create_app

_VITE_ORIGIN = "http://localhost:5173"  # 允许清单命中的跨端口开发 Origin
_EVIL_ORIGIN = "http://evil.example.com"  # 陌生 Origin


def _make_client():
    app = create_app(
        object(), config={"host": "127.0.0.1", "auth": {"enabled": False}}
    )
    return app.test_client()


def test_preflight_from_localhost_allowlist_is_allowed():
    """Vite 端口的 CORS 预检不应被 before_request 403 拦截。"""
    client = _make_client()
    resp = client.open(
        "/api/v1/health", method="OPTIONS", headers={"Origin": _VITE_ORIGIN}
    )
    assert resp.status_code == 204


def test_get_from_localhost_allowlist_passes_with_cors_header():
    """allowlist 命中的 GET 正常到达 handler，且 CORS 头真正生效（不再是死代码）。"""
    client = _make_client()
    resp = client.get("/api/v1/health", headers={"Origin": _VITE_ORIGIN})
    assert resp.status_code == 200
    assert resp.headers.get("Access-Control-Allow-Origin") == _VITE_ORIGIN


def test_unfamiliar_origin_is_rejected():
    """陌生 Origin 的普通请求与预检都要 403。"""
    client = _make_client()
    resp = client.get("/api/v1/health", headers={"Origin": _EVIL_ORIGIN})
    assert resp.status_code == 403
    assert resp.get_json()["error"] == "Cross-origin request denied"
    preflight = client.open(
        "/api/v1/health", method="OPTIONS", headers={"Origin": _EVIL_ORIGIN}
    )
    assert preflight.status_code == 403


def test_request_without_origin_passes_normally():
    """同源请求/无 Origin 请求（非浏览器客户端）不受影响。"""
    client = _make_client()
    assert client.get("/api/v1/health").status_code == 200


def test_cross_site_mutation_guard_semantics_preserved():
    """_reject_cross_site_mutations 的 CSRF 语义：allowlist 例外、陌生来源仍拒。

    localhost:5173 属于 allowlist——跨端口开发流（Vite 5173 → API 16200）
    的 POST 与 CORS 读腿口径一致地放行；陌生 Origin 的变更请求与
    Sec-Fetch-Site 标记的跨站写入仍被显式拒绝。
    """
    client = _make_client()
    # allowlist 命中的跨端口变更请求放行（到达 handler，非 403 拦截）
    resp = client.post(
        "/api/v1/find",
        headers={"Origin": _VITE_ORIGIN},
        json={"query": "x"},
    )
    assert resp.status_code != 403

    # 陌生 Origin 的变更请求仍被拒（第一层 preflight 即拦截）
    resp_evil = client.post(
        "/api/v1/find",
        headers={"Origin": _EVIL_ORIGIN},
        json={"query": "x"},
    )
    assert resp_evil.status_code == 403
    assert "Cross-origin" in resp_evil.get_json()["error"]

    # Sec-Fetch-Site 标记的跨站写入（不带 allowlist Origin）同样被拒
    resp2 = client.post(
        "/api/v1/find",
        headers={"Sec-Fetch-Site": "cross-site"},
        json={"query": "x"},
    )
    assert resp2.status_code == 403
    assert resp2.get_json()["error"] == "Cross-origin mutation denied"


def test_extra_allowed_origins_env_escape_hatch(monkeypatch):
    """DEEP_DREAM_ALLOWED_ORIGINS：反向代理部署的显式放行逃生门。"""
    monkeypatch.setenv(
        "DEEP_DREAM_ALLOWED_ORIGINS",
        "https://dd.example.com,proxy.internal",
    )
    client = _make_client()

    # 完整 origin 精确匹配（scheme+host+port）
    ok = client.get("/api/v1/health", headers={"Origin": "https://dd.example.com"})
    assert ok.status_code == 200
    assert ok.headers.get("Access-Control-Allow-Origin") == "https://dd.example.com"
    mutation = client.post(
        "/api/v1/find", headers={"Origin": "https://dd.example.com"}, json={"query": "x"}
    )
    assert mutation.status_code != 403

    # 裸主机名：任意 scheme/端口
    bare = client.get("/api/v1/health", headers={"Origin": "http://proxy.internal:8080"})
    assert bare.status_code == 200

    # 近似但不完全匹配的 origin 不放行
    near = client.get("/api/v1/health", headers={"Origin": "https://dd.example.org"})
    assert near.status_code == 403
    wrong_port = client.get(
        "/api/v1/health", headers={"Origin": "http://dd.example.com:9999"}
    )
    assert wrong_port.status_code == 403
