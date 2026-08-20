"""EmbeddingClient 远程 OpenAI 兼容 /v1/embeddings 模式（P6.2）。

不加载本地 sentence-transformers 模型、不发真实网络请求。
"""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import core.storage.embedding as emb_mod
from core.storage.embedding import EmbeddingClient


class _StubEmbeddings:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def create(self, model, input):
        self.calls.append((model, list(input)))
        if self.fail:
            raise RuntimeError("endpoint down")
        # 每文本一个 4 维向量，值由文本长度决定（可区分不同输入）
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(len(t)), 1.0, 0.0, 0.0]) for t in input]
        )


class _StubOpenAI:
    last = None

    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.embeddings = _StubEmbeddings()
        _StubOpenAI.last = self


@pytest.fixture()
def stub_openai():
    with patch.object(emb_mod.openai, "OpenAI", _StubOpenAI):
        _StubOpenAI.last = None
        yield _StubOpenAI


def _remote_client():
    return EmbeddingClient(
        api_key="sk-test", api_base="http://127.0.0.1:9999/v1",
        model_name="remote-emb")


def test_remote_mode_skips_local_model(stub_openai):
    c = _remote_client()
    assert c.model is None
    assert c._remote_client is not None
    assert c.is_available() is True
    # OpenAI 客户端按配置构造
    assert stub_openai.last.base_url == "http://127.0.0.1:9999/v1"


def test_remote_encode_shape_and_model(stub_openai):
    c = _remote_client()
    out = c.encode(["hello", "x"])
    assert out.shape == (2, 4)
    assert out.dtype == np.float32
    model, texts = stub_openai.last.embeddings.calls[0]
    assert model == "remote-emb"
    assert texts == ["hello", "x"]


def test_remote_single_text(stub_openai):
    c = _remote_client()
    v = c.encode("hello")
    assert v.shape == (4,)


def test_remote_cache_hits_on_second_encode(stub_openai):
    c = _remote_client()
    c.encode(["hello", "x"])
    n_calls = len(stub_openai.last.embeddings.calls)
    c.encode(["hello", "x"])
    assert len(stub_openai.last.embeddings.calls) == n_calls  # 全部命中缓存
    assert c.cache_stats()["hits"] >= 2


def test_remote_error_returns_none(stub_openai):
    c = _remote_client()
    stub_openai.last.embeddings.fail = True
    assert c.encode_uncached(["boom"]) is None
    assert c.encode(["boom"]) is None  # 无缓存可用


def test_local_mode_still_default(stub_openai):
    """未配置 api_key/api_base 时不构造远程客户端（保持本地路径）。"""
    c = EmbeddingClient.__new__(EmbeddingClient)  # 不触发 _init_model / 模型下载
    c._remote_client = None
    c.model = None
    assert c.is_available() is False
    assert stub_openai.last is None
