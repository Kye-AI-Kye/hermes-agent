"""Unit tests for the holographic neural-embedding layer (hermetic — NIM mocked)."""
import os
import sys
import types

import pytest

np = pytest.importorskip("numpy")

# The holographic plugin dir is loaded onto sys.path at plugin-load time, not as
# a `plugins.memory.holographic` package; import the module the same way here.
_HG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "plugins", "memory", "holographic")
)
if _HG not in sys.path:
    sys.path.insert(0, _HG)
import neural  # noqa: E402


# ---- pure helpers (no network) ------------------------------------------------
def test_cosine_identical_orthogonal_opposite():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert neural.NeuralEmbedder.cosine(a, a) == pytest.approx(1.0, abs=1e-5)
    assert neural.NeuralEmbedder.cosine(a, b) == pytest.approx(0.0, abs=1e-5)
    assert neural.NeuralEmbedder.cosine(a, -a) == pytest.approx(-1.0, abs=1e-5)


def test_cosine_degenerate_returns_zero():
    z = np.zeros(3, dtype=np.float32)
    assert neural.NeuralEmbedder.cosine(z, z) == 0.0
    # mismatched shape -> 0.0, never raises
    assert neural.NeuralEmbedder.cosine(np.ones(3), np.ones(4)) == 0.0


def test_bytes_roundtrip():
    v = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    b = neural.NeuralEmbedder.to_bytes(v)
    out = neural.NeuralEmbedder.from_bytes(b)
    assert np.allclose(v, out)


# ---- availability / graceful fallback ----------------------------------------
def test_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
    e = neural.NeuralEmbedder({"key_env": "NVIDIA_NIM_API_KEY"})
    assert e.available() is False
    assert e.embed(["hi"], "passage") is None  # never raises, returns None


def test_embed_empty_returns_empty(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "x")
    e = neural.NeuralEmbedder()
    assert e.embed([], "passage") == []


# ---- mocked NIM HTTP ----------------------------------------------------------
class _FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, status=200, vectors=None):
        self._status = status
        self._vectors = vectors or [[0.1, 0.2, 0.3]]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, headers=None, json=None):
        n = len(json["input"])
        data = [{"index": i, "embedding": self._vectors[i % len(self._vectors)]} for i in range(n)]
        return _FakeResp(self._status, {"data": data})


def _patch_httpx(monkeypatch, client):
    fake = types.SimpleNamespace(Client=lambda *a, **k: client)
    monkeypatch.setattr(neural, "_httpx", fake)
    monkeypatch.setattr(neural, "_HAS_HTTPX", True)


def test_embed_success_mocked(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "x")
    _patch_httpx(monkeypatch, _FakeClient(200, [[1.0, 0.0], [0.0, 1.0]]))
    e = neural.NeuralEmbedder({"batch_size": 32})  # single batch so fake indexes line up
    vecs = e.embed(["alpha", "beta"], "passage")
    assert vecs is not None and len(vecs) == 2
    assert np.allclose(vecs[0], [1.0, 0.0])
    assert np.allclose(vecs[1], [0.0, 1.0])


def test_embed_http_error_returns_none(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "x")
    _patch_httpx(monkeypatch, _FakeClient(429, [[0.1]]))
    e = neural.NeuralEmbedder()
    assert e.embed(["x"], "passage") is None


def test_embed_count_mismatch_returns_none(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "x")

    class _BadClient(_FakeClient):
        def post(self, url, headers=None, json=None):
            return _FakeResp(200, {"data": []})  # zero vectors for non-empty input

    _patch_httpx(monkeypatch, _BadClient())
    e = neural.NeuralEmbedder()
    assert e.embed(["x"], "passage") is None
