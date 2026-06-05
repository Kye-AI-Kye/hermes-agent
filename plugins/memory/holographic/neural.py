"""Optional neural-embedding layer for the holographic memory provider.

Adds true semantic recall (synonyms / paraphrase / concept match) on top of
holographic's existing FTS5 + Jaccard + HRR scoring. Mirrors the HRR-optional
pattern: everything degrades gracefully to a no-op when the embedding backend
is unavailable (no key, no httpx/numpy, or API error), so it can never break
existing keyword/HRR recall.

Default backend: NVIDIA NIM `nvidia/nv-embedqa-e5-v5` (1024-dim, OpenAI-compatible
/v1/embeddings, free tier). nv-embedqa is an *asymmetric* retrieval embedder:
stored facts are encoded as `passage`, search queries as `query`.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

# Lazy/optional deps — import failures must never crash the memory provider.
try:
    import numpy as _np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    import httpx as _httpx
    _HAS_HTTPX = True
except Exception:  # pragma: no cover
    _httpx = None  # type: ignore[assignment]
    _HAS_HTTPX = False


_DEFAULTS = {
    "base_url": "https://integrate.api.nvidia.com/v1",
    "model": "nvidia/nv-embedqa-e5-v5",
    "key_env": "NVIDIA_NIM_API_KEY",
    "dim": 1024,
    "timeout": 30.0,
    "batch_size": 32,
}


class NeuralEmbedder:
    """Thin OpenAI-compatible /v1/embeddings client with graceful fallback."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = {**_DEFAULTS, **(config or {})}
        self.base_url = str(cfg["base_url"]).rstrip("/")
        self.model = str(cfg["model"])
        self.key_env = str(cfg["key_env"])
        self.dim = int(cfg["dim"])
        self.timeout = float(cfg["timeout"])
        self.batch_size = int(cfg["batch_size"])
        # Allow an explicit key (tests / non-env wiring) to override the env var.
        self._explicit_key = cfg.get("api_key") or ""

    # ------------------------------------------------------------------
    def _key(self) -> str:
        return self._explicit_key or os.environ.get(self.key_env, "")

    def available(self) -> bool:
        """True only if we can actually produce embeddings."""
        return bool(_HAS_NUMPY and _HAS_HTTPX and self._key())

    # ------------------------------------------------------------------
    def embed(self, texts: "list[str]", input_type: str = "passage") -> "list | None":
        """Embed texts. input_type is 'passage' (stored facts) or 'query'.

        Returns a list of float32 numpy vectors (one per input), or ``None`` if
        the backend is unavailable or every batch failed. Partial failure raises
        so callers don't silently store misaligned vectors.
        """
        if not texts:
            return []
        if not self.available():
            return None

        key = self._key()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # urllib/default UAs get Cloudflare-1010 blocked on integrate.api.nvidia.com.
            "User-Agent": "hermes-holographic-neural/1.0",
        }
        out: list = []
        try:
            with _httpx.Client(timeout=self.timeout) as client:
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i : i + self.batch_size]
                    body = {
                        "model": self.model,
                        "input": batch,
                        "input_type": input_type,
                        "encoding_format": "float",
                        "truncate": "END",
                    }
                    r = client.post(f"{self.base_url}/embeddings", headers=headers, json=body)
                    if r.status_code >= 400:
                        return None
                    data = r.json().get("data") or []
                    if len(data) != len(batch):
                        return None
                    # NIM returns data already index-ordered, but sort defensively.
                    data = sorted(data, key=lambda d: d.get("index", 0))
                    for d in data:
                        vec = _np.asarray(d["embedding"], dtype=_np.float32)
                        out.append(vec)
        except Exception:
            return None
        return out

    def embed_one(self, text: str, input_type: str = "query") -> "object | None":
        res = self.embed([text], input_type=input_type)
        if not res:
            return None
        return res[0]

    # ------------------------------------------------------------------
    @staticmethod
    def to_bytes(vec) -> bytes:
        return _np.asarray(vec, dtype=_np.float32).tobytes()

    @staticmethod
    def from_bytes(b: bytes):
        return _np.frombuffer(b, dtype=_np.float32)

    @staticmethod
    def cosine(a, b) -> float:
        """Cosine similarity in [-1, 1]; 0.0 on degenerate input."""
        a = _np.asarray(a, dtype=_np.float32)
        b = _np.asarray(b, dtype=_np.float32)
        na = float(_np.linalg.norm(a))
        nb = float(_np.linalg.norm(b))
        if na == 0.0 or nb == 0.0 or a.shape != b.shape:
            return 0.0
        return float(_np.dot(a, b) / (na * nb))
