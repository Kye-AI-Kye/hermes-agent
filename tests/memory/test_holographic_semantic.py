"""Integration: neural layer wired into holographic store + retrieval.

Hermetic — uses a controllable fake embedder (no network). Proves the WIRING:
a fact sharing NO keywords with the query still surfaces via vector candidates,
and that embedder=None preserves the original (keyword/HRR) behaviour.
"""
# store/retrieval are loaded onto sys.path at plugin-load time (below), not as a
# package, so static analysis can't resolve them. The fake embedders mirror the
# real interface (e.g. available() -> bool, input_type kwarg) so overrides that
# return a narrower literal are fine here.
# pyright: reportMissingImports=false, reportIncompatibleMethodOverride=false
import os
import sys

import pytest

np = pytest.importorskip("numpy")

_HG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "plugins", "memory", "holographic")
)
if _HG not in sys.path:
    sys.path.insert(0, _HG)

from store import MemoryStore       # noqa: E402
from retrieval import FactRetriever  # noqa: E402


class FakeEmbedder:
    """Deterministic embedder: each text maps to a caller-chosen vector so we can
    control cosine similarity and test the wiring without a real model."""

    def __init__(self, vectors: dict):
        self._v = {k: np.asarray(v, dtype=np.float32) for k, v in vectors.items()}

    def available(self) -> bool:
        return True

    def embed_one(self, text, input_type="passage"):
        return self._v.get(text)

    def embed(self, texts, input_type="passage"):
        return [self._v.get(t) for t in texts]

    @staticmethod
    def to_bytes(v):
        return np.asarray(v, dtype=np.float32).tobytes()

    @staticmethod
    def from_bytes(b):
        return np.frombuffer(b, dtype=np.float32)

    @staticmethod
    def cosine(a, b):
        a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0 or nb == 0 or a.shape != b.shape:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


def test_semantic_match_with_zero_keyword_overlap(tmp_path):
    # Query and the relevant fact share NO tokens, so FTS5/Jaccard cannot find it.
    # Only the neural vector stage can surface it.
    QUERY = "qqqq wwww"
    RELEVANT = "zzzz xxxx"      # semantically near (we set vectors), no shared tokens
    DISTRACTOR = "pizza pasta"  # far
    vectors = {
        QUERY: [1.0, 0.0, 0.0],
        RELEVANT: [0.96, 0.28, 0.0],   # high cosine to query
        DISTRACTOR: [0.0, 0.0, 1.0],   # orthogonal
    }
    emb = FakeEmbedder(vectors)
    db = str(tmp_path / "m.db")
    store = MemoryStore(db_path=db, embedder=emb)
    store.add_fact(RELEVANT)
    store.add_fact(DISTRACTOR)
    retr = FactRetriever(store=store, embedder=emb, embed_weight=0.6)

    results = retr.search(QUERY, min_trust=0.0, limit=5)
    contents = [r["content"] for r in results]
    assert RELEVANT in contents, f"semantic match not surfaced; got {contents}"
    assert contents[0] == RELEVANT, f"semantic match not ranked #1; got {contents}"
    # results must be JSON-clean (no raw vector bytes leaked)
    assert all("embedding" not in r and "hrr_vector" not in r for r in results)


def test_embedding_column_persisted(tmp_path):
    emb = FakeEmbedder({"hello world": [0.1, 0.2, 0.3]})
    store = MemoryStore(db_path=str(tmp_path / "m.db"), embedder=emb)
    fid = store.add_fact("hello world")
    row = store._conn.execute("SELECT embedding FROM facts WHERE fact_id=?", (fid,)).fetchone()
    assert row["embedding"] is not None
    assert np.allclose(emb.from_bytes(row["embedding"]), [0.1, 0.2, 0.3])


def test_backward_compat_no_embedder(tmp_path):
    # No embedder → original keyword/HRR behaviour, no crash, embedding stays NULL.
    store = MemoryStore(db_path=str(tmp_path / "m.db"))  # embedder=None
    fid = store.add_fact("the project uses python and sqlite")
    retr = FactRetriever(store=store)  # embed_weight defaults 0.0
    assert retr.embed_weight == 0.0
    results = retr.search("python", min_trust=0.0, limit=5)
    assert any("python" in r["content"] for r in results)
    row = store._conn.execute("SELECT embedding FROM facts WHERE fact_id=?", (fid,)).fetchone()
    assert row["embedding"] is None


def test_reindex_backfills_null_embeddings(tmp_path):
    # Facts added without an embedder have NULL embeddings; attaching an embedder
    # and calling reindex_embeddings() should fill exactly those.
    store = MemoryStore(db_path=str(tmp_path / "m.db"))  # embedder=None
    store.add_fact("alpha beta")
    store.add_fact("gamma delta")
    nulls = store._conn.execute(
        "SELECT count(*) FROM facts WHERE embedding IS NULL"
    ).fetchone()[0]
    assert nulls == 2

    store.embedder = FakeEmbedder({"alpha beta": [1.0, 0.0], "gamma delta": [0.0, 1.0]})
    n = store.reindex_embeddings()
    assert n == 2
    remaining = store._conn.execute(
        "SELECT count(*) FROM facts WHERE embedding IS NULL"
    ).fetchone()[0]
    assert remaining == 0
    # second call is a no-op — nothing left to backfill
    assert store.reindex_embeddings() == 0


def test_reindex_force_reembeds_all(tmp_path):
    emb = FakeEmbedder({"alpha beta": [1.0, 0.0]})
    store = MemoryStore(db_path=str(tmp_path / "m.db"), embedder=emb)
    fid = store.add_fact("alpha beta")
    before = store._conn.execute(
        "SELECT embedding FROM facts WHERE fact_id=?", (fid,)
    ).fetchone()["embedding"]

    # Simulate a model change: same text now maps to a different vector.
    store.embedder = FakeEmbedder({"alpha beta": [0.0, 1.0]})
    # Without force, nothing is NULL so nothing changes.
    assert store.reindex_embeddings() == 0
    # With force, the stale vector is replaced.
    assert store.reindex_embeddings(force=True) == 1
    after = store._conn.execute(
        "SELECT embedding FROM facts WHERE fact_id=?", (fid,)
    ).fetchone()["embedding"]
    assert after != before
    assert np.allclose(emb.from_bytes(after), [0.0, 1.0])


def test_reindex_noop_without_embedder(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "m.db"))  # embedder=None
    store.add_fact("alpha beta")
    assert store.reindex_embeddings() == 0
    assert store.reindex_embeddings(force=True) == 0


def test_unavailable_embedder_disables_neural(tmp_path):
    class Down(FakeEmbedder):
        def available(self) -> bool:
            return False
    store = MemoryStore(db_path=str(tmp_path / "m.db"), embedder=Down({}))
    store.add_fact("alpha beta")
    retr = FactRetriever(store=store, embedder=Down({}), embed_weight=0.6)
    assert retr.embed_weight == 0.0  # auto-disabled
    # still works via keyword
    assert any("alpha" in r["content"] for r in retr.search("alpha", min_trust=0.0))
