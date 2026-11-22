import json
import sqlite3

import numpy as np
import pytest

from conftest import FakeEncoder, build_fake_embeddings
from recall_data.build import run
from retrieval.bm25 import BM25Index, tokenize
from retrieval.corpus import CorpusRecord, build_search_text, corpus_fingerprint, load_corpus
from retrieval.embed import DEFAULT_DB, DEFAULT_OUT
from retrieval.loader import build_engine
from retrieval.search import SearchEngine, SearchUnavailableError, rank
from retrieval.semantic import (
    EmbeddingIndex,
    StaleIndexError,
    load_embeddings,
    save_embeddings,
)


def _record(recall_id: int, title: str, issue: str | None = None) -> CorpusRecord:
    return CorpusRecord(
        recall_id=recall_id, title=title, category="Test", issue=issue,
        url=f"https://example.test/{recall_id}",
        search_text=build_search_text(title, None, issue, "Test"),
    )


CORPUS = [
    _record(10, "Brand A peanut butter recalled due to Salmonella"),
    _record(20, "Children's hooded sweatshirts with drawstrings", "Strangulation hazard"),
    _record(30, "Brand B cheese recalled due to Listeria monocytogenes"),
    _record(40, "Portable phone charger may overheat", "Fire hazard"),
]


@pytest.fixture
def fixture_db(raw_csv, tmp_path):
    out = tmp_path / "processed"
    run(raw_csv, out, size=30, seed=5)
    return out / "recalliq.db"


# --- corpus / indexing -----------------------------------------------------------------

def test_search_text_uses_verbatim_fields_in_fixed_order():
    assert build_search_text("Title", "Product", "Issue", "Cat") == "Title. Product. Issue. Cat"
    assert build_search_text("Title", None, None, "Cat") == "Title. Cat"
    assert build_search_text("Same", "Same", None, None) == "Same"


def test_load_corpus_matches_database(fixture_db):
    corpus = load_corpus(fixture_db)
    db = sqlite3.connect(fixture_db)
    try:
        ids = [row[0] for row in db.execute("SELECT id FROM recalls ORDER BY id")]
    finally:
        db.close()
    assert [r.recall_id for r in corpus] == ids
    assert all(r.search_text.startswith(r.title) for r in corpus)


def test_corpus_fingerprint_is_stable_and_detects_text_changes():
    assert corpus_fingerprint(CORPUS) == corpus_fingerprint(list(CORPUS))
    changed = CORPUS[:-1] + [_record(40, "Portable phone charger", "Fire hazard")]
    assert corpus_fingerprint(changed) != corpus_fingerprint(CORPUS)


def test_tokenize_casefolds_drops_stopwords_keeps_accents():
    assert tokenize("The Crème BRÛLÉE is in a jar") == ["crème", "brûlée", "jar"]


# --- BM25 ------------------------------------------------------------------------------

def test_bm25_ranks_matching_document_first():
    index = BM25Index([r.search_text for r in CORPUS])
    scores = index.scores("listeria cheese")
    assert int(np.argmax(scores)) == 2
    assert scores[2] > 0 and scores[0] == 0


def test_bm25_returns_zero_scores_without_overlap():
    index = BM25Index([r.search_text for r in CORPUS])
    assert not index.scores("bicycle helmet").any()
    assert not index.scores("the and of").any()  # stopwords only


def test_bm25_prefers_rarer_terms_and_shorter_documents():
    index = BM25Index(["recall milk", "recall milk", "recall egg", "recall egg " + "filler " * 20])
    scores = index.scores("recall egg")
    assert scores[2] > scores[3] > scores[0]  # rare 'egg' matters; length penalized


def test_rank_breaks_ties_by_recall_id():
    assert rank(np.array([1.0, 2.0, 2.0, 0.5]), [40, 30, 10, 20], 3) == [2, 1, 0]


# --- shared search interface -----------------------------------------------------------

def test_bm25_search_maps_results_to_original_records():
    hits = SearchEngine(CORPUS).search("peanut butter salmonella", model="bm25", top_k=3)
    assert hits[0].recall_id == 10
    assert hits[0].url == "https://example.test/10"
    assert hits[0].title == CORPUS[0].title
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(h.score > 0 for h in hits)  # zero-overlap documents are not returned


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_is_rejected(query):
    with pytest.raises(ValueError, match="empty"):
        SearchEngine(CORPUS).search(query)


def test_invalid_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        SearchEngine(CORPUS).search("cheese", model="tfidf")


@pytest.mark.parametrize("top_k", [0, 51])
def test_top_k_bounds(top_k):
    with pytest.raises(ValueError, match="top_k"):
        SearchEngine(CORPUS).search("cheese", top_k=top_k)


def test_minilm_unavailable_without_embeddings():
    with pytest.raises(SearchUnavailableError):
        SearchEngine(CORPUS).search("cheese", model="minilm")


def test_model_load_failure_becomes_search_unavailable():
    from retrieval.semantic import ModelUnavailableError

    class BrokenEncoder:
        def encode(self, texts):
            raise ModelUnavailableError("offline")

    engine = SearchEngine(CORPUS, embeddings=_fake_index(), encoder=BrokenEncoder())
    with pytest.raises(SearchUnavailableError, match="offline"):
        engine.search("cheese", model="minilm")


# --- semantic index --------------------------------------------------------------------

def _fake_index(corpus=CORPUS) -> EmbeddingIndex:
    matrix = FakeEncoder().encode([r.search_text for r in corpus])
    return EmbeddingIndex(
        recall_ids=[r.recall_id for r in corpus], matrix=matrix,
        metadata={"corpus_fingerprint": corpus_fingerprint(corpus)},
    )


def test_cosine_search_ranks_identical_text_first_and_stays_in_range():
    engine = SearchEngine(CORPUS, embeddings=_fake_index(), encoder=FakeEncoder())
    hits = engine.search(CORPUS[3].search_text, model="minilm", top_k=4)
    assert hits[0].recall_id == 40
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert all(-1.0 - 1e-6 <= h.score <= 1.0 + 1e-6 for h in hits)
    assert len(hits) == 4  # cosine returns top_k even for weak matches


def test_embeddings_round_trip(tmp_path):
    index = _fake_index()
    save_embeddings(tmp_path, index.recall_ids, index.matrix, {**index.metadata, "dimension": 16})
    loaded = load_embeddings(tmp_path)
    assert loaded.matrix.shape == (len(CORPUS), FakeEncoder.dimension)
    assert loaded.matrix.dtype == np.float32
    assert loaded.recall_ids == [10, 20, 30, 40]
    assert not list(tmp_path.glob("*.tmp*"))


def test_save_rejects_duplicate_ids_and_shape_mismatch(tmp_path):
    matrix = FakeEncoder().encode(["a", "b"])
    with pytest.raises(ValueError, match="Duplicate"):
        save_embeddings(tmp_path, [1, 1], matrix, {"dimension": 16})
    with pytest.raises(ValueError, match="shape"):
        save_embeddings(tmp_path, [1, 2, 3], matrix, {"dimension": 16})


def test_stale_embeddings_are_detected():
    index = _fake_index()
    reordered = list(reversed(CORPUS))
    with pytest.raises(StaleIndexError, match="ids"):
        SearchEngine(reordered, embeddings=index, encoder=FakeEncoder())
    edited = CORPUS[:-1] + [_record(40, "Different text")]
    with pytest.raises(StaleIndexError, match="changed"):
        SearchEngine(edited, embeddings=index, encoder=FakeEncoder())


def test_query_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="dim"):
        _fake_index().cosine_scores(np.ones(8, dtype=np.float32))


def test_build_engine_degrades_to_bm25_when_embeddings_missing(fixture_db, tmp_path):
    engine = build_engine(fixture_db, tmp_path / "no-embeddings")
    assert engine.search("Generated test recall", model="bm25", top_k=5)
    with pytest.raises(SearchUnavailableError, match="retrieval.embed"):
        engine.search("cheese", model="minilm")


def test_build_engine_with_fake_embeddings(fixture_db, tmp_path):
    emb_dir = build_fake_embeddings(fixture_db, tmp_path / "emb")
    engine = build_engine(fixture_db, emb_dir, encoder=FakeEncoder())
    hits = engine.search("Generated test recall", model="minilm", top_k=5)
    assert len(hits) == 5 and len({h.recall_id for h in hits}) == 5


def test_equal_bm25_scores_rank_lower_recall_id_first():
    corpus = [_record(90, "Identical frozen peas"), _record(15, "Identical frozen peas"),
              _record(50, "Other product")]
    hits = SearchEngine(corpus).search("frozen peas", model="bm25", top_k=3)
    assert [h.recall_id for h in hits] == [15, 90]
    assert hits[0].score == hits[1].score


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda d: (d / "embeddings.npy").write_bytes(b"\x93NUMPY truncated"),
        lambda d: (d / "recall_ids.json").write_text("[1, 2", "utf-8"),
        lambda d: (d / "metadata.json").write_text("{not json", "utf-8"),
    ],
)
def test_corrupt_artifacts_degrade_to_bm25(fixture_db, tmp_path, corrupt):
    emb_dir = build_fake_embeddings(fixture_db, tmp_path / "emb")
    corrupt(emb_dir)
    engine = build_engine(fixture_db, emb_dir, encoder=FakeEncoder())
    assert engine.search("Generated test recall", model="bm25", top_k=3)
    with pytest.raises(SearchUnavailableError, match="unreadable|shape"):
        engine.search("cheese", model="minilm")


def test_non_finite_or_non_unit_rows_are_rejected(tmp_path):
    index = _fake_index()
    for bad in (np.full_like(index.matrix, np.nan), index.matrix * 2.0):
        save_embeddings(tmp_path, index.recall_ids, bad, {**index.metadata, "dimension": 16})
        with pytest.raises(StaleIndexError, match="NaN|unit"):
            load_embeddings(tmp_path)


def test_real_encoder_requires_matching_model_and_pinned_revision(fixture_db, tmp_path):
    # Fake artifacts declare model_id "fake" and no revision: without an injected test
    # encoder the loader must refuse rather than encode queries with a different model.
    emb_dir = build_fake_embeddings(fixture_db, tmp_path / "emb")
    engine = build_engine(fixture_db, emb_dir)
    with pytest.raises(SearchUnavailableError, match="built with"):
        engine.search("cheese", model="minilm")


def test_model_load_failure_is_cached_briefly(monkeypatch):
    import sys

    from retrieval.semantic import MiniLMEncoder, ModelUnavailableError

    calls = []

    class Boom:
        def __init__(self, *args, **kwargs):
            calls.append(1)
            raise OSError("offline")

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        type(sys)("sentence_transformers"))
    sys.modules["sentence_transformers"].SentenceTransformer = Boom
    encoder = MiniLMEncoder(revision="x")
    for _ in range(3):
        with pytest.raises(ModelUnavailableError, match="offline"):
            encoder.encode(["q"])
    assert len(calls) == 1  # later calls fail fast instead of retrying the download


def test_pinned_revision_is_a_full_commit_hash():
    from retrieval import MINILM_REVISION

    assert len(MINILM_REVISION) == 40 and all(c in "0123456789abcdef" for c in MINILM_REVISION)


# --- real artifacts (skipped until `python -m retrieval.embed` has been run) ------------

REAL_READY = (DEFAULT_OUT / "metadata.json").is_file() and DEFAULT_DB.is_file()


@pytest.mark.skipif(not REAL_READY, reason="real MiniLM embeddings not built")
def test_real_embeddings_shape_norms_and_id_mapping():
    index = load_embeddings(DEFAULT_OUT)
    corpus = load_corpus(DEFAULT_DB)
    assert index.matrix.shape == (len(corpus), 384)
    assert index.matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(index.matrix, axis=1), 1.0, atol=1e-4)
    assert index.metadata["model_id"] == "sentence-transformers/all-MiniLM-L6-v2"
    from retrieval import MINILM_REVISION

    assert index.metadata["model_revision"] == MINILM_REVISION  # built with the pinned commit
    index.check_matches(corpus)  # raises if ids/order/text drifted


@pytest.mark.skipif(not REAL_READY, reason="real MiniLM embeddings not built")
def test_real_minilm_query_embedding_and_self_retrieval_sanity():
    """Sanity check only (a record's own text should retrieve itself), not an accuracy metric."""
    engine = build_engine(DEFAULT_DB, DEFAULT_OUT)
    vector = engine.encoder.encode(["cheese with listeria"])
    assert vector.shape == (1, 384)
    for record in engine.corpus[:: len(engine.corpus) // 5][:5]:
        hits = engine.search(record.search_text, model="minilm", top_k=1)
        assert hits[0].recall_id == record.recall_id
    metadata = json.loads((DEFAULT_OUT / "metadata.json").read_text("utf-8"))
    assert metadata["count"] == len(engine.corpus)
