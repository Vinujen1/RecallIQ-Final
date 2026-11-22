"""POST /explain with the OpenAI provider configured (all OpenAI calls mocked).

Synthetic DB + FakeEncoder (as test_search_api.py). A fake client writes its answer from
the notices it actually receives, so grounding is checked against real API evidence.
"""

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import rag.llm
from app.config import Settings, get_settings
from app.main import app
from app.search_service import get_search_engine
from conftest import FakeOpenAI, FakeEncoder, build_fake_embeddings, openai_error
from recall_data.build import run
from retrieval.loader import build_engine

KEY = "sk-test-not-a-real-key-abcdef"
QUERY = "Generated test recall"


class GroundedFake(FakeOpenAI):
    """Answers from the notices in the request (like a well-behaved model) unless told to
    hallucinate."""

    def __init__(self, hallucinate: bool = False, **kw):
        super().__init__(**kw)
        self.hallucinate = hallucinate

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        notices = json.loads(kwargs["messages"][1]["content"])["notices"]
        top = notices[0]
        text = (f"Recall {top['recall_id']} may be relevant: \"{top['title']}\". Check the "
                "official notice; this search does not mean any product is safe.")
        if self.hallucinate:
            text += " It affects lot 99Z123 and UPC 036000291452."
        body = {"answer": text, "cited_recall_ids": [top["recall_id"]], "evidence_sufficient": True}
        self.response.choices[0].message.content = json.dumps(body)
        return self.response


@pytest.fixture
def paths(raw_csv, tmp_path):
    out = tmp_path / "processed"
    run(raw_csv, out, size=30, seed=11)
    db = out / "recalliq.db"
    return db, build_fake_embeddings(db, tmp_path / "emb")


@pytest.fixture
def make_client(paths, monkeypatch):
    db, emb = paths
    engine = build_engine(db, emb, encoder=FakeEncoder())

    def factory(fake=None, **overrides):
        values = {"database_path": db, "embeddings_dir": emb, "llm_provider": "openai",
                  "openai_api_key": KEY, **overrides}
        app.dependency_overrides[get_settings] = lambda: Settings(**values)
        app.dependency_overrides[get_search_engine] = lambda: engine
        if fake is not None:
            monkeypatch.setattr(rag.llm, "get_client", lambda config: fake)
        return TestClient(app)

    yield factory
    app.dependency_overrides.clear()


def _explain(client, **body):
    response = client.post("/explain", json={"query": QUERY, **body})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("model", ["bm25", "minilm"])
def test_llm_answer_is_returned_with_citations_and_stats(make_client, model):
    fake = GroundedFake()
    body = _explain(make_client(fake), model=model)
    exp = body["explanation"]
    assert body["model"] == model and body["evidence_status"] == "evidence_found"
    assert exp["generator"] == "llm:openai" and exp["fallback_reason"] is None
    facts = {f["recall_id"]: f for f in body["retrieved_facts"]}
    assert exp["cited_recall_ids"] and set(exp["cited_recall_ids"]) <= set(facts)
    assert body["sources"] == [f["url"] for f in body["retrieved_facts"]]
    assert all("what_you_should_do" in f for f in body["retrieved_facts"])
    stats = exp["generation"]
    assert stats["outcome"] == "accepted" and stats["total_tokens"] == 908
    assert stats["latency_ms"] >= 0 and stats["prompt_version"] == "rag-v3"
    assert fake.calls[0]["model"] == "gpt-4o-mini"
    assert KEY not in json.dumps(body)


def test_hallucinated_codes_are_replaced_by_template(make_client):
    body = _explain(make_client(GroundedFake(hallucinate=True)))
    exp = body["explanation"]
    assert exp["generator"] == "template" and exp["generation"]["outcome"] == "rejected"
    assert "99Z123" not in exp["text"] and "evidence checks" in exp["fallback_reason"]


@pytest.mark.parametrize("kind", ["timeout", "connection", "rate_limit", "server"])
def test_openai_outage_still_returns_200_with_template(make_client, kind):
    body = _explain(make_client(GroundedFake(error=openai_error(kind))))
    exp = body["explanation"]
    assert exp["generator"] == "template" and exp["fallback_reason"]
    assert exp["generation"]["outcome"].startswith("error:")
    assert body["retrieved_facts"] and body["sources"]


def test_no_results_skips_the_model(make_client):
    fake = GroundedFake()
    body = _explain(make_client(fake), query="zzqx")
    assert fake.calls == [] and body["evidence_status"] == "no_results"
    assert body["explanation"]["generation"]["outcome"] == "skipped:no_results"


def test_missing_key_reports_not_configured(make_client):
    body = _explain(make_client(GroundedFake(), openai_api_key=""))
    assert body["explanation"]["generation"]["outcome"] == "error:not_configured"


def test_default_provider_is_template_only(make_client):
    fake = GroundedFake()
    body = _explain(make_client(fake, llm_provider="none"))
    assert fake.calls == [] and body["explanation"]["generator"] == "template"
    assert body["explanation"]["generation"] is None


def test_rag_top_k_and_max_evidence_come_from_settings(make_client):
    fake = GroundedFake()
    client = make_client(fake, rag_top_k=4, rag_max_evidence=2, openai_model="gpt-4o-mini-2024-07-18",
                         llm_max_output_tokens=300)
    body = _explain(client)
    assert len(body["retrieved_facts"]) == 2
    assert len(json.loads(fake.calls[0]["messages"][1]["content"])["notices"]) == 2
    assert fake.calls[0]["model"] == "gpt-4o-mini-2024-07-18"
    assert fake.calls[0]["max_completion_tokens"] == 300
    explicit = _explain(client, top_k=1)
    assert len(explicit["retrieved_facts"]) == 1


def test_rejection_reason_reaches_the_server_log(make_client, capsys):
    """Under uvicorn nothing configured `recalliq`, so rejection reasons were invisible.
    The app now attaches one handler; the line has the reason category, not the query or key."""
    import logging

    from app.main import configure_logging

    configure_logging("INFO")
    _explain(make_client(GroundedFake(hallucinate=True)))
    err = capsys.readouterr().err
    line = next(l for l in err.splitlines() if "rag_generation" in l)
    assert "outcome=rejected" in line and "reason='mentions codes/numbers" in line
    assert QUERY not in err and KEY not in err
    configure_logging("INFO")  # idempotent: still exactly one RecallIQ handler
    ours = [h for h in logging.getLogger("recalliq").handlers if getattr(h, "_recalliq", False)]
    assert len(ours) == 1


def test_log_level_setting_is_validated():
    assert Settings(log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValueError):
        Settings(log_level="loud")


def test_settings_read_key_from_env_alias_and_validate(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setenv("RECALLIQ_LLM_PROVIDER", "OpenAI")
    settings = Settings(_env_file=None)
    assert settings.llm_config.api_key == KEY and settings.llm_config.provider == "openai"
    assert KEY not in repr(settings) and KEY not in repr(settings.llm_config)
    env = tmp_path / ".env"
    env.write_text("RECALLIQ_OPENAI_API_KEY=from-dotenv\nRECALLIQ_RAG_TOP_K=3\n", "utf-8")
    monkeypatch.delenv("OPENAI_API_KEY")
    from_file = Settings(_env_file=env)
    assert from_file.llm_config.api_key == "from-dotenv" and from_file.rag_top_k == 3
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_provider="anthropic")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, rag_top_k=9)
