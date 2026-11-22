"""Grounded explanation pipeline: evidence-only wording, missing fields, mismatch handling,
grounding guard for future LLM output, provider fallback (no network)."""

from rag.pipeline import (
    EvidenceItem,
    check_grounded,
    explain,
    select_evidence,
    template_explanation,
)

HITS = [
    {"rank": 1, "recall_id": 101, "title": "Acme brand Cheddar recalled due to Listeria",
     "url": "https://recalls-rappels.canada.ca/en/a", "issue": "Listeria", "category": "Food",
     "product": None, "recall_class": "Class 1", "last_updated": "2024-01-15",
     "identifier_verdict": "UNKNOWN"},
    {"rank": 2, "recall_id": 102, "title": "Other cheese recalled",
     "url": "https://recalls-rappels.canada.ca/en/b", "issue": None, "category": None,
     "identifier_verdict": "MISMATCH"},
    {"rank": 3, "recall_id": 103, "title": "Third cheese recalled",
     "url": "https://recalls-rappels.canada.ca/en/c", "identifier_verdict": "MATCH"},
]


def test_template_uses_only_stored_fields_and_marks_missing_ones():
    answer = explain("is my cheddar safe", "bm25", HITS, provider="none")
    text = answer.explanation.text
    assert answer.evidence_status == "evidence_found" and answer.explanation.generator == "template"
    assert "Acme brand Cheddar" in text and "Class 1" in text and "2024-01-15" in text
    assert "Product: not listed in the indexed record" in text
    assert check_grounded(text, answer.explanation.cited_recall_ids, answer.retrieved_facts) is None
    assert answer.sources == [f.url for f in answer.retrieved_facts]
    assert "not confirmation" in answer.disclaimer


def test_identifier_mismatches_are_dropped_from_evidence_unless_nothing_else():
    assert [e.recall_id for e in select_evidence(HITS)] == [101, 103]
    answer = explain("model XJ-600", "bm25", [HITS[1]], provider="none")
    assert answer.evidence_status == "identifier_mismatch"
    assert "differ from the ones in your description" in answer.explanation.text


def test_no_results_never_claims_safety():
    answer = explain("anything", "bm25", [], provider="none")
    assert answer.evidence_status == "no_results" and answer.sources == []
    assert "safe" not in answer.explanation.text.casefold()
    assert "recalls-rappels.canada.ca" in answer.explanation.text


def test_grounding_guard_rejects_invented_codes_and_foreign_ids():
    evidence = select_evidence(HITS)
    assert "not in the evidence" in check_grounded("see recall 999", [999], evidence)
    invented = check_grounded("Affected UPC 036000291452 lot 24A119.", [101], evidence)
    assert invented and "036000291452" in invented
    assert check_grounded("Recall 101 concerns Listeria (Class 1).", [101], evidence) is None


def test_unconfigured_llm_provider_falls_back_to_template(monkeypatch):
    monkeypatch.setenv("RECALLIQ_LLM_PROVIDER", "anthropic")
    answer = explain("cheddar", "bm25", HITS)
    assert answer.explanation.generator == "template"
    assert "not implemented" in answer.explanation.fallback_reason
    monkeypatch.delenv("RECALLIQ_LLM_PROVIDER")
    assert explain("cheddar", "bm25", HITS).explanation.fallback_reason is None


def test_template_is_deterministic():
    evidence = [EvidenceItem(**HITS[0])]
    assert template_explanation(evidence, "evidence_found") == template_explanation(evidence, "evidence_found")


def test_skipped_higher_ranked_mismatches_are_reported_and_not_called_closest():
    hits = [{**HITS[1], "rank": 1}, {**HITS[0], "rank": 2}]
    answer = explain("model XJ-600 cheddar", "minilm_finetuned", hits, provider="none")
    assert answer.skipped_identifier_mismatches == 1
    text = answer.explanation.text
    assert "1 higher-ranked notice(s) were set aside" in text
    assert "Highest-ranked notice without conflicting identifiers" in text and "Closest" not in text
    assert "nearest notices" in text                        # dense model caveat


def test_bm25_answer_has_no_dense_caveat():
    assert "nearest notices" not in explain("cheddar", "bm25", HITS, provider="none").explanation.text
