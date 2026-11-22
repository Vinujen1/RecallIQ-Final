"""Word-date handling in the answer guard (rag.guards.resolve_word_dates / validate_answer).

Notices store ISO dates; models write "October 1, 2012". The rewording must be accepted
only when the date is in the evidence: the guard must stay as strict for invented dates.
"""

from rag.guards import resolve_word_dates, validate_answer
from rag.pipeline import EvidenceItem

EVIDENCE = [EvidenceItem(rank=1, recall_id=101, title="Acme brand Cheddar recalled due to Listeria",
                         url="https://recalls-rappels.canada.ca/en/a", issue="Listeria",
                         last_updated="2012-10-01", identifier_verdict="UNKNOWN")]


# The basic accept/reject cases (incl. through the full pipeline) are in test_llm.py.

def test_word_date_with_ordinal_is_accepted():
    assert validate_answer("Recall 101 was updated on 1st October 2012.", [101], EVIDENCE) is None


def test_word_date_is_resolved_to_the_evidence_iso_date():
    resolved, reason = resolve_word_dates("Updated October 1, 2012.", EVIDENCE)
    assert reason is None and resolved == "Updated 2012-10-01."


def test_invented_word_dates_are_rejected():
    for text in ("Recall 101 was updated on October 2, 2012.",   # wrong day
                 "Recall 101 was updated on October 1, 2013.",   # wrong year
                 "Recall 101 was updated in March 2012."):       # month not in evidence
        reason = validate_answer(text, [101], EVIDENCE)
        assert reason and "dates not present" in reason, text


def test_iso_date_still_checked_and_bare_month_still_a_name():
    assert validate_answer("Recall 101 was updated on 2012-10-01.", [101], EVIDENCE) is None
    assert validate_answer("Recall 101 was updated on 2012-10-02.", [101], EVIDENCE)
    # A month without a year is not a date the guard can verify: still an unknown name.
    assert validate_answer("Recall 101 was updated in late October.", [101], EVIDENCE)


def test_word_date_copied_verbatim_from_evidence_is_kept():
    evidence = [EvidenceItem(rank=1, recall_id=7, title="Acme Soup best before October 15, 2024",
                             url="https://recalls-rappels.canada.ca/en/s",
                             last_updated="2024-11-02", identifier_verdict="UNKNOWN")]
    text = 'Recall 7 is "Acme Soup best before October 15, 2024".'
    assert resolve_word_dates(text, evidence) == (text, None)
    assert validate_answer(text, [7], evidence) is None


def test_sentence_start_after_trailing_space_and_newline_is_not_a_name():
    # Live gpt-4o-mini output (2026-11-19): ". \n\nSince ..." was rejected as the name "Since".
    text = "Recall 101 concerns Listeria. \n\nSince details are missing, read the notice."
    assert validate_answer(text, [101], EVIDENCE) is None
    # A real unknown name mid-sentence, also after doubled whitespace, is still rejected.
    assert "Zorblax" in validate_answer("Recall 101 concerns  Zorblax cheese.", [101], EVIDENCE)
    assert "Zorblax" in validate_answer("Recall 101 concerns\nZorblax cheese.", [101], EVIDENCE)


def test_missing_field_wording_from_the_prompt_passes_and_the_guard_is_unchanged():
    # rag-v2 rule 8 told the model to say missing fields are "not listed", which the safety
    # guard (rightly) reads like "your product is not listed". rag-v3 asks for "the notice
    # does not state ..."; the guard itself is not relaxed.
    from rag.llm import SYSTEM_PROMPT
    assert "not listed" not in SYSTEM_PROMPT and "does not state" in SYSTEM_PROMPT
    ok = "Recall 101 concerns Listeria. The notice does not state the product or what to do."
    assert validate_answer(ok, [101], EVIDENCE) is None
    for claim in ("Recall 101 concerns Listeria, but the details are not listed.",
                  "Recall 101 concerns Listeria. Your cheese is not listed in it."):
        assert "safety" in validate_answer(claim, [101], EVIDENCE)


def test_date_fix_does_not_relax_other_guards():
    assert "safety" in validate_answer(
        "Recall 101 was updated on October 1, 2012, so your cheese is safe.", [101], EVIDENCE)
    assert validate_answer("See https://example.com, updated October 1, 2012.", [101], EVIDENCE)
