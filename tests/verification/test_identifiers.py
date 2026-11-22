"""Deterministic identifier verification. Example codes are standard GS1 examples with
correct check digits (UPC-A 036000291452, EAN-13 4006381333931); recall-like texts are
written for these tests."""

import pytest

from verification.identifiers import (
    NOT_A_SAFETY_GUARANTEE,
    Verdict,
    extract,
    has_identifier,
    normalize_upc,
    verify,
)


@pytest.mark.parametrize("text,expected", [
    ("036000291452", "0036000291452"),          # UPC-A, padded to 13
    ("0 36000 29145 2", "0036000291452"),       # spaced, as printed on notices
    ("036000-291452", "0036000291452"),
    ("4006381333931", "4006381333931"),         # EAN-13
    ("036000291453", None),                      # wrong check digit
    ("12345", None),
])
def test_upc_normalisation_and_check_digit(text, expected):
    assert normalize_upc(text) == expected


def test_upc_a_and_ean13_forms_of_one_code_match():
    result = verify("upc 036000291452", "UPC: 0036000291452")
    assert result.verdict is Verdict.MATCH
    assert result.by_type["upc"].matched == ["0036000291452"]


def test_upc_mismatch_is_confirmed_only_when_both_sides_have_one():
    assert verify("036000291452", "UPC 4006381333931").verdict is Verdict.MISMATCH
    assert verify("036000291452", "Brand X cookies, no codes listed").verdict is Verdict.UNKNOWN
    assert verify("my cookies", "UPC 4006381333931").verdict is Verdict.UNKNOWN


def test_phone_numbers_and_invalid_codes_are_not_upcs():
    assert extract("call 1-800-555-0199 or 416 555 0123").upc == set()


def test_model_lists_and_separators_normalise():
    ids = extract("Models XJ-500, XJ-600 and XJ700B are affected")
    assert ids.model == {"XJ500", "XJ600", "XJ700B"}
    assert extract("Model no.: 12A and blue").model == {"12A"}
    assert extract("Item # 4410-22").model == {"441022"}


def test_similar_model_numbers_are_a_mismatch_not_a_match():
    result = verify("model XJ-600 heater", "Recalled: space heater model XJ-500")
    assert result.verdict is Verdict.MISMATCH
    assert result.by_type["model"].query_values == ["XJ600"]


def test_loose_query_code_can_confirm_but_never_refute():
    assert verify("my xj500 heater sparks", "heater model XJ-500").verdict is Verdict.MATCH
    assert verify("my xj900 heater sparks", "heater model XJ-500").verdict is Verdict.UNKNOWN


def test_hazard_tokens_are_not_codes():
    assert extract("E. coli O157:H7 in beef", loose=True).loose == set()


def test_lot_mismatch_overrides_upc_match():
    evidence = "UPC 036000291452, lot 24A118"
    result = verify("upc 036000291452 lot 24A119", evidence)
    assert result.by_type["upc"].verdict is Verdict.MATCH
    assert result.by_type["lot"].verdict is Verdict.MISMATCH
    assert result.verdict is Verdict.MISMATCH


def test_best_before_formats():
    assert extract("Best Before 2024 JA 15").best_before == {"2024-01-15"}
    assert extract("BB: 2024-03-02").best_before == {"2024-03-02"}
    assert extract("best before January 5, 2025").best_before == {"2025-01-05"}
    assert extract("best before 05/01/2025").best_before == set()  # ambiguous: skipped
    assert extract("Best Before 2024 ZZ 15").best_before == set()
    assert verify("best before 2024-01-15", "Best Before 2024 JA 15").verdict is Verdict.MATCH


def test_missing_identifiers_are_unknown_with_safety_note():
    result = verify("is the peanut butter i bought safe", "Brand X peanut butter, Salmonella")
    assert result.verdict is Verdict.UNKNOWN
    assert all(r.verdict is Verdict.UNKNOWN for r in result.by_type.values())
    assert result.note == NOT_A_SAFETY_GUARANTEE
    assert result.to_json()["verdict"] == "missing_or_unknown"


@pytest.mark.parametrize("query,evidence", [
    ("lot 24123A", "Lot code: 24 123 A"),                                   # spacing differs
    ("best before 2024-01-17", "Best before 2024 JA 15 to 2024 JA 20"),    # date range
    ("best before 2024-01-17", "Best before 2024 JA 15 and 2024 JA 20"),   # date list
    ("model 500", "model 500-S"),                                           # suffix variant
])
def test_ambiguous_differences_are_unknown_not_mismatch(query, evidence):
    """Regression (ECC review 2026-11-16): these were false CONFIRMED_MISMATCH verdicts."""
    assert verify(query, evidence).verdict is Verdict.UNKNOWN


def test_lot_number_is_not_also_read_as_upc():
    ids = extract("lot 036000291452")
    assert ids.lot == {"036000291452"} and ids.upc == set()
    assert verify("lot 036000291452", "UPC 4006381333931").verdict is Verdict.UNKNOWN


def test_has_identifier_requires_a_tagged_code_or_upc():
    assert has_identifier("model XJ-500")
    assert has_identifier("036000291452")
    assert not has_identifier("my xj500 heater")
