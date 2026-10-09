"""Every chat answer is rendered from one fixed schema, so its shape never depends on how the model felt that day."""
from service.services import structured


def render(**reply):
    return structured.render({"kind": "alert", "summary": "S.", **reply})


def test_an_alert_answer_has_the_same_sections_in_the_same_order_whatever_order_the_model_sent_them():
    out = render(sections=[
        {"heading": "Suggested next steps", "points": ["Call the customer."]},
        {"heading": "Key facts", "points": ["RM 1,028.23"]},
        {"heading": "Why it was flagged", "points": ["Foreign transaction."]},
    ])
    labels = [line for line in out.splitlines() if line.startswith("**")]
    assert labels == ["**Key facts**", "**Why it was flagged**", "**Suggested next steps**"]


def test_summary_comes_first_then_labelled_bullet_sections():
    out = render(summary="This P1 alert looks like card-not-present fraud.", sections=[{"heading": "Key facts", "points": ["RM 1,028.23", "CARD_ECOM"]}])
    assert out == "This P1 alert looks like card-not-present fraud.\n\n**Key facts**\n- RM 1,028.23\n- CARD_ECOM"


def test_headings_are_matched_ignoring_case_and_unknown_headings_are_dropped():
    out = render(sections=[{"heading": "why it was FLAGGED", "points": ["a"]}, {"heading": "My own heading", "points": ["b"]}])
    assert "**Why it was flagged**" in out and "My own heading" not in out and "- b" not in out


def test_empty_sections_are_left_out():
    out = render(sections=[{"heading": "Key facts", "points": []}, {"heading": "Why it was flagged", "points": ["x"]}])
    assert "Key facts" not in out


def test_a_data_answer_is_a_table_then_a_one_sentence_takeaway():
    out = structured.render({"kind": "data", "summary": "Alerts by channel.", "table": {"headers": ["Channel", "Alerts"], "rows": [["CARD_ECOM", "114"], ["ATM", "12"]]}, "takeaway": "CARD_ECOM has the most."})
    assert out == "Alerts by channel.\n\n| Channel | Alerts |\n|---|---:|\n| CARD_ECOM | 114 |\n| ATM | 12 |\n\nCARD_ECOM has the most."


def test_numeric_columns_are_right_aligned():
    out = structured.render({"kind": "data", "summary": "x", "table": {"headers": ["Channel", "Alerts", "Total (RM)"], "rows": [["ATM", "12", "3,353.22"]]}})
    assert "|---|---:|---:|" in out


def test_pipes_and_newlines_in_cells_cannot_break_the_table():
    out = structured.render({"kind": "data", "summary": "x", "table": {"headers": ["A", "B"], "rows": [["x | y", "line1\nline2"]]}})
    rows = [l for l in out.splitlines() if l.startswith("| x")]
    assert rows == ["| x / y | line1 line2 |"]


def test_a_policy_answer_uses_its_own_two_sections():
    out = structured.render({"kind": "policy", "summary": "Yes, within 3 working days.", "sections": [
        {"heading": "How it applies", "points": ["Applies to disputes."]}, {"heading": "What the policy says", "points": ["Acknowledge within 3 working days [a.pdf p.1]."]}]})
    assert out.index("**What the policy says**") < out.index("**How it applies**")


def test_a_general_answer_is_just_the_summary():
    assert structured.render({"kind": "general", "summary": "I can help with alerts, data and policies.", "sections": [{"heading": "Key facts", "points": ["x"]}]}) == "I can help with alerts, data and policies."


def test_size_limits_keep_every_answer_short():
    out = render(summary=" ".join(["w"] * 80), sections=[{"heading": "Key facts", "points": [f"p{i}" for i in range(9)]}])
    assert out.count("\n- ") == structured.MAX_POINTS
    assert len(out.split("\n\n")[0].split()) <= structured.MAX_SUMMARY_WORDS + 1


def test_an_unknown_kind_is_treated_as_general_and_a_missing_summary_is_filled():
    assert structured.render({"kind": "poem", "summary": "Hello."}) == "Hello."
    assert structured.render({"kind": "alert"}).strip() != ""


def test_parse_accepts_json_and_json_in_a_code_fence_and_rejects_anything_else():
    assert structured.parse('{"kind": "general", "summary": "x"}')["summary"] == "x"
    assert structured.parse('```json\n{"kind": "general", "summary": "x"}\n```')["summary"] == "x"
    assert structured.parse("Sure! Here you go: it is fine.") is None
    assert structured.parse('{"kind": "general"}') is None  # no summary: not usable
    assert structured.parse("[1, 2]") is None
