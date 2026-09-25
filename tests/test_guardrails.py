from hr_rag.guardrails import _deidentified_text, _matched


def test_matched_finds_nested_match_state():
    assert _matched({"pi_and_jailbreak_filter_result": {"match_state": "MATCH_FOUND"}})
    assert _matched({"rai_filter_result": {"rai_filter_type_results": {"hate": {"match_state": "MATCH_FOUND"}}}})
    assert not _matched({"rai_filter_result": {"match_state": "NO_MATCH_FOUND"}})


def test_deidentified_text_extraction():
    results = {"sdp": {"sdp_filter_result": {"deidentify_result": {"data": {"text": "Call [PHONE]"}}}}}
    assert _deidentified_text(results) == "Call [PHONE]"
    assert _deidentified_text({}) is None
