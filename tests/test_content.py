import pytest

from clea.content import (
    _BANNED_DENTAL, _FORMAT_GUIDANCE, _compliance_check, _system_prompt,
    CONTENT_TYPES, REEL_FORMATS, ContentPack, format_pack,
)


def test_dental_system_prompt_has_guardrail():
    sp = _system_prompt("dental")
    assert "COMPLIANCE RULES" in sp
    assert "consult your dentist" in sp.lower()


def test_educational_system_prompt_has_no_guardrail():
    assert "COMPLIANCE RULES" not in _system_prompt("educational-other")


def test_every_reel_format_has_guidance():
    assert set(REEL_FORMATS) == set(_FORMAT_GUIDANCE)


def test_case_study_guidance_protects_patients():
    g = _FORMAT_GUIDANCE["case-study"]
    assert "results vary" in g.lower()
    assert "patient-identifying" in g.lower() or "anonymous" in g.lower()


@pytest.mark.parametrize("bad", ["Instant whitening!", "100% painless",
                                 "We cure gum disease", "guaranteed results"])
def test_compliance_check_flags_banned_words(bad):
    pack = ContentPack(topic="t", content_type="dental", hook=bad)
    assert _compliance_check(pack)


def test_compliance_check_passes_soft_claims():
    pack = ContentPack(
        topic="t", content_type="dental",
        hook="Gentle cleaning, explained",
        script={"solution": "In most cases scaling can help reduce tartar. "
                            "Consult your dentist."},
        caption="Ask your dentist what's right for you.")
    assert _compliance_check(pack) == []


def test_format_pack_includes_warnings_and_overlays():
    pack = ContentPack(topic="t", content_type="dental",
                       hook="h", caption="c",
                       hashtags=["#a"], overlay_texts=["Line one"],
                       compliance_warnings=["instant"])
    out = format_pack(pack)
    assert "COMPLIANCE WARNINGS" in out
    assert "Line one" in out


def test_known_constants_are_stable():
    assert "dental" in CONTENT_TYPES
    assert "case-study" in REEL_FORMATS
    assert "instant" in _BANNED_DENTAL
