import pytest

from clea.imagegen import PromptRejected, STYLE_SUFFIX, build_safe_prompt


@pytest.mark.parametrize("bad", [
    "a photo of a patient smiling",
    "realistic portrait of our dentist",
    "before/after of a real person's teeth",
    "photograph of my clinic team",
    "a selfie in the waiting room",
])
def test_realism_and_person_prompts_are_rejected(bad):
    with pytest.raises(PromptRejected):
        build_safe_prompt(bad)


@pytest.mark.parametrize("ok", [
    "a tooth with a shield icon representing protection",
    "abstract illustration of a toothbrush and sparkles",
    "flat icon of a calendar reminding you to floss",
    "database tables connected by lines, minimalist diagram",
])
def test_illustrative_prompts_pass_and_get_style_suffix(ok):
    result = build_safe_prompt(ok)
    assert result.startswith(ok)
    assert STYLE_SUFFIX in result
    assert "no photorealism" in result and "no human faces" in result
