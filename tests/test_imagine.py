"""Imagine pipeline tested with a mocked image generator — validates the
orchestration (content pack -> scene prompts -> Ken Burns -> beat-synced
edit with scene captions) without needing torch/diffusers or a GPU."""

import numpy as np
import pytest
from PIL import Image

import clea.imagine as imagine_mod
from clea.imagegen import GeneratedImage
from clea.imagine import ImagineOptions, generate_video_from_topic, scene_prompts
from clea.content import ContentPack


def test_scene_prompts_prefers_overlay_texts():
    pack = ContentPack(topic="t", content_type="dental", hook="H",
                       script={"problem": "P", "solution": "S"},
                       overlay_texts=["Line one", "Line two", "Line one"])
    prompts = scene_prompts(pack, max_scenes=5)
    assert prompts == ["Line one", "Line two"]  # de-duped, order preserved


def test_scene_prompts_falls_back_to_script_parts():
    pack = ContentPack(topic="t", content_type="dental", hook="Hook line",
                       script={"problem": "Problem line", "solution": "Solution line"})
    prompts = scene_prompts(pack, max_scenes=5)
    assert prompts[0] == "Hook line"
    assert "Problem line" in prompts and "Solution line" in prompts


def test_scene_prompts_respects_max_scenes():
    pack = ContentPack(topic="t", content_type="dental",
                       overlay_texts=[f"L{i}" for i in range(10)])
    assert len(scene_prompts(pack, max_scenes=3)) == 3


@pytest.fixture()
def fake_pack():
    return ContentPack(
        topic="flossing", content_type="dental",
        hook="Floss like you mean it",
        script={"problem": "Plaque hides between teeth.",
                "solution": "Floss once a day, gently.",
                "cta": "Ask your dentist for technique tips."},
        caption="Small habit, big difference.",
        hashtags=["#dentalhealth", "#flossing"],
    )


def _fake_generate_content(topic, content_type, cfg, reel_format="informational"):
    return _fake_generate_content.pack


def _fake_generate_images(prompts, out_dir, cfg, progress=lambda *a: None):
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = []
    for i, p in enumerate(prompts):
        img_path = str(Path(out_dir) / f"img_{i}.png")
        arr = (np.random.default_rng(i).random((300, 300, 3)) * 255).astype("uint8")
        Image.fromarray(arr).save(img_path)
        out.append(GeneratedImage(prompt=p, path=img_path))
    return out


def test_generate_video_from_topic_end_to_end(monkeypatch, fake_pack, cfg, media, tmp_path):
    _fake_generate_content.pack = fake_pack
    monkeypatch.setattr(imagine_mod, "generate_content", _fake_generate_content)
    monkeypatch.setattr(imagine_mod, "generate_images", _fake_generate_images)

    opts = ImagineOptions(topic="flossing", content_type="dental",
                          audio_path=str(media["audio"]), duration=15.0)
    result = generate_video_from_topic(
        opts, cfg, workdir=str(tmp_path / "work"), out_path=str(tmp_path / "out.mp4"))

    assert result.pack.hook == "Floss like you mean it"
    assert len(result.image_prompts) >= 1
    assert result.edit is not None
    from pathlib import Path
    assert Path(result.edit.out_path).exists()

    ass_text = Path(result.edit.out_path).with_suffix(".ass").read_text()
    assert "Style: Hook," in ass_text and fake_pack.hook in ass_text
    assert "Style: Note," in ass_text  # scene captions rendered as note cards


def test_requires_audio_or_music_id(cfg, tmp_path):
    opts = ImagineOptions(topic="x")  # neither audio_path nor music_id
    with pytest.raises(ValueError, match="audio_path or music_id"):
        generate_video_from_topic(opts, cfg, workdir=str(tmp_path), out_path=str(tmp_path / "o.mp4"))
