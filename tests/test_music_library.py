from clea.config import Config
from clea.music_library import ensure_starter_pack, load_manifest, scan_library, track_path


def _lib_cfg(cfg: Config, tmp_path) -> Config:
    c = Config(dict(cfg))
    c["music_library"] = {"dir": str(tmp_path / "tracks"),
                          "manifest": str(tmp_path / "manifest.json")}
    return c


def test_starter_pack_generates_and_tags_tempo(cfg, tmp_path):
    lib_cfg = _lib_cfg(cfg, tmp_path)
    tracks = ensure_starter_pack(lib_cfg)
    assert len(tracks) == 4
    for t in tracks:
        assert t.duration > 25.0
        assert 60 <= t.tempo <= 200  # sane BPM range
        assert t.license  # starter pack tracks are labelled


def test_scan_is_idempotent_and_preserves_edits(cfg, tmp_path):
    lib_cfg = _lib_cfg(cfg, tmp_path)
    ensure_starter_pack(lib_cfg)
    tracks = load_manifest(lib_cfg)
    tracks[0].tags = ["custom-tag"]
    from clea.music_library import save_manifest
    save_manifest(lib_cfg, tracks)

    rescanned = scan_library(lib_cfg)
    assert len(rescanned) == 4
    edited = next(t for t in rescanned if t.id == tracks[0].id)
    assert edited.tags == ["custom-tag"]  # not clobbered by rescan


def test_track_path_resolves_and_missing_id_is_none(cfg, tmp_path):
    lib_cfg = _lib_cfg(cfg, tmp_path)
    tracks = ensure_starter_pack(lib_cfg)
    p = track_path(lib_cfg, tracks[0].id)
    assert p is not None and p.exists()
    assert track_path(lib_cfg, "does-not-exist") is None


def test_empty_library_scans_to_empty_list(cfg, tmp_path):
    lib_cfg = _lib_cfg(cfg, tmp_path)
    (tmp_path / "tracks").mkdir(parents=True)
    assert scan_library(lib_cfg) == []
