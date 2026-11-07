"""Giọng người dùng (clone đã lưu) cho v3 Turbo: nạp gộp, không ghi đè giọng gốc, lưu atomic."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vieneu.v3turbo import V3TurboVieNeuTTS, default_user_voices_path


def _bare(user_path: Path) -> V3TurboVieNeuTTS:
    """Instance không nạp engine: chỉ phần quản lý giọng."""
    t = object.__new__(V3TurboVieNeuTTS)
    t.default_style = "tu_nhien"
    t._preset_voices = {}
    t._default_voice = None
    t._user_voice_names = set()
    t._builtin_voice_names = set()
    t._removed_user_voices = set()
    t._changed_user_voices = set()
    t.user_voices_path = user_path
    return t


class _FakeEngine:
    def __init__(self, seed=0.5):
        self.seed = seed
        self.calls = []
        self.emb_calls = 0

    def prepare_reference(self, path, denoise=True, use_ref_codes=True):
        self.calls.append((path, denoise, use_ref_codes))
        return np.full(192, self.seed, np.float32), np.array([[1, 2, 3, 4]] * 5, np.int64)

    def extract_speaker_emb(self, wav, sr):
        """Loud clip → 10·e0, quiet clip → 12·e1 (lets tests check per-file weighting)."""
        self.emb_calls += 1
        e = np.zeros(192, np.float32)
        if float(np.sqrt(np.mean(np.asarray(wav) ** 2))) > 0.2:
            e[0] = 10.0
        else:
            e[1] = 12.0
        return e


def _voice(seed: float) -> dict:
    return {"description": "Nam · test", "gender": "male", "style": "tu_nhien",
            "speaker_emb": [seed] * 192, "codes": [[1, 2, 3, 4]] * 3}


def _ref_wav(tmp_path: Path, name="ref.wav") -> Path:
    p = tmp_path / name
    sr = 16000
    wav = np.zeros(sr * 2, np.float32)
    wav[sr // 2: sr * 3 // 2] = 0.3
    sf.write(p, wav, sr)
    return p


def test_default_path_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("VIENEU_USER_VOICES", raising=False)
    assert default_user_voices_path() == Path.home() / ".cache" / "vieneu" / "user_voices.json"
    monkeypatch.setenv("VIENEU_USER_VOICES", str(tmp_path / "v.json"))
    assert default_user_voices_path() == tmp_path / "v.json"


def test_builtin_voices_load_and_default(tmp_path):
    t = _bare(tmp_path / "none.json")
    t._load_v3_voices()
    ids = [i for _, i in t.list_preset_voices()]
    assert len(ids) >= 20 and "Adam" in ids
    assert t._default_voice in ids
    assert not any(v["custom"] for v in t.list_voices())
    assert t._builtin_voice_names == set(ids)


def test_user_voices_merge_and_collision(tmp_path):
    up = tmp_path / "user_voices.json"
    up.write_text(json.dumps({"presets": {
        "clone_A": _voice(0.5),
        "Adam": _voice(0.9),            # trùng giọng có sẵn → bỏ qua
        "no_emb": {"description": "x", "codes": [[1]]},   # thiếu speaker_emb → bỏ qua
    }}, ensure_ascii=False), encoding="utf-8")
    t = _bare(up)
    t._load_v3_voices()
    by_id = {v["id"]: v for v in t.list_voices()}
    assert by_id["clone_A"]["custom"] is True
    assert by_id["clone_A"]["label"] == "clone_A — Nam · test"
    assert t.is_user_voice("clone_A") and not t.is_user_voice("Adam")
    assert "no_emb" not in by_id
    # giọng gốc Adam không bị ghi đè bởi file người dùng
    assert float(t.get_preset_voice("Adam")["speaker_emb"][0]) != pytest.approx(0.9)
    assert t.get_preset_voice("clone_A")["speaker_emb"].dtype == np.float32
    assert t.get_preset_voice("clone_A")["codes"].shape == (3, 4)


@pytest.mark.parametrize("content", [
    "{not json",
    "[]",
    "null",
    '"str"',
    '{"presets": []}',
    '{"presets": {"x": null}}',
    '{"presets": {"x": "oops"}}',
    '{"presets": {"x": {"speaker_emb": "abc"}}}',
    '{"presets": {"x": {"speaker_emb": [[1, 2], [3]]}}}',
    '{"presets": {"x": {"speaker_emb": [0.1, 0.2], "codes": [[1, 2, 3], [1, 2]]}}}',
    '{"presets": {"x": {"speaker_emb": [0.1, 0.2], "codes": [["a", "b"]]}}}',
    '{"presets": {"x": {"speaker_emb": [0.1, 0.2], "codes": [1, 2, 3]}}}',
    '{"presets": {"x": {"speaker_emb": []}}}',
])
def test_malformed_user_file_never_raises(tmp_path, content):
    up = tmp_path / "user_voices.json"
    up.write_text(content, encoding="utf-8")
    t = _bare(up)
    t._load_v3_voices()          # không ném lỗi
    assert "x" not in t._preset_voices
    assert len(t._preset_voices) >= 20 and t._default_voice in t._preset_voices


def test_malformed_entry_does_not_block_healthy_sibling(tmp_path):
    up = tmp_path / "user_voices.json"
    up.write_text(json.dumps({"presets": {"bad": {"speaker_emb": "abc"}, "clone_A": _voice(0.4)}}), encoding="utf-8")
    t = _bare(up)
    t._load_v3_voices()
    assert t.is_user_voice("clone_A") and "bad" not in t._preset_voices


def test_add_voice_happy_path_and_replace(tmp_path):
    up = tmp_path / "user_voices.json"
    t = _bare(up)
    t._load_v3_voices()
    t.engine = _FakeEngine(0.5)
    ref = _ref_wav(tmp_path)

    assert t.add_voice(" clone_A ", ref, description="d1", save=True) == "clone_A"
    assert t.is_user_voice("clone_A")
    assert t.get_preset_voice("clone_A")["speaker_emb"].shape == (192,)
    assert t.get_preset_voice("clone_A")["codes"].shape == (5, 4)
    assert t._default_voice != "clone_A"          # đã có giọng mặc định gốc
    assert json.loads(up.read_text(encoding="utf-8"))["presets"]["clone_A"]["description"] == "d1"
    assert t.engine.calls[0][1] is True           # denoise mặc định
    assert not list(tmp_path.glob("**/temp_clone_optimized_*"))  # file tạm đã dọn

    # thay thế giọng người dùng cùng tên
    t.engine = _FakeEngine(0.7)
    t.add_voice("clone_A", ref, description="d2", denoise=False, save=True)
    assert float(t.get_preset_voice("clone_A")["speaker_emb"][0]) == pytest.approx(0.7)
    assert json.loads(up.read_text(encoding="utf-8"))["presets"]["clone_A"]["description"] == "d2"
    assert t.engine.calls[0][1] is False

    # tên trùng giọng có sẵn → lỗi, kể cả khi đã remove khỏi phiên
    with pytest.raises(ValueError):
        t.add_voice("Adam", ref)
    t.remove_voice("Adam")
    with pytest.raises(ValueError):
        t.add_voice("Adam", ref)
    with pytest.raises(ValueError):
        t.add_voice("   ", ref)


def test_add_voice_save_failure_rolls_back(tmp_path):
    up = tmp_path / "dir_as_file"
    up.mkdir()                                     # ghi file vào đường dẫn là thư mục → thất bại
    t = _bare(up)
    t._load_v3_voices()
    t.engine = _FakeEngine()
    with pytest.raises(Exception):
        t.add_voice("clone_X", _ref_wav(tmp_path), save=True)
    assert "clone_X" not in t._preset_voices and not t.is_user_voice("clone_X")


def test_save_user_voices_merges_with_disk(tmp_path):
    up = tmp_path / "sub" / "user_voices.json"
    t = _bare(up)
    t._load_v3_voices()
    t._preset_voices["clone_B"] = {"description": "d", "gender": "", "style": "tu_nhien",
                                   "speaker_emb": np.full(192, 0.25, np.float32),
                                   "codes": np.array([[7, 8]], np.int64)}
    t._user_voice_names.add("clone_B")
    out = t.save_user_voices()
    assert Path(out) == up and up.exists()
    data = json.loads(up.read_text(encoding="utf-8"))
    assert list(data["presets"]) == ["clone_B"]
    assert data["presets"]["clone_B"]["speaker_emb"][0] == pytest.approx(0.25)
    assert not list(up.parent.glob(".user_voices_*"))   # file tạm đã dọn

    # tiến trình khác thêm clone_C vào file → save của phiên này phải giữ lại clone_C
    data["presets"]["clone_C"] = _voice(0.6)
    up.write_text(json.dumps(data), encoding="utf-8")
    t.save_user_voices()
    assert set(json.loads(up.read_text(encoding="utf-8"))["presets"]) == {"clone_B", "clone_C"}

    # nạp lại ở phiên mới → cả hai là custom; remove(save=True) chỉ bỏ giọng đã xoá
    t2 = _bare(up)
    t2._load_v3_voices()
    assert t2.is_user_voice("clone_B") and t2.is_user_voice("clone_C")
    t2.remove_voice("clone_B", save=True)
    assert set(json.loads(up.read_text(encoding="utf-8"))["presets"]) == {"clone_C"}
    assert "Adam" in t2._preset_voices

    # giọng có sẵn lỡ có trong file người dùng thì không được ghi lại
    up.write_text(json.dumps({"presets": {"Adam": _voice(0.9), "clone_C": _voice(0.6)}}), encoding="utf-8")
    t2.save_user_voices()
    assert set(json.loads(up.read_text(encoding="utf-8"))["presets"]) == {"clone_C"}


def test_save_voices_exports_everything_to_explicit_path(tmp_path):
    t = _bare(tmp_path / "u.json")
    t._load_v3_voices()
    out = tmp_path / "all.json"
    t.save_voices(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["presets"]) >= 20 and data["default_voice"] == t._default_voice


# ── Chỉnh giọng: embedding trung bình + vùng miền ───────────────────────────────

def test_accent_direction_separates_builtin_regions():
    d = V3TurboVieNeuTTS.accent_direction("bac")
    assert d.shape == (192,) and np.linalg.norm(d) == pytest.approx(1.0, abs=1e-5)
    presets = json.loads(V3TurboVieNeuTTS.builtin_voices_path().read_text(encoding="utf-8"))["presets"]
    proj = {"Bắc": [], "Nam": []}
    for v in presets.values():
        if v.get("region") in proj:
            e = np.asarray(v["speaker_emb"], np.float32)
            proj[v["region"]].append(float(e @ d / np.linalg.norm(e)))
    assert np.mean(proj["Bắc"]) > np.mean(proj["Nam"]) + 0.1
    np.testing.assert_allclose(V3TurboVieNeuTTS.accent_direction("nam"), -d, atol=1e-6)
    d[:] = 0                                   # bản trả về là bản sao, không làm hỏng cache
    assert np.linalg.norm(V3TurboVieNeuTTS.accent_direction("bac")) == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError):
        V3TurboVieNeuTTS.accent_direction("trung")


def test_steer_accent_keeps_norm_and_moves_toward_region():
    rng = np.random.default_rng(0)
    e = rng.normal(size=192).astype(np.float32) * 0.8
    d = V3TurboVieNeuTTS.accent_direction("bac")
    out = V3TurboVieNeuTTS.steer_accent(e, "bac", 0.5)
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(e), rel=1e-5)
    unit = lambda x: x / np.linalg.norm(x)
    assert float(unit(out) @ d) > float(unit(e) @ d) + 0.2
    np.testing.assert_allclose(V3TurboVieNeuTTS.steer_accent(e, "bac", 0.0), e, rtol=1e-5)
    with pytest.raises(ValueError):
        V3TurboVieNeuTTS.steer_accent(np.zeros(192), "bac")


def test_speaker_emb_from_audio_equal_weight_per_file_and_skips_silence(tmp_path):
    sr = 16000
    loud = np.full(sr * 20, 0.5, np.float32)
    loud[: sr * 8] = 0.0                       # 8 s im lặng ở đầu → các đoạn này bị bỏ
    quiet = np.full(sr * 4, 0.1, np.float32)   # file ngắn hơn nhiều nhưng vẫn nặng 50%
    sf.write(tmp_path / "loud.wav", loud, sr)
    sf.write(tmp_path / "quiet.wav", quiet, sr)
    t = _bare(tmp_path / "u.json")
    t.engine = _FakeEngine()
    emb = t.speaker_emb_from_audio([tmp_path / "loud.wav", str(tmp_path / "quiet.wav")], window=6.0, step=1.0)
    assert emb.shape == (192,)
    assert emb[0] == pytest.approx(emb[1]) and emb[0] > 0     # hai file nặng như nhau
    assert np.count_nonzero(emb) == 2
    # độ lớn = trung bình độ lớn các đoạn: loud có 15 đoạn 6 s, 3 đoạn đầu im lặng hẳn bị bỏ
    # (đoạn im lặng một phần vẫn tính) → 12 đoạn × 10, cộng 1 đoạn quiet (cả file 4 s) × 12
    assert t.engine.emb_calls == 13
    assert np.linalg.norm(emb) == pytest.approx((12 * 10 + 12) / 13, rel=1e-4)
    with pytest.raises(ValueError):
        t.speaker_emb_from_audio([])
    sf.write(tmp_path / "silent.wav", np.zeros(sr * 7, np.float32), sr)
    with pytest.raises(ValueError):
        t.speaker_emb_from_audio(tmp_path / "silent.wav")


def test_tune_voice_updates_embedding_keeps_codes_and_saves(tmp_path):
    up = tmp_path / "user_voices.json"
    up.write_text(json.dumps({"presets": {"Tùng": _voice(0.5)}}, ensure_ascii=False), encoding="utf-8")
    t = _bare(up)
    t._load_v3_voices()
    t.engine = _FakeEngine()
    before = t.get_preset_voice("Tùng")["speaker_emb"].copy()
    codes = t.get_preset_voice("Tùng")["codes"].copy()

    out = t.tune_voice("Tùng", accent="bac", save=True)
    np.testing.assert_allclose(out, V3TurboVieNeuTTS.steer_accent(before, "bac", 0.5), rtol=1e-6)
    np.testing.assert_array_equal(t.get_preset_voice("Tùng")["codes"], codes)
    saved = json.loads(up.read_text(encoding="utf-8"))["presets"]["Tùng"]
    assert saved["speaker_emb"][0] == pytest.approx(float(out[0]), abs=1e-5)
    assert saved["codes"] == codes.tolist() and saved["description"] == "Nam · test"

    wav = tmp_path / "long.wav"
    sf.write(wav, np.full(16000 * 10, 0.5, np.float32), 16000)
    out2 = t.tune_voice("Tùng", emb_audio=[wav])                  # tính lại từ đầu, không cộng dồn
    assert out2[0] == pytest.approx(10.0) and np.count_nonzero(out2) == 1

    for bad in ({"name": "Adam", "accent": "bac"}, {"name": "khong_co", "accent": "bac"}, {"name": "Tùng"}):
        with pytest.raises(ValueError):
            t.tune_voice(bad.pop("name"), **bad)


def test_save_keeps_voice_tuned_by_another_process(tmp_path):
    """API server đang chạy (phiên 1) lưu giọng khác không được ghi đè giọng CLI (phiên 2) vừa chỉnh."""
    up = tmp_path / "user_voices.json"
    up.write_text(json.dumps({"presets": {"Tùng": _voice(0.5)}}, ensure_ascii=False), encoding="utf-8")
    server = _bare(up)
    server._load_v3_voices()
    server.engine = _FakeEngine(0.3)

    cli = _bare(up)
    cli._load_v3_voices()
    cli.engine = _FakeEngine()
    tuned = cli.tune_voice("Tùng", accent="bac", save=True)

    server.add_voice("clone_B", _ref_wav(tmp_path), save=True)
    on_disk = json.loads(up.read_text(encoding="utf-8"))["presets"]
    assert set(on_disk) == {"Tùng", "clone_B"}
    assert on_disk["Tùng"]["speaker_emb"][0] == pytest.approx(float(tuned[0]), abs=1e-5)

    # giọng phiên 1 tự chỉnh thì phiên 1 vẫn được ghi đè như cũ
    server.tune_voice("Tùng", accent="nam", save=True)
    assert json.loads(up.read_text(encoding="utf-8"))["presets"]["Tùng"]["speaker_emb"][0] != pytest.approx(
        float(tuned[0]), abs=1e-5)


def test_load_audio_mono_decodes_video_with_ffmpeg(tmp_path):
    import shutil
    import subprocess
    from vieneu.v3turbo import load_audio_mono
    wav_path = tmp_path / "a.wav"
    sf.write(wav_path, (0.3 * np.sin(np.arange(16000 * 2) * 0.05)).astype(np.float32), 16000)
    wav, sr = load_audio_mono(wav_path)
    assert sr == 16000 and wav.shape == (32000,)
    with pytest.raises(FileNotFoundError):
        load_audio_mono(tmp_path / "missing.mp4")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg không có trong PATH")
    m4a = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav_path), str(m4a)], check=True)
    wav, sr = load_audio_mono(m4a)
    assert sr == 48000 and abs(len(wav) / sr - 2.0) < 0.1 and float(np.abs(wav).max()) > 0.1
