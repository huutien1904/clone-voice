"""
Quản lý giọng người dùng (clone → lưu bền) cho Clone-voice-TTS v3 Turbo.
====================================================================
Giọng lưu trong ``~/.cache/vieneu/user_voices.json`` (hoặc ``$VIENEU_USER_VOICES``)
và tự xuất hiện trong SDK (``voice="tên"``), web UI (dropdown Giọng mẫu sau khi
tải model) và API server (``GET /voices``).

    uv run --frozen python -m apps.voice_tool add clone_A outputs/clone/mau_A.wav --description "Nam · Giọng từ video"
    uv run --frozen python -m apps.voice_tool list
    uv run --frozen python -m apps.voice_tool remove clone_A
    uv run --frozen python -m apps.voice_tool rename clone_A "clone_bản tin vip"
    uv run --frozen python -m apps.voice_tool path

Clone thẳng từ video, giống người thật hơn và giữ giọng miền Bắc:

    uv run --frozen python -m apps.voice_tool add "Tùng" video2.mp4 --start 12.93 --end 20.53 --no-denoise \
        --emb-from video1.mp4 video2.mp4 --accent bac --test outputs/thu.wav

    # chỉnh lại giọng đã có (không cần cắt mẫu lại)
    uv run --frozen python -m apps.voice_tool tune "Tùng" --emb-from video1.mp4 video2.mp4 --accent bac

- ``--start/--end``: cắt đoạn mẫu (nên 3-8 giây, một người nói) từ file audio/video dài.
- ``--emb-from``: audio/video dài của cùng người → dấu vân giọng trung bình, ổn định hơn.
- ``--accent bac|nam`` + ``--accent-strength`` (mặc định 0.5): đẩy giọng về miền đó.
  Model tự đoán giọng miền từ dấu vân giọng nên clone người Bắc có thể bị đọc giọng Nam.

``add``/``tune`` cần nạp model (~5-10 s); ``list``/``remove``/``path`` chạy tức thì.
Đọc video/m4a cần ffmpeg trong PATH. API server/web UI đang chạy tự nhận giọng vừa
thêm/chỉnh/đổi tên: API dùng được ngay, web thì tải lại trang (F5).
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _user_path() -> Path:
    from vieneu.v3turbo import default_user_voices_path
    return default_user_voices_path()


def _builtin_names() -> list:
    from vieneu.v3turbo import V3TurboVieNeuTTS
    p = V3TurboVieNeuTTS.builtin_voices_path()
    if not p.exists():
        return []
    try:
        return list((json.loads(p.read_text(encoding="utf-8")).get("presets") or {}).keys())
    except Exception:  # noqa: BLE001
        return []


def _read_user(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")).get("presets") or {}
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Không đọc được {path}: {e}")
        return {}


def _write_user(path: Path, presets: dict) -> None:
    data = {"meta": {"note": "Clone-voice-TTS user voices (cloned): speaker embedding + reference codes",
                     "count": len(presets)}, "presets": presets}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".user_voices_", suffix=".json", dir=str(path.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:  # noqa: BLE001
                pass


DEFAULT_TEST_TEXT = "Xin chào, đây là giọng vừa được thêm vào danh sách. Dạo này trời ra sao, về nhà rồi gọi lại nhé."
RESTART_HINT = ("ℹ️  API server đang chạy dùng được giọng mới ngay; web UI đang mở thì "
                "tải lại trang (F5) để thấy trong danh sách.")


def _prepare_ref(src: Path, start, end) -> tuple:
    """Đọc ``src`` (audio hoặc video), cắt ``[start, end]`` giây nếu có → ``(wav tạm, thời lượng)``."""
    import soundfile as sf
    from vieneu.v3turbo import load_audio_mono
    wav, sr = load_audio_mono(src)
    a = int(round((start or 0.0) * sr))
    b = int(round(end * sr)) if end is not None else len(wav)
    if not 0 <= a < b <= len(wav):
        raise ValueError(f"Đoạn cắt {start}-{end}s nằm ngoài file ({len(wav) / sr:.2f}s).")
    fd, tmp = tempfile.mkstemp(prefix="vieneu_ref_", suffix=".wav")
    os.close(fd)
    sf.write(tmp, wav[a:b], sr)
    return Path(tmp), (b - a) / sr


def _check_tune_files(args) -> bool:
    for f in args.emb_from or []:
        if not Path(f).exists():
            print(f"❌ Không thấy file --emb-from: {f}")
            return False
    return True


def _load_tts():
    from vieneu import Vieneu
    t0 = time.time()
    print("⏳ Đang nạp model để trích đặc trưng giọng...")
    tts = Vieneu()
    print(f"   model sẵn sàng sau {time.time() - t0:.1f}s (backend={tts.backend})")
    return tts


def _apply_tune(tts, args) -> None:
    if not (args.emb_from or args.accent):
        return
    if args.emb_from:
        print(f"🎚️  Tính dấu vân giọng trung bình từ {len(args.emb_from)} file...")
    tts.tune_voice(args.name, emb_audio=args.emb_from or None, accent=args.accent,
                   accent_strength=args.accent_strength)
    if args.accent:
        print(f"🎚️  Đẩy giọng về miền {args.accent} (strength={args.accent_strength})")


def _write_test(tts, args) -> None:
    if args.test:
        out = Path(args.test)
        audio = tts.infer(args.test_text, voice=args.name)
        tts.save(audio, out)
        print(f"🔊 Đã sinh thử → {out} ({len(audio) / tts.sample_rate:.2f}s)")


def cmd_add(args) -> int:
    ref = Path(args.ref_audio)
    if not ref.exists():
        print(f"❌ Không thấy file audio mẫu: {ref}")
        return 1
    if args.name in _builtin_names():
        print(f"❌ '{args.name}' là giọng có sẵn, hãy chọn tên khác.")
        return 1
    if not _check_tune_files(args):
        return 1
    try:
        clip, dur = _prepare_ref(ref, args.start, args.end)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"❌ {e}")
        return 1
    if not 3.0 <= dur <= 8.0:
        print(f"⚠️ Đoạn mẫu dài {dur:.2f}s; nên 3-8 giây (model chỉ dùng 8 giây đầu).")
    try:
        tts = _load_tts()
        tts.add_voice(args.name, str(clip), denoise=not args.no_denoise,
                      description=args.description, gender=args.gender, save=False)
        _apply_tune(tts, args)
        tts.save_user_voices()
    finally:
        try:
            os.remove(clip)
        except OSError:
            pass
    print(f"✅ Đã lưu giọng '{args.name}' → {tts.user_voices_path}")
    _write_test(tts, args)
    print(RESTART_HINT)
    return 0


def cmd_tune(args) -> int:
    if not (args.emb_from or args.accent):
        print("❌ Cần --emb-from và/hoặc --accent.")
        return 1
    if args.name not in _read_user(_user_path()):
        print(f"❌ Không có giọng người dùng tên '{args.name}' trong {_user_path()}")
        return 1
    if not _check_tune_files(args):
        return 1
    if args.accent and not args.emb_from:
        print("⚠️ Chỉ có --accent: giọng được đẩy tiếp từ dấu vân hiện tại (chạy nhiều lần sẽ cộng dồn).")
    tts = _load_tts()
    _apply_tune(tts, args)
    tts.save_user_voices()
    print(f"✅ Đã chỉnh giọng '{args.name}' → {tts.user_voices_path}")
    _write_test(tts, args)
    print(RESTART_HINT)
    return 0


def cmd_list(args) -> int:
    path = _user_path()
    user = _read_user(path)
    builtin = _builtin_names()
    print(f"📁 File giọng người dùng: {path} ({'có' if path.exists() else 'chưa có'})")
    print(f"🎙️  Giọng có sẵn: {len(builtin)}  |  Giọng người dùng: {len(user)}")
    for name, v in user.items():
        desc = v.get("description") or ""
        print(f"   - {name}" + (f" — {desc}" if desc else ""))
    if args.all:
        for name in builtin:
            print(f"   · {name} (có sẵn)")
    return 0


def cmd_remove(args) -> int:
    path = _user_path()
    user = _read_user(path)
    if args.name not in user:
        print(f"❌ Không có giọng người dùng tên '{args.name}' trong {path}")
        return 1
    user.pop(args.name)
    _write_user(path, user)
    print(f"🗑️  Đã xoá '{args.name}'. Còn {len(user)} giọng người dùng.")
    return 0


def cmd_rename(args) -> int:
    path = _user_path()
    user = _read_user(path)
    old, new = args.old.strip(), args.new.strip()
    if old not in user:
        print(f"❌ Không có giọng người dùng tên '{old}' trong {path}")
        return 1
    if not new:
        print("❌ Tên mới không được để trống.")
        return 1
    if new == old and args.description is None:
        print("ℹ️ Tên mới trùng tên cũ, không đổi gì.")
        return 0
    if new != old and new in user:
        print(f"❌ Đã có giọng người dùng tên '{new}'.")
        return 1
    if new in _builtin_names():
        print(f"❌ '{new}' là tên giọng có sẵn, hãy chọn tên khác.")
        return 1
    # Giữ nguyên thứ tự và toàn bộ dữ liệu giọng (speaker_emb, codes, ...), chỉ đổi tên.
    user = {(new if k == old else k): v for k, v in user.items()}
    if args.description is not None:
        user[new]["description"] = args.description
    _write_user(path, user)
    print(f"✏️  Đã đổi tên giọng '{old}' → '{new}'.")
    print(RESTART_HINT)
    return 0


def cmd_path(args) -> int:
    print(_user_path())
    return 0


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Quản lý giọng người dùng Clone-voice-TTS (clone đã lưu)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def tune_args(sp):
        sp.add_argument("--emb-from", nargs="+", metavar="FILE",
                        help="Audio/video dài của cùng người nói → dấu vân giọng trung bình (giống người thật hơn)")
        sp.add_argument("--accent", choices=["bac", "nam"], help="Đẩy giọng về miền Bắc/Nam")
        sp.add_argument("--accent-strength", type=float, default=0.5,
                        help="Mức đẩy vùng miền: 0.25 nhẹ, 0.5 vừa (mặc định), 0.8 mạnh")
        sp.add_argument("--test", metavar="OUT.wav", help="Sinh thử một câu bằng giọng này và lưu ra file")
        sp.add_argument("--test-text", default=DEFAULT_TEST_TEXT, help="Câu dùng cho --test")

    a = sub.add_parser("add", help="Đăng ký giọng mới từ file audio/video mẫu và lưu bền")
    a.add_argument("name", help="ID giọng, vd clone_A")
    a.add_argument("ref_audio", help="File audio mẫu 3-8 giây, hoặc file audio/video dài kèm --start/--end")
    a.add_argument("--start", type=float, help="Giây bắt đầu đoạn mẫu trong ref_audio")
    a.add_argument("--end", type=float, help="Giây kết thúc đoạn mẫu trong ref_audio")
    a.add_argument("--description", default="", help="Mô tả hiển thị, vd 'Nam · Giọng từ video X'")
    a.add_argument("--gender", default="", help="male/female (tuỳ chọn)")
    a.add_argument("--no-denoise", action="store_true", help="Không khử nhiễu audio mẫu (nên dùng khi audio đã sạch)")
    tune_args(a)
    a.set_defaults(fn=cmd_add)

    t = sub.add_parser("tune", help="Chỉnh giọng người dùng đã có: dấu vân trung bình và/hoặc vùng miền")
    t.add_argument("name", help="ID giọng người dùng, vd Tùng")
    tune_args(t)
    t.set_defaults(fn=cmd_tune)

    l = sub.add_parser("list", help="Liệt kê giọng người dùng")
    l.add_argument("--all", action="store_true", help="Liệt kê cả giọng có sẵn")
    l.set_defaults(fn=cmd_list)

    r = sub.add_parser("remove", help="Xoá giọng người dùng")
    r.add_argument("name")
    r.set_defaults(fn=cmd_remove)

    rn = sub.add_parser("rename", help="Đổi tên giọng người dùng (giữ nguyên giọng)")
    rn.add_argument("old", help="Tên hiện tại, vd Tùng")
    rn.add_argument("new", help="Tên mới, vd clone_admin_Tùng")
    rn.add_argument("--description", help="Đổi luôn mô tả hiển thị (tuỳ chọn)")
    rn.set_defaults(fn=cmd_rename)

    sub.add_parser("path", help="In đường dẫn file giọng người dùng").set_defaults(fn=cmd_path)

    args = p.parse_args(argv)
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
