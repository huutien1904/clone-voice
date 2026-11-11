"""
VieNeu-TTS v3 Turbo backend (PyTorch).
======================================
    from vieneu import Vieneu
    tts = Vieneu(mode="v3turbo")
    wav = tts.infer("Xin chào", ref_audio="ref.wav")   # clone a voice
    wav = tts.infer("Xin chào", voice="Xuân Vĩnh")     # preset voice
    tts.save(wav, "out.wav")

``style`` is DEPRECATED on v3 Turbo: the speaking style is already implied by the
reference voice (speaker embedding + reference codes), so generation is always the
natural style. The argument is still accepted everywhere for backward compatibility,
but it is ignored.
"""
import functools
import threading
import time
import logging
import os
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Union

import numpy as np

from .base import BaseVieneuTTS
from ._v3_turbo_engine.rep_history import DEFAULT_REP_WINDOW
from vieneu_utils.phonemize_text import (
    phonemize_text_with_emotions,
    normalize_to_chunks_v3,
    normalize_to_chunks_v3_with_gaps,
)
from vieneu_utils.core_utils import join_audio_chunks, gaps_to_silence, max_expected_frames


def _cap_frames(sampling: dict, cap: int) -> dict:
    """Bản sao ``sampling`` với ``max_new_frames`` chặn trên bởi ``cap``.

    Chặn phần "nói thêm" khi stop token bắn trượt trên chunk ngắn (xem
    :func:`vieneu_utils.core_utils.max_expected_frames`); chunk dài không bị
    ảnh hưởng vì cap luôn vượt ``max_new_frames`` mặc định.
    """
    out = dict(sampling)
    out["max_new_frames"] = min(out.get("max_new_frames", 300), cap)
    return out

logger = logging.getLogger("Vieneu.V3Turbo")

# Giọng người dùng (clone rồi lưu bằng ``add_voice(..., save=True)``) nằm NGOÀI
# package asset: không ghi đè ``assets/voices_v3_turbo.json`` (20 giọng gốc) và
# sống sót qua ``pip install -U vieneu``. SDK, web UI và API server đều tự nạp
# file này nên giọng đã lưu xuất hiện ở mọi nơi.
USER_VOICES_ENV = "VIENEU_USER_VOICES"


def default_user_voices_path() -> Path:
    """File giọng người dùng: ``$VIENEU_USER_VOICES`` hoặc ``~/.cache/vieneu/user_voices.json``."""
    env = os.environ.get(USER_VOICES_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "vieneu" / "user_voices.json"


def load_audio_mono(path: Union[str, Path], sr: int = 48000) -> Tuple[np.ndarray, int]:
    """Đọc ``path`` thành ``(wav float32 mono, sample_rate)``.

    File audio (wav/flac/mp3/ogg) đọc bằng soundfile. File soundfile không đọc được
    (video mp4/mov/mkv, m4a/aac...) được giải mã bằng ``ffmpeg`` (cần có trong PATH)
    về mono ``sr`` Hz.
    """
    import soundfile as sf
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không thấy file: {p}")
    try:
        wav, file_sr = sf.read(str(p), dtype="float32", always_2d=True)
        return wav.mean(axis=1), int(file_sr)
    except Exception as e:  # noqa: BLE001
        read_err = e
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(f"Không đọc được {p} ({read_err}). Cài ffmpeg để đọc video/m4a.") from read_err
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(p), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        msg = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg không giải mã được {p}: {msg}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), sr


# Vùng miền dùng để chỉnh giọng: tên tham số → giá trị ``region`` trong voices_v3_turbo.json.
def normalize_reference_loudness(wav: np.ndarray, sr: int, target_db: float = -20.0,
                                 peak_db: float = -1.0, max_gain_db: float = 30.0) -> np.ndarray:
    """Đưa giọng mẫu về âm lượng chuẩn: RMS các khung có tiếng ≈ ``target_db`` dBFS, đỉnh ≤ ``peak_db``.

    Mẫu thu nhỏ tiếng (vd video điện thoại để xa, -40 dB) làm giọng clone bị cao và mỏng hơn giọng
    thật (đo: 135 Hz so với 108 Hz); chuẩn hoá đưa về ~112 Hz. Không khuếch đại quá ``max_gain_db``
    để khỏi đẩy tạp âm lên quá mức; đoạn im lặng giữ nguyên."""
    x = np.asarray(wav, dtype=np.float32)
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak < 1e-6:
        return x
    hop = max(1, int(0.02 * sr))
    n = len(x) // hop
    if n < 2:
        rms = float(np.sqrt(np.mean(x ** 2)))
    else:
        fr = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
        loud = fr[fr >= np.median(fr)]
        rms = float(np.sqrt(np.mean(loud ** 2)))
    if rms < 1e-9:
        return x
    gain = min(10 ** (target_db / 20) / rms, 10 ** (peak_db / 20) / peak, 10 ** (max_gain_db / 20))
    return (x * gain).astype(np.float32)


ACCENTS = {"bac": "Bắc", "nam": "Nam"}
# Một khoá cho mọi lần đọc lại/ghi file giọng người dùng trong tiến trình (API chạy nhiều luồng).
_USER_VOICES_LOCK = threading.RLock()


@functools.lru_cache(maxsize=8)
def _accent_direction_cached(voices_file: str, accent: str) -> np.ndarray:
    return _compute_accent_direction(voices_file, accent)


def _accent_direction(voices_file: str, accent: str) -> np.ndarray:
    return _accent_direction_cached(voices_file, accent).copy()


def _compute_accent_direction(voices_file: str, accent: str) -> np.ndarray:
    """Vector đơn vị (192-d) chỉ từ giọng miền khác về giọng ``accent`` trong không gian speaker embedding.

    Tính từ các giọng có sẵn có nhãn ``region``: hiệu trung bình (miền ``accent`` − miền
    còn lại), tính riêng từng giới tính rồi lấy trung bình để giới tính không lẫn vào hướng.
    """
    import json
    if accent not in ACCENTS:
        raise ValueError(f"accent phải là một trong {sorted(ACCENTS)}, nhận được {accent!r}")
    target = ACCENTS[accent]
    presets = json.loads(Path(voices_file).read_text(encoding="utf-8")).get("presets") or {}
    groups: dict = {}
    for v in presets.values():
        region, emb = v.get("region"), v.get("speaker_emb")
        if region not in ACCENTS.values() or emb is None:
            continue
        e = np.asarray(emb, dtype=np.float32).reshape(-1)
        side = "target" if region == target else "other"
        groups.setdefault((v.get("gender") or "", side), []).append(e / np.linalg.norm(e))
    diffs = [np.mean(groups[(g, "target")], 0) - np.mean(groups[(g, "other")], 0)
             for g in {g for g, _ in groups} if (g, "target") in groups and (g, "other") in groups]
    if not diffs:
        raise RuntimeError(f"Không đủ giọng có sẵn gắn nhãn vùng miền trong {voices_file} để tính hướng giọng.")
    d = np.mean(diffs, 0)
    return (d / np.linalg.norm(d)).astype(np.float32)


class V3TurboVieNeuTTS(BaseVieneuTTS):
    """VieNeu-TTS v3 Turbo (PyTorch)."""

    @staticmethod
    def _preclean_reference_audio(
        ref_audio: Union[str, Path],
        *,
        top_db: int = 30,
        out_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Trim silence at the edges and save a cleaned reference clip.
        """
        import os
        import tempfile
        import soundfile as sf

        src = Path(ref_audio)
        if not src.exists():
            return str(src)

        try:
            wav, sr = sf.read(str(src), dtype="float32", always_2d=False)
        except Exception:
            # m4a/aac/mp4... (vd file ghi âm iPhone): soundfile không đọc được → giải mã qua ffmpeg.
            wav, sr = load_audio_mono(src)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype=np.float32)

        try:
            import librosa
            wav_trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
        except Exception:
            nonzero = np.flatnonzero(np.abs(wav) > 1e-5)
            if nonzero.size == 0:
                wav_trimmed = wav
            else:
                start = max(int(nonzero[0]) - int(0.01 * sr), 0)
                end = min(int(nonzero[-1]) + int(0.01 * sr), len(wav))
                wav_trimmed = wav[start:end]

        if out_path is None:
            try:
                secure_dir = Path.home() / ".cache" / "vieneu" / "tmp"
            except Exception:
                secure_dir = Path("./outputs/tmp")

            secure_dir.mkdir(parents=True, exist_ok=True)
            try:
                secure_dir.chmod(0o700)
            except Exception:
                pass

            fd, out_path = tempfile.mkstemp(
                prefix="temp_clone_optimized_",
                suffix=".wav",
                dir=str(secure_dir)
            )
            os.close(fd)
            out_path = Path(out_path)

        sf.write(str(out_path), normalize_reference_loudness(wav_trimmed, sr), sr)
        return str(out_path)

    def __init__(
        self,
        backbone_repo: str = "pnnbao-ump/VieNeu-TTS-v3-Turbo",
        model_subfolder: str = "update",
        moss_tokenizer: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        device: str = "auto",
        dtype: str = "auto",
        backend: str = "auto",   # "auto" → ONNX on CPU, PyTorch on GPU; "onnx"|"pytorch" to force
        onnx_repo: Optional[str] = None,
        onnx_dir: Optional[str] = None,
        precision: str = "int8",   # ONNX/CPU backbone: "int8" (mặc định, nhanh ~3x/frame, nhỏ 4x) | "fp32" (chất-lượng-tối-đa)
        onnx_subfolder: Optional[str] = None,   # override thủ công subfolder; None → suy từ `precision`
        threads: int = 0,   # ONNX/CPU intra-op threads; 0 = mặc định engine (~nhân vật lý, cap 8). Đặt số cụ thể để tinh chỉnh.
        max_batch_size: int = 32,   # GPU/PyTorch: trần số chunk gộp vào một forward (static batching). Batch thực = min(số_chunk, max_batch_size). Bỏ qua trên CPU/ONNX.
        user_voices: Optional[Union[str, Path]] = None,   # file JSON giọng người dùng (clone đã lưu); None → default_user_voices_path()
        **kwargs: Any,
    ):
        super().__init__()
        self.sample_rate = 48_000

        # `precision` chỉ áp cho đường ONNX/CPU (chọn subfolder graph int8 vs fp32).
        # Đường PyTorch/GPU dùng torch fp32/bf16, không liên quan.
        if onnx_subfolder is None:
            onnx_subfolder = {"int8": "onnx_int8", "fp32": "onnx_update"}.get(str(precision).lower(), "onnx_int8")

        if device in (None, "auto"):
            try:
                import torch
                dev_type = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev_type = "cpu"
        else:
            dev_type = "cuda" if "cuda" in str(device).lower() else str(device).lower()
        use_onnx = backend == "onnx" or (backend == "auto" and dev_type == "cpu")

        if use_onnx:
            # Torch-free CPU engine. Reads its ONNX graphs from `onnx_subfolder` in the
            # model repo (uploaded separately).
            from ._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine
            logger.info(f"⏳ Loading VieNeu-TTS v3 Turbo (ONNX/CPU) from: {backbone_repo}/{onnx_subfolder} ...")
            self.engine = OnnxV3LiteEngine(
                checkpoint_path=backbone_repo,
                onnx_repo=onnx_repo,
                onnx_dir=onnx_dir,
                onnx_subfolder=onnx_subfolder,
                threads=threads,
            )
            self.backend = "onnx"
        else:
            from ._v3_turbo_engine import VieNeuTTSv3Turbo
            logger.info(f"⏳ Loading VieNeu-TTS v3 Turbo (PyTorch) from: {backbone_repo}/{model_subfolder} ...")
            self.engine = VieNeuTTSv3Turbo(
                checkpoint_path=backbone_repo,
                model_subfolder=model_subfolder,
                moss_tokenizer_path=moss_tokenizer,
                device=device,
                dtype=dtype,
            )
            self.backend = "pytorch"
        logger.info(f"✅ VieNeu-TTS v3 Turbo ready (backend={self.backend})")

        # Style is deprecated on v3 Turbo: it is implied by the reference (speaker
        # embedding + ref codes), so every generation uses the natural style. Kept
        # only as voice metadata / for backward-compatible call signatures.
        self.default_style = "tu_nhien"
        self._preset_voices: dict = {}
        self._default_voice: Optional[str] = None
        self._user_voice_names: set = set()
        self._builtin_voice_names: set = set()
        self._removed_user_voices: set = set()
        # Giọng người dùng mà phiên này đã thêm/chỉnh: chỉ những giọng này được ghi đè
        # bản trên đĩa khi lưu (giọng khác có thể đã được tiến trình khác chỉnh sau đó).
        self._changed_user_voices: set = set()
        self.user_voices_path: Path = (
            Path(user_voices).expanduser() if user_voices else default_user_voices_path()
        )
        self._load_v3_voices()

        # Static-batching (GPU/PyTorch). Dựng lười ở lần batch đầu; None trên CPU/ONNX.
        self.max_batch_size = max(1, int(max_batch_size))
        self._batch_engine = None

    # ── Preset voices (speaker embedding + reference codes) ─────────────────────
    @staticmethod
    def builtin_voices_path() -> Path:
        """File giọng có sẵn đi kèm package (không nên ghi vào)."""
        return Path(__file__).parent / "assets" / "voices_v3_turbo.json"

    def _load_voices_file(self, path: Path, *, user: bool) -> int:
        """Nạp một file giọng (schema ``{"default_voice"?, "presets": {name: {...}}}``).

        Mỗi giọng mang ``speaker_emb`` 192-d và ``codes`` tham chiếu đã mã hoá.
        ``user=True``: giọng người dùng — tên trùng giọng có sẵn bị bỏ qua (không
        cho ghi đè giọng gốc) và được đánh dấu để ``save_user_voices`` chỉ lưu chúng.
        Trả về số giọng đã nạp; file thiếu/hỏng → 0 và chỉ cảnh báo, không ném lỗi.
        """
        entries, default_voice = self._parse_voices_file(path, user=user)
        for name, voice in entries.items():
            self._preset_voices[name] = voice
            if user:
                self._user_voice_names.add(name)
            else:
                self._builtin_voice_names.add(name)
        if user:
            # Nhớ giọng nào đang nằm trong file: mất khỏi file sau này = bị xoá/đổi tên ở nơi khác.
            self._persisted_user_names = set(getattr(self, "_persisted_user_names", set())) | set(entries)
        if not user and default_voice:
            self._default_voice = default_voice
        return len(entries)

    def _parse_voices_file(self, path: Path, *, user: bool, skip=(),
                           strict: bool = False) -> Tuple[dict, Optional[str]]:
        """Đọc file giọng → ``({tên: giọng}, default_voice)`` mà không sửa trạng thái của đối tượng.

        ``skip``: các tên bỏ qua (vd giọng đã đổi trong phiên này khi nạp lại). ``strict=True``: file
        hỏng/sai schema thì ném ``ValueError`` (để người gọi giữ nguyên giọng đang có) thay vì coi là
        file trống. Đọc được cả file UTF-8 có BOM (Notepad, PowerShell 5.1)."""
        import json
        if not path.exists():
            return {}, None
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as e:  # noqa: BLE001
            if strict:
                raise ValueError(f"File giọng {path} hỏng: {e}") from e
            logger.warning(f"⚠️ Không đọc được file giọng {path}: {e}")
            return {}, None
        presets = data.get("presets") if isinstance(data, dict) else None
        if presets is None:
            presets = {}
        if not isinstance(presets, dict):
            if strict:
                raise ValueError(f"File giọng {path} sai schema (cần {{\"presets\": {{...}}}})")
            logger.warning(f"⚠️ File giọng {path} sai schema (cần {{\"presets\": {{tên: {{...}}}}}}) — bỏ qua.")
            return {}, None
        entries = {}
        for name, v in presets.items():
            if name in skip:
                continue
            if user and name in self._builtin_voice_names:
                logger.warning(f"⚠️ Giọng người dùng '{name}' trùng tên giọng có sẵn — bỏ qua.")
                continue
            if not isinstance(v, dict) or v.get("speaker_emb") is None:
                logger.warning(f"⚠️ Giọng '{name}' trong {path} sai schema hoặc thiếu speaker_emb — bỏ qua.")
                continue
            try:
                emb = np.asarray(v["speaker_emb"], dtype=np.float32).reshape(-1)
                codes = v.get("codes")
                codes = np.asarray(codes, dtype=np.int64) if codes is not None else None
                if emb.size == 0 or (codes is not None and codes.ndim != 2):
                    raise ValueError("speaker_emb rỗng hoặc codes không phải ma trận 2 chiều")
            except (TypeError, ValueError, OverflowError) as e:
                logger.warning(f"⚠️ Giọng '{name}' trong {path} có speaker_emb/codes không hợp lệ ({e}) — bỏ qua.")
                continue
            entries[name] = {
                "description": v.get("description", "") or "",
                "gender": v.get("gender", "") or "",
                "style": v.get("style", self.default_style),
                "speaker_emb": emb,
                "codes": codes,
            }
        default_voice = data.get("default_voice") if isinstance(data, dict) else None
        return entries, default_voice

    def _user_file_stamp(self):
        try:
            st = os.stat(self.user_voices_path)
            # st_ino đổi mỗi lần os.replace: bắt được cả hai lần ghi cùng cỡ trong cùng một tích tắc đồng hồ.
            return (st.st_mtime_ns, st.st_size, st.st_ino)
        except OSError:
            return None

    def reload_user_voices(self, force: bool = False) -> bool:
        """Nạp lại giọng người dùng nếu file đã đổi (giọng vừa lưu từ web, CLI hay tiến trình khác)
        → dùng được ngay, không cần khởi động lại. Trả về True nếu có nạp lại.

        Giọng đã thêm/sửa/xoá trong chính phiên này được giữ nguyên như trong bộ nhớ. Danh sách giọng
        được thay bằng bản mới hoàn chỉnh (không sửa tại chỗ) để yêu cầu khác đang đọc không bị lỗi."""
        with _USER_VOICES_LOCK:
            return self._reload_user_voices_locked(force)

    def _reload_user_voices_locked(self, force: bool) -> bool:
        stamp = self._user_file_stamp()
        if not force and stamp == getattr(self, "_user_voices_stamp", None):
            return False
        protected = set(self._changed_user_voices) | set(self._removed_user_voices)
        try:
            on_disk, _ = self._parse_voices_file(self.user_voices_path, user=True, strict=True)
        except ValueError as e:
            # File đang hỏng (vd sửa tay lỗi cú pháp): giữ nguyên giọng trong bộ nhớ, không coi là trống.
            logger.warning(f"⚠️ {e} — giữ nguyên danh sách giọng hiện có.")
            self._user_voices_stamp = stamp
            return False
        entries = {n: v for n, v in on_disk.items() if n not in protected}
        persisted = set(getattr(self, "_persisted_user_names", set()))
        # Giữ trong bộ nhớ: giọng đổi trong phiên này, và giọng chưa từng lưu (chỉ có trong bộ nhớ).
        keep = {n for n in self._user_voice_names
                if n in self._changed_user_voices or (n not in persisted and n not in on_disk)}
        presets = {n: v for n, v in self._preset_voices.items()
                   if n not in self._user_voice_names or n in keep}
        presets.update(entries)
        users = keep | set(entries)
        self._preset_voices, self._user_voice_names = presets, users
        self._persisted_user_names = set(on_disk)
        if self._default_voice not in self._preset_voices:
            self._default_voice = next(iter(self._preset_voices), None)
        self._user_voices_stamp = stamp
        return True

    def _load_v3_voices(self) -> None:
        """Nạp giọng có sẵn (``assets/voices_v3_turbo.json``) rồi giọng người dùng (``user_voices_path``)."""
        n_builtin = self._load_voices_file(self.builtin_voices_path(), user=False)
        # Lấy dấu thời gian TRƯỚC khi đọc: file đổi ngay trong lúc đọc thì lần reload sau vẫn bắt được.
        self._user_voices_stamp = self._user_file_stamp()
        n_user = self._load_voices_file(self.user_voices_path, user=True)
        if self._default_voice not in self._preset_voices:
            self._default_voice = next(iter(self._preset_voices), None)
        logger.info(f"📢 Loaded {n_builtin} preset voices + {n_user} user voices "
                    f"(default: {self._default_voice}; user file: {self.user_voices_path})")

    def list_preset_voices(self) -> List[tuple]:
        """Return ``[(label, voice_id), ...]`` for every voice (built-in + user)."""
        return [(f"{n} — {v['description']}" if v["description"] else n, n)
                for n, v in list(self._preset_voices.items())]

    def list_voices(self) -> List[dict]:
        """Return ``[{"id", "label", "description", "gender", "custom"}, ...]`` (``custom`` = giọng người dùng)."""
        out = []
        for n, v in list(self._preset_voices.items()):
            desc = v.get("description", "") or ""
            out.append({
                "id": n,
                "label": f"{n} — {desc}" if desc else n,
                "description": desc,
                "gender": v.get("gender", "") or "",
                "custom": n in self._user_voice_names,
            })
        return out

    def is_user_voice(self, name: str) -> bool:
        """True nếu ``name`` là giọng người dùng (clone đã đăng ký), không phải giọng có sẵn."""
        return name in self._user_voice_names

    def get_preset_voice(self, voice_name: Optional[str] = None) -> dict:
        name = voice_name or self._default_voice
        if name not in self._preset_voices:
            raise ValueError(f"Voice '{name}' not found. Available: {list(self._preset_voices)}")
        return self._preset_voices[name]

    def encode_reference(self, ref_audio: Union[str, Path], denoise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Enroll a voice from a wav → ``(speaker_emb, ref_codes)``.
        """
        import os
        clean_ref = self._preclean_reference_audio(ref_audio)
        try:
            return self.engine.prepare_reference(str(clean_ref), denoise=denoise, use_ref_codes=True)
        finally:
            if clean_ref and Path(clean_ref).resolve() != Path(ref_audio).resolve():
                try:
                    os.remove(clean_ref)
                except Exception:
                    pass

    def denoise(self, ref_audio: Union[str, Path], out_path: Optional[Union[str, Path]] = None,
                max_seconds: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Clean up a reference clip and return ``(wav, sample_rate)`` at 44.1 kHz.

        Pass ``out_path`` to also save the cleaned wav. ``max_seconds`` optionally
        trims the clip first. Use the result as a nicer reference for cloning.
        """
        import soundfile as sf
        den = getattr(self.engine, "denoiser", None)
        if den is None:
            raise RuntimeError("Denoiser không khả dụng trên backend này.")
        wav, sr = sf.read(str(ref_audio))
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype=np.float32)
        if max_seconds and len(wav) > int(max_seconds * sr):
            wav = wav[: int(max_seconds * sr)]
        clean = den.denoise(wav, sr)          # -> float32 mono @ 44100
        if out_path is not None:
            sf.write(str(out_path), clean, 44100)
        return clean, 44100

    def add_voice(self, name: str, ref_audio: Union[str, Path], *, denoise: bool = True,
                  use_ref_codes: bool = True, description: str = "", gender: str = "",
                  style: Any = None, save: bool = False,
                  speaker_emb: Optional[np.ndarray] = None) -> str:
        """Register a custom voice under ``name`` for use as ``infer(..., voice=name)``.

        ``ref_audio`` is enrolled once (denoised + trimmed, then speaker embedding +
        reference codes are extracted). Pass ``denoise=False`` if the clip is already
        clean. ``save=True`` ghi giọng vào file giọng người dùng (``user_voices_path``:
        ``$VIENEU_USER_VOICES`` hoặc ``~/.cache/vieneu/user_voices.json``) để các phiên
        sau (SDK, web UI, API server) tự nạp lại. File asset gốc không bị đụng.

        ``name`` không được trùng giọng có sẵn (không cho ghi đè giọng gốc); trùng
        giọng người dùng cũ thì thay thế.

        ``speaker_emb``: dùng dấu vân giọng tính sẵn thay cho dấu vân của ``ref_audio`` (vd trung
        bình trên cả file dài từ :meth:`reference_profile`, đã đẩy giọng miền); ``ref_audio`` khi đó
        chỉ dùng để lấy mã giọng mẫu.

        ``style`` is deprecated: it is stored as metadata only and never changes how
        the voice is synthesized (the style is implied by the reference itself).
        """
        name = str(name or "").strip()
        if not name:
            raise ValueError("Tên giọng không được để trống.")
        if name in self._builtin_voice_names or (name in self._preset_voices and name not in self._user_voice_names):
            raise ValueError(f"'{name}' là giọng có sẵn, không thể ghi đè. Hãy chọn tên khác.")
        override_emb = speaker_emb
        clean_ref = self._preclean_reference_audio(ref_audio)
        try:
            speaker_emb, ref_codes = self.engine.prepare_reference(
                str(clean_ref), denoise=denoise, use_ref_codes=use_ref_codes)
            if override_emb is not None:
                speaker_emb = override_emb
        finally:
            if clean_ref and Path(clean_ref).resolve() != Path(ref_audio).resolve():
                try:
                    os.remove(clean_ref)
                except Exception:
                    pass
        previous = self._preset_voices.get(name)
        self._preset_voices[name] = {
            "description": description or "",
            "gender": gender or "",
            "style": style or self.default_style,   # metadata only (deprecated)
            "speaker_emb": np.asarray(speaker_emb, dtype=np.float32).reshape(-1),
            "codes": None if ref_codes is None else np.asarray(ref_codes, dtype=np.int64),
        }
        self._user_voice_names.add(name)
        self._removed_user_voices.discard(name)
        self._changed_user_voices.add(name)
        if not self._default_voice:
            self._default_voice = name
        if save:
            try:
                self.save_user_voices()
            except Exception:
                # Ghi đĩa thất bại → không giữ giọng "nửa vời" trong phiên; báo lỗi cho caller.
                if previous is None:
                    self._preset_voices.pop(name, None)
                    self._user_voice_names.discard(name)
                    self._changed_user_voices.discard(name)
                    if self._default_voice == name:
                        self._default_voice = next(iter(self._preset_voices), None)
                else:
                    self._preset_voices[name] = previous
                raise
        logger.info(f"➕ Added voice '{name}'.")
        return name

    def remove_voice(self, name: str, save: bool = False) -> None:
        """Remove a registered voice by name.

        Giọng có sẵn chỉ bị gỡ khỏi phiên hiện tại (file asset không đổi);
        ``save=True`` ghi lại file giọng người dùng (bỏ giọng vừa xoá nếu là giọng người dùng).
        """
        self._preset_voices.pop(name, None)
        if name in self._user_voice_names or name not in self._builtin_voice_names:
            self._removed_user_voices.add(name)
        self._user_voice_names.discard(name)
        self._changed_user_voices.discard(name)
        if self._default_voice == name:
            self._default_voice = next(iter(self._preset_voices), None)
        if save:
            self.save_user_voices()

    # ── Chỉnh giọng người dùng: embedding trung bình + vùng miền ────────────────
    def speaker_emb_from_audio(self, audio: Union[str, Path, List[Union[str, Path]]], *,
                               window: float = 6.0, step: float = 1.0, robust: bool = False) -> np.ndarray:
        """Speaker embedding trung bình trên một hoặc nhiều file audio/video dài của cùng người nói.

        Mỗi file được cắt thành đoạn ``window`` giây (bước ``step``), trích embedding từng
        đoạn rồi lấy trung bình hướng; mỗi file nặng như nhau dù dài ngắn khác nhau. Kết
        quả cùng thang độ lớn với embedding thường và ổn định hơn embedding từ một đoạn
        mẫu 3-8 giây. Video/m4a cần ffmpeg (xem :func:`load_audio_mono`).

        ``robust=True``: mỗi file bỏ dần 30% đoạn lệch xa giọng chính nhất (lặp 3 lần) trước
        khi lấy trung bình — hợp với video/stream có nhạc, tiếng game hay người khác chen vào.
        """
        paths = [audio] if isinstance(audio, (str, Path)) else list(audio)
        if not paths:
            raise ValueError("Cần ít nhất một file audio/video.")
        centroids, norms = [], []
        for p in paths:
            wav, sr = load_audio_mono(p)
            if wav.size == 0:
                raise ValueError(f"File rỗng: {p}")
            win = min(int(window * sr), len(wav))
            hop = max(1, int(step * sr))
            peak = float(np.abs(wav).max())
            units, file_norms = [], []
            for s in range(0, len(wav) - win + 1, hop):
                seg = wav[s:s + win]
                if peak == 0.0 or float(np.sqrt(np.mean(seg ** 2))) < 1e-3 * peak:   # bỏ đoạn im lặng
                    continue
                e = np.asarray(self.engine.extract_speaker_emb(seg, sr), dtype=np.float32).reshape(-1)
                n = float(np.linalg.norm(e))
                if n > 0:
                    file_norms.append(n)
                    units.append(e / n)
            if not units:
                raise ValueError(f"Không có đoạn có tiếng nói nào trong {p}.")
            U = np.stack(units)
            keep = np.ones(len(U), dtype=bool)
            c = U.mean(0)
            c /= np.linalg.norm(c)
            if robust and len(U) >= 5:
                for _ in range(3):
                    sim = U @ c
                    keep = sim >= np.percentile(sim, 30)
                    c = U[keep].mean(0)
                    c /= np.linalg.norm(c)
            centroids.append(c)
            norms.extend(np.asarray(file_norms)[keep].tolist())
        avg = np.mean(centroids, 0)
        return (avg / np.linalg.norm(avg) * float(np.mean(norms))).astype(np.float32)

    def reference_profile(self, audio: Union[str, Path], *, window: float = 6.0, step: float = 1.0,
                          clip: float = 8.0, min_first_sim: float = 0.80,
                          min_first_speech: float = 0.5,
                          max_seconds: float = 120.0) -> Tuple[np.ndarray, float, float]:
        """Phân tích một file giọng mẫu dài trong một lượt → ``(speaker_emb, bắt_đầu, kết_thúc)``.

        - ``speaker_emb``: trung bình các đoạn ``window`` giây, bỏ dần 30% đoạn lệch giọng chính
          nhất (nhạc, tiếng game, người khác chen vào) — như ``speaker_emb_from_audio(robust=True)``.
        - ``bắt_đầu``/``kết_thúc`` (giây): đoạn ``clip`` giây để lấy mã giọng mẫu. Mặc định là đoạn mở
          đầu (chỗ có tiếng đầu tiên); chỉ đổi sang đoạn điểm cao nhất (giống giọng chính + nhiều tiếng
          nói) khi đoạn mở đầu rõ ràng hỏng: giống giọng chính dưới ``min_first_sim`` (nhạc, người
          khác) hoặc có tiếng nói dưới ``min_first_speech``. Đo thực tế: đoạn "giống giọng chính
          nhất" chưa chắc đọc ra giống hơn đoạn mở đầu, nên chỉ tự cắt khi đoạn mở đầu thật sự tệ.
        """
        wav, sr = load_audio_mono(audio)
        if wav.size == 0:
            raise ValueError(f"File rỗng: {audio}")
        if max_seconds and len(wav) > int(max_seconds * sr):
            wav = wav[: int(max_seconds * sr)]       # chỉ phân tích phần đầu: file dài cả giờ không làm đầy RAM
        total = len(wav) / sr
        win = min(int(window * sr), len(wav))
        hop = max(1, int(step * sr))
        peak = float(np.abs(wav).max())
        fl = max(1, int(0.05 * sr))

        def frame_db(x):
            n = len(x) // fl
            if n == 0:
                return np.array([20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)])
            return 20 * np.log10(np.sqrt(np.mean(x[: n * fl].reshape(n, fl) ** 2, axis=1)) + 1e-9)

        thr = float(np.percentile(frame_db(wav), 95)) - 25.0   # khung "có tiếng nói"
        starts, units, norms, covs = [], [], [], []
        for s in range(0, len(wav) - win + 1, hop):
            seg = wav[s:s + win]
            if peak == 0.0 or float(np.sqrt(np.mean(seg ** 2))) < 1e-3 * peak:
                continue
            e = np.asarray(self.engine.extract_speaker_emb(seg, sr), dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(e))
            if n > 0:
                starts.append(s)
                units.append(e / n)
                norms.append(n)
                covs.append(float((frame_db(seg) > thr).mean()))
        if not units:
            raise ValueError(f"Không có đoạn có tiếng nói nào trong {audio}.")
        U, norms, covs, starts = np.stack(units), np.asarray(norms), np.asarray(covs), np.asarray(starts)
        keep = np.ones(len(U), dtype=bool)
        c = U.mean(0)
        c /= np.linalg.norm(c)
        if len(U) >= 5:
            for _ in range(3):
                sim = U @ c
                keep = sim >= np.percentile(sim, 30)
                c = U[keep].mean(0)
                c /= np.linalg.norm(c)
        emb = (c * float(norms[keep].mean())).astype(np.float32)

        # Điểm đoạn mẫu: trung bình điểm các cửa sổ nằm trọn trong [t, t + clip].
        sims = U @ c
        score = 2.0 * sims + covs
        span = max(0, int(round((clip - window) * sr)))
        inside = [(starts >= s) & (starts <= s + span) for s in starts]
        clip_scores = [float(score[m].mean()) for m in inside]
        first_ok = float(sims[inside[0]].mean()) >= min_first_sim and float(covs[inside[0]].mean()) >= min_first_speech
        best = 0 if first_ok else int(np.argmax(clip_scores))   # đoạn mở đầu ổn → giữ nguyên
        start = float(starts[best]) / sr
        end = min(start + clip, total)
        if end - start < clip:                      # sát cuối file: lùi điểm bắt đầu cho đủ dài
            start = max(0.0, end - clip)
        return emb, round(start, 2), round(end, 2)

    @classmethod
    def accent_direction(cls, accent: str = "bac") -> np.ndarray:
        """Vector đơn vị chỉ về giọng ``accent`` ("bac" | "nam"), tính từ các giọng có sẵn gắn nhãn vùng miền."""
        return _accent_direction(str(cls.builtin_voices_path()), accent)

    @classmethod
    def steer_accent(cls, speaker_emb: np.ndarray, accent: str = "bac", strength: float = 0.5) -> np.ndarray:
        """Đẩy ``speaker_emb`` về giọng miền ``accent``, giữ nguyên độ lớn.

        Model không có tuỳ chọn vùng miền: nó đoán giọng Bắc/Nam từ speaker embedding,
        nên clone của người miền Bắc có thể bị đọc giọng Nam. ``strength`` là bước đi theo
        hướng vùng miền trên embedding đã chuẩn hoá: 0.25 nhẹ, 0.5 vừa (thường đủ ra giọng
        rõ), 0.8 mạnh (dễ lệch âm sắc).
        """
        e = np.asarray(speaker_emb, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(e))
        if n == 0.0:
            raise ValueError("speaker_emb toàn 0.")
        u = e / n + float(strength) * cls.accent_direction(accent)
        return (u / np.linalg.norm(u) * n).astype(np.float32)

    def tune_voice(self, name: str, *, emb_audio: Union[None, str, Path, List[Union[str, Path]]] = None,
                   accent: Optional[str] = None, accent_strength: float = 0.5,
                   window: float = 6.0, save: bool = False) -> np.ndarray:
        """Chỉnh speaker embedding của giọng người dùng ``name``; reference codes giữ nguyên.

        - ``emb_audio``: file audio/video dài của cùng người nói → thay embedding bằng
          embedding trung bình (:meth:`speaker_emb_from_audio`), giọng giống người thật hơn.
        - ``accent``: "bac" hoặc "nam" → đẩy embedding về giọng miền đó (:meth:`steer_accent`).

        Chỉ truyền ``accent`` thì đẩy tiếp từ embedding hiện tại, nên gọi nhiều lần sẽ cộng
        dồn; truyền kèm ``emb_audio`` để luôn tính lại từ đầu. Trả về embedding mới.
        """
        if name not in self._user_voice_names or name not in self._preset_voices:
            raise ValueError(f"'{name}' không phải giọng người dùng (chỉ chỉnh được giọng đã clone).")
        if not emb_audio and not accent:
            raise ValueError("Cần emb_audio và/hoặc accent.")
        previous = self._preset_voices[name]
        if emb_audio:
            emb = self.speaker_emb_from_audio(emb_audio, window=window)
        else:
            emb = np.asarray(previous["speaker_emb"], dtype=np.float32).reshape(-1)
        if accent:
            emb = self.steer_accent(emb, accent, accent_strength)
        self._preset_voices[name] = {**previous, "speaker_emb": emb}
        was_changed = name in self._changed_user_voices
        self._changed_user_voices.add(name)
        if save:
            try:
                self.save_user_voices()
            except Exception:
                self._preset_voices[name] = previous
                if not was_changed:
                    self._changed_user_voices.discard(name)
                raise
        logger.info(f"🎚️ Tuned voice '{name}' (emb_audio={'yes' if emb_audio else 'no'}, "
                    f"accent={accent or '-'}, strength={accent_strength if accent else '-'}).")
        return emb

    def _voice_to_json(self, v: dict) -> dict:
        emb = v.get("speaker_emb")
        codes = v.get("codes")
        return {
            "description": v.get("description", "") or "",
            "gender": v.get("gender", "") or "",
            "style": v.get("style", self.default_style),
            "speaker_emb": [round(float(x), 6) for x in np.asarray(emb).reshape(-1)] if emb is not None else None,
            "codes": np.asarray(codes, dtype=int).tolist() if codes is not None else None,
        }

    def save_user_voices(self, path: Optional[Union[str, Path]] = None) -> str:
        """Ghi CHỈ các giọng người dùng ra JSON (mặc định ``user_voices_path``), ghi atomic.

        Gộp với nội dung đang có trên đĩa: giọng do tiến trình khác (CLI, web UI, API)
        thêm vào hoặc chỉnh sau khi phiên này nạp file vẫn được giữ; phiên này chỉ ghi
        đè các giọng nó đã ``add_voice``/``tune_voice`` (hoặc giọng đã mất khỏi đĩa), và
        chỉ giọng phiên này đã ``remove_voice`` mới bị bỏ. Trả về đường dẫn file.
        """
        with _USER_VOICES_LOCK:
            return self._save_user_voices_locked(path)

    def _save_user_voices_locked(self, path) -> str:
        import json
        import shutil
        import tempfile
        path = Path(path).expanduser() if path else self.user_voices_path
        presets: dict = {}
        disk_ok = True   # False = file hỏng → ghi lại toàn bộ giọng trong bộ nhớ (không coi giọng nào "đã bị xoá")
        if path.exists():
            try:
                data_on_disk = json.loads(path.read_text(encoding="utf-8-sig"))
                if not isinstance(data_on_disk, dict):
                    raise ValueError("sai schema (cần {\"presets\": {...}})")
                on_disk = data_on_disk.get("presets")
                if on_disk is None:
                    on_disk = {}                      # file hợp lệ nhưng chưa có giọng nào
                if not isinstance(on_disk, dict):
                    raise ValueError("sai schema (cần {\"presets\": {...}})")
                presets = {n: v for n, v in on_disk.items()
                           if isinstance(v, dict) and n not in self._builtin_voice_names}
            except Exception as e:  # noqa: BLE001
                disk_ok = False
                bak = path.with_name(f"{path.stem}.hong-{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}")
                try:
                    shutil.copy2(path, bak)   # giữ lại bản hỏng để còn cứu dữ liệu bằng tay
                except Exception:  # noqa: BLE001
                    bak = None
                logger.warning(f"⚠️ Không đọc được {path} để gộp ({e}); sẽ ghi lại từ bộ nhớ"
                               + (f", bản hỏng lưu ở {bak}" if bak else ""))
        try:
            is_main = path.resolve() == Path(self.user_voices_path).expanduser().resolve()
        except Exception:  # noqa: BLE001
            is_main = False
        for n in self._removed_user_voices:
            presets.pop(n, None)
        # Giọng từng nằm trong file chính mà nay biến mất = bị xoá/đổi tên ở tiến trình khác → không ghi lại
        # ("hồi sinh"); giọng chưa từng lưu (chỉ có trong bộ nhớ) thì ghi vào như trước.
        old_persisted = set(getattr(self, "_persisted_user_names", set())) if (is_main and disk_ok) else set()
        gone = set()
        for n in list(self._user_voice_names):
            if n not in self._preset_voices:
                continue
            if n in self._changed_user_voices:
                presets[n] = self._voice_to_json(self._preset_voices[n])
            elif n not in presets:
                if n in old_persisted:
                    gone.add(n)
                else:
                    presets[n] = self._voice_to_json(self._preset_voices[n])
        data = {"meta": {"note": "VieNeu user voices (cloned): speaker embedding + reference codes",
                         "count": len(presets)},
                "presets": presets}
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
                except Exception:
                    pass
        if is_main:
            # File chính đã khớp bộ nhớ với các giọng vừa ghi/xoá → bỏ đánh dấu "đổi trong phiên" để
            # reload_user_voices() sau tôn trọng thay đổi từ tiến trình khác; bỏ khỏi bộ nhớ các giọng
            # đã bị xoá ở nơi khác. Xuất ra file khác (bản sao) thì giữ nguyên đánh dấu.
            self._persisted_user_names = set(presets)
            self._changed_user_voices.clear()
            self._removed_user_voices.clear()
            if gone:
                self._preset_voices = {k: v for k, v in self._preset_voices.items() if k not in gone}
                self._user_voice_names = self._user_voice_names - gone
                if self._default_voice in gone:
                    self._default_voice = next(iter(self._preset_voices), None)
        logger.info(f"💾 Saved {len(presets)} user voices → {path}")
        return str(path)

    def save_voices(self, path: Optional[Union[str, Path]] = None) -> str:
        """Export ALL current voices (built-in + user) to one JSON file.

        Mặc định ghi đè file asset của package — chỉ dùng khi cố ý xuất bộ giọng mới.
        Muốn lưu giọng vừa clone, dùng ``add_voice(..., save=True)`` / ``save_user_voices``.
        """
        import json
        path = Path(path) if path else self.builtin_voices_path()
        presets = {n: self._voice_to_json(v) for n, v in self._preset_voices.items()}
        data = {"meta": {"note": "v3 turbo voices: speaker embedding + reference codes"},
                "default_voice": self._default_voice, "presets": presets}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info(f"💾 Saved {len(presets)} voices → {path}")
        return str(path)

    def _resolve_ref(self, voice, ref_audio, denoise, use_ref_codes) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Resolve the requested voice to ``(speaker_emb, ref_codes)``.

        Precedence: cloned ``ref_audio`` → preset ``voice`` (name or dict) → default preset.
        """
        if ref_audio is not None:
            import os
            clean_ref = self._preclean_reference_audio(ref_audio)
            try:
                return self.engine.prepare_reference(str(clean_ref), denoise=denoise, use_ref_codes=use_ref_codes)
            finally:
                if clean_ref and Path(clean_ref).resolve() != Path(ref_audio).resolve():
                    try:
                        os.remove(clean_ref)
                    except Exception:
                        pass
        preset = None
        if isinstance(voice, str):
            preset = self._preset_voices.get(voice)
            if preset is None:
                raise ValueError(f"Voice '{voice}' not found. Available: {list(self._preset_voices)}")
        elif isinstance(voice, dict):
            preset = voice
        elif self._default_voice:
            preset = self._preset_voices[self._default_voice]
        if preset is None:
            raise ValueError("Provide a preset `voice` name or a `ref_audio` to clone.")
        codes = preset.get("codes") if use_ref_codes else None
        return preset["speaker_emb"], codes

    # ── Batched inference (GPU/PyTorch) ──────────────────────────────────────
    def _get_batch_engine(self):
        """Lazily build the static-batching engine. Returns ``None`` on CPU/ONNX.

        The batching runtime (``vieneu.v3_turbo_serve``) is pure PyTorch/CUDA, so it
        only applies to the ``pytorch`` backend. Importing it is deferred until the
        first batched call to keep the torch-free CPU path from touching torch.
        """
        if self.backend != "pytorch":
            return None
        if self._batch_engine is None:
            from .v3_turbo_serve import V3TurboBatchEngine
            self._batch_engine = V3TurboBatchEngine(self.engine)
        return self._batch_engine

    def _infer_chunks(
        self,
        chunks: List[str],
        speaker_emb: np.ndarray,
        ref_codes: Optional[np.ndarray],
        use_ref_codes: bool,
        batch_size: int,
        sampling: dict,
    ) -> List[np.ndarray]:
        """Synthesize one waveform per chunk (no join, no watermark).

        On the PyTorch/GPU backend with ``batch_size > 1`` and more than one chunk,
        the chunks are pushed through the static-batching engine in groups of
        ``batch_size`` (chunks share each forward step — the throughput win). Groups
        are formed over chunks SORTED by phoneme length (length bucketing): batching
        similar-length prompts together minimizes left-padding, which cuts wasted
        prefill compute and padding-induced numeric noise. On CPU (ONNX) or a single
        chunk, each chunk goes through the single-sequence engine sequentially.
        Output order always matches ``chunks``.
        """
        n = len(chunks)
        engine = self._get_batch_engine() if (batch_size > 1 and n > 1) else None

        if engine is None:
            wavs: List[np.ndarray] = []
            for chunk in chunks:
                ph = phonemize_text_with_emotions(chunk)
                wavs.append(self.engine.infer(
                    phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                    use_ref_codes=use_ref_codes,
                    **_cap_frames(sampling, max_expected_frames(ph)),
                ))
            return wavs

        phs = [phonemize_text_with_emotions(c) for c in chunks]
        order = sorted(range(n), key=lambda i: len(phs[i]))
        wavs = [None] * n
        for i in range(0, n, batch_size):
            idxs = order[i:i + batch_size]
            reqs = [{
                "phonemes": phs[j],
                "speaker_emb": speaker_emb, "ref_codes": ref_codes,
                "use_ref_codes": use_ref_codes,
            } for j in idxs]
            # Trần chung cho cả group = trần của row dài nhất — group đã được
            # bucket theo độ dài phoneme nên trần vẫn sát với từng row.
            group_cap = max(max_expected_frames(phs[j]) for j in idxs)
            for j, w in zip(idxs, engine.generate_batch(reqs, **_cap_frames(sampling, group_cap))):
                wavs[j] = w
        return wavs

    # ── Public API ───────────────────────────────────────────────────────────
    def infer(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: Any = None,   # DEPRECATED: bỏ qua — style đã nằm trong ref code (luôn tự nhiên)
        denoise: bool = True,
        use_ref_codes: bool = True,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_chars: int = 256,
        silence_p: float = 0.15,
        crossfade_p: float = 0.0,
        apply_watermark: bool = True,
        batch_size: Optional[int] = None,   # GPU: trần chunk/forward (None → self.max_batch_size; 1 → tắt batch)
        **kwargs: Any,
    ) -> np.ndarray:
        """Synthesize ``text`` into one 48 kHz waveform.

        ``style`` is deprecated and ignored (kept only so older code keeps running):
        the reading style comes from the reference voice, so output is always the
        natural style.
        """
        speaker_emb, ref_codes = self._resolve_ref(voice, ref_audio, denoise, use_ref_codes)

        chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)

        bs = self.max_batch_size if batch_size is None else max(1, int(batch_size))
        sampling = dict(
            temperature=temperature, top_k=top_k, top_p=top_p,
            max_new_frames=max_new_frames, repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
        # GPU gộp các chunk vào cùng forward; CPU/1-chunk chạy tuần tự (xem _infer_chunks).
        all_wavs = self._infer_chunks(
            chunks, speaker_emb, ref_codes, use_ref_codes, bs, sampling
        )

        # Im lặng theo loại ranh giới: ngắt đoạn > hết câu > ngắt trong câu.
        final_wav = join_audio_chunks(
            all_wavs, self.sample_rate, silence_ps=gaps_to_silence(gaps)
        )
        return self._apply_watermark(final_wav) if apply_watermark else final_wav

    def infer_stream(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: Any = None,   # DEPRECATED: bỏ qua (xem `infer`)
        denoise: bool = True,
        use_ref_codes: bool = True,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_chars: int = 256,
        apply_watermark: bool = True,
        **kwargs: Any,
    ) -> Generator[np.ndarray, None, None]:
        speaker_emb, ref_codes = self._resolve_ref(voice, ref_audio, denoise, use_ref_codes)
        chunks = normalize_to_chunks_v3(text, max_chars=max_chars)
        # Prefer the engine's native frame-level streaming (low first-audio latency);
        # both the PyTorch and ONNX engines expose infer_stream. Fall back to one
        # full infer per chunk if not available.
        stream_fn = getattr(self.engine, "infer_stream", None)
        for chunk in chunks:
            ph = phonemize_text_with_emotions(chunk)
            chunk_frames_cap = min(max_new_frames, max_expected_frames(ph))
            if stream_fn is not None:
                for sub in stream_fn(
                    phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                    use_ref_codes=use_ref_codes,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    max_new_frames=chunk_frames_cap, repetition_penalty=repetition_penalty,
                    repetition_window=repetition_window,
                ):
                    if sub is None or len(sub) == 0:
                        continue
                    yield self._apply_watermark(sub) if apply_watermark else sub
            else:
                wav = self.engine.infer(
                    phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                    use_ref_codes=use_ref_codes,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    max_new_frames=chunk_frames_cap, repetition_penalty=repetition_penalty,
                    repetition_window=repetition_window,
                )
                yield self._apply_watermark(wav) if apply_watermark else wav

    def infer_batch(
        self,
        texts: List[str],
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: Any = None,   # DEPRECATED: bỏ qua (xem `infer`)
        denoise: bool = True,
        use_ref_codes: bool = True,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_chars: int = 256,
        apply_watermark: bool = True,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        """Synthesize many texts, returning one waveform each (same order as ``texts``).

        All texts share one voice (resolved once). On the PyTorch/GPU backend the
        chunks from *every* text are flattened and batched together in groups of
        ``batch_size`` — so even many short texts fill the batch and share forward steps.
        On CPU (ONNX) this runs sequentially. Output equals per-text ``infer`` (equivalent,
        not bit-identical to the single path, since sampling runs batched).
        """
        if not texts:
            return []

        speaker_emb, ref_codes = self._resolve_ref(voice, ref_audio, denoise, use_ref_codes)
        bs = self.max_batch_size if batch_size is None else max(1, int(batch_size))
        sampling = dict(
            temperature=temperature, top_k=top_k, top_p=top_p,
            max_new_frames=max_new_frames, repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )

        # Cắt chunk từng text, nhớ chunk thuộc text nào (owner) và gaps để join lại.
        per_text_gaps: List[Any] = []
        flat_chunks: List[str] = []
        owner: List[int] = []
        for ti, t in enumerate(texts):
            chunks, gaps = normalize_to_chunks_v3_with_gaps(t, max_chars=max_chars)
            per_text_gaps.append(gaps)
            for c in chunks:
                flat_chunks.append(c)
                owner.append(ti)

        empty = np.array([], dtype=np.float32)
        if not flat_chunks:
            return [empty for _ in texts]

        flat_wavs = self._infer_chunks(
            flat_chunks, speaker_emb, ref_codes, use_ref_codes, bs, sampling
        )

        # Gom wav về đúng text (thứ tự flat_wavs khớp flat_chunks/owner), rồi join từng text.
        grouped: List[List[np.ndarray]] = [[] for _ in texts]
        for w, ti in zip(flat_wavs, owner):
            grouped[ti].append(w)

        results: List[np.ndarray] = []
        for ti in range(len(texts)):
            if not grouped[ti]:
                results.append(empty)
                continue
            joined = join_audio_chunks(
                grouped[ti], self.sample_rate, silence_ps=gaps_to_silence(per_text_gaps[ti])
            )
            results.append(self._apply_watermark(joined) if apply_watermark else joined)
        return results

    def close(self) -> None:
        self.engine = None
        self._batch_engine = None
