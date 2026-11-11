import sys
import io
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import gradio as gr
print("⏳ Đang khởi động VieNeu-TTS... Vui lòng chờ...")
import soundfile as sf
import tempfile
from vieneu import Vieneu
from vieneu.v3turbo import ACCENTS
from apps.emotion_script import (
    PRESET_EMOTIONS, emotion_sets, emotion_voice_name, example_script, norm_label,
    parse_emotion_script, split_emotion_voice_name, validate_emotion, validate_set_name,
)
import os
import time
import numpy as np
import queue
import threading
import yaml
import uuid
from vieneu_utils.core_utils import join_audio_chunks, env_bool, get_silence_duration_v2, gaps_to_silence
from vieneu_utils.phonemize_text import phonemize_to_chunks, normalize_to_chunks, normalize_to_chunks_v3, normalize_to_chunks_v3_with_gaps
# PuncNormalizer = sea_g2p.Normalizer luôn bật punc_norm=True.
from vieneu_utils.phonemize_text import PuncNormalizer as Normalizer
import gc

# PDF text extraction
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    fitz = None

from apps.ui_utils import (
    _format_duration,
    _split_estimate_status,
    wrap_with_estimate,
    cleanup_gpu_memory,
    get_ref_text_cached,
    on_codec_change,
    validate_audio_duration,
    on_custom_id_change
)
from apps.ui_constants import (
    theme,
    css,
    head_html,
    DEFAULT_TEXT_GPU,
    DEFAULT_TEXT_TURBO,
    DEFAULT_TEXT_V3
)

# --- CONSTANTS & CONFIG ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}
except Exception as e:
    raise RuntimeError(f"Không thể đọc config.yaml: {e}")

BACKBONE_CONFIGS = _config.get("backbone_configs", {})
CODEC_CONFIGS = _config.get("codec_configs", {})

# Refilter and Simplify Configs per requirements
HAS_GPU = False
try:
    import torch
    HAS_GPU = torch.cuda.is_available() or (sys.platform == "darwin" and torch.backends.mps.is_available())
except ImportError:
    pass

filtered_backbones = {}

# VieNeu-TTS v3 Turbo (early access) — PyTorch, runs on both CPU and GPU.
if not HAS_GPU:
    filtered_backbones["VieNeu-TTS-v3-Turbo (int8)"] = {
        "repo": "pnnbao-ump/VieNeu-TTS-v3-Turbo",
        "precision": "int8",
        "supports_streaming": False,
        "description": "🆕 v3 Turbo (int8) — 48kHz, TỐI ƯU CHO CPU: nhanh nhất (backbone nén int8, ~3x/frame, nhẹ 4x). Khuyến nghị cho máy CPU. Giọng mặc định dùng speaker token; Voice Cloning clone từ audio mẫu; tag cảm xúc [cười]/[hắng giọng]/[thở dài] (thử nghiệm)."
    }
filtered_backbones["VieNeu-TTS-v3-Turbo"] = {
    "repo": "pnnbao-ump/VieNeu-TTS-v3-Turbo",
    "precision": "fp32",
    "supports_streaming": False,
    "description": (
        "🆕 v3 Turbo — 48kHz. Giọng mặc định dùng speaker token (ổn định hơn); Voice Cloning "
        "clone từ audio mẫu; tag cảm xúc [cười]/[hắng giọng]/[thở dài] (thử nghiệm)."
        + ("" if HAS_GPU else " Trên CPU đây là bản chất-lượng-tối-đa (chậm hơn bản int8).")
    )
}

# GPU-only extras. On GPU the default is VieNeu-TTS-v2 (GPU); v3 Turbo stays the
# default (and only option) on CPU machines (the v2/v1 GGUF CPU builds were removed).
if HAS_GPU:
    filtered_backbones["VieNeu-TTS-v2 (GPU)"] = {
        "repo": "pnnbao-ump/VieNeu-TTS-v2",
        "supports_streaming": False,
        "description": "VieNeu-TTS Version 2 - hỗ trợ song ngữ (Anh-Việt) và chế độ podcast"
    }
    filtered_backbones["VieNeu-TTS (GPU)"] = {
        "repo": "pnnbao-ump/VieNeu-TTS",
        "supports_streaming": False,
        "description": "VieNeu-TTS Version 1 - ổn định, production-ready"
    }

BACKBONE_CONFIGS = filtered_backbones

filtered_codecs = {
    "NeuCodec (Distill)": {
        "repo": "neuphonic/distill-neucodec",
        "description": "Codec mặc định cho model GPU",
        "use_preencoded": False
    },
    "NeuCodec (ONNX)": {
        "repo": "neuphonic/neucodec-onnx-decoder-int8",
        "description": "Codec siêu nhẹ, tối ưu cho CPU (ONNX)",
        "use_preencoded": False
    },
    "VieNeu-Codec": {
        "repo": "pnnbao-ump/VieNeu-Codec",
        "description": "Codec tối ưu cho Turbo v2 (ONNX)",
        "use_preencoded": False
    }
}
CODEC_CONFIGS = filtered_codecs

_text_settings = _config.get("text_settings", {})
MAX_CHARS_PER_CHUNK = _text_settings.get("max_chars_per_chunk", 256)
MAX_TOTAL_CHARS_STREAMING = _text_settings.get("max_total_chars_streaming", 3000)

if not BACKBONE_CONFIGS or not CODEC_CONFIGS:
    raise ValueError("config.yaml thiếu backbone_configs hoặc codec_configs")

# --- 1. MODEL CONFIGURATION ---
# Global model instance
tts = None
current_backbone = None
current_codec = None
model_loaded = False
using_lmdeploy = False
PRESET_VOICES_CACHE = []  # List of all voices (tuples or strings)
CONV_VOICES_CACHE = []    # Filtered list for conversation (podcast=True)
MAX_SPEAKERS = 8          # Max concurrent speakers in conversation tab

# Normalizer (module-level singleton)
_text_normalizer = Normalizer()

def get_available_devices() -> list[str]:
    """Get list of available devices for current platform."""
    devices = ["Auto", "CPU"]
    
    try:
        import torch
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            devices.append("MPS")
        elif torch.cuda.is_available():
            devices.append("CUDA")
    except ImportError:
        pass

    return devices


# Nhãn dễ hiểu cho người dùng phổ thông; giá trị bên trong (dùng trong code) giữ nguyên.
DEVICE_LABELS = {
    "Auto": "Tự động (khuyên dùng)",
    "CPU": "CPU (bộ xử lý)",
    "CUDA": "Card đồ hoạ NVIDIA",
    "MPS": "Chip Apple (GPU)",
}
BACKBONE_LABELS = {
    "VieNeu-TTS-v3-Turbo (int8)": "VieNeu-TTS v3 · Nhanh (khuyên dùng)",
    "VieNeu-TTS-v3-Turbo": "VieNeu-TTS v3 · Chất lượng cao nhất (chậm hơn)",
    "Custom Model": "Model tuỳ chỉnh (nâng cao)",
}
NOT_LOADED_MSG = ("⏳ Chưa tải model. Mở mục **🛠️ Cài đặt mô hình** rồi bấm **🔄 Tải Model** "
                  "(mất khoảng 10–30 giây).")
# Link chia sẻ cho người ngoài: ẩn giọng clone đã lưu và tắt nút Lưu giọng (VIENEU_WEB_HIDE_USER_VOICES=1).
_HIDE_USER_VOICES = env_bool("VIENEU_WEB_HIDE_USER_VOICES", default=False)
# Tải sẵn model mặc định khi khởi động web (tắt bằng VIENEU_WEB_PRELOAD=0).
_PRELOAD_ENABLED = env_bool("VIENEU_WEB_PRELOAD", default=True)
CLONE_GUIDE_V3 = (
    "**Cách dùng giọng của bạn:**\n"
    "1. Bấm **Record** rồi đọc to, rõ khoảng **5–8 giây**, xong bấm **Stop**. Nếu trình duyệt hỏi "
    "dùng micro, chọn **Cho phép**. Máy không có micro, hoặc đã có sẵn file ghi âm/video, thì bấm nút "
    "**📁 Tải file ghi âm / video lên** ngay dưới khung ghi âm.\n"
    "2. Chỉ **một người** nói, ở chỗ **yên tĩnh**, không có nhạc nền.\n"
    "3. Bấm **🔊 Đọc thành tiếng** ở bước ③.\n"
    "4. Ưng giọng rồi? Đặt tên và bấm **💾 Lưu giọng** ở dưới để lần sau dùng lại ngay.\n\n"
    "Câu gợi ý để đọc: *“Xin chào, hôm nay trời đẹp quá. Mình đang thử tạo giọng đọc bằng chính "
    "giọng nói của mình.”*"
)
PREVIEW_TEXT = "Xin chào, tôi là {name}. Rất vui được đọc cho bạn nghe."
QUEUED_MSG = ("⏳ **Đã nhận!** Nếu có người khác đang dùng, bạn sẽ được xếp hàng chờ tới lượt. "
              "Đừng tải lại trang trong lúc chờ.")
# Ước lượng độ dài audio: tiếng Việt đọc khoảng 4 chữ (âm tiết) mỗi giây. Bản JS chạy ngay trên
# trình duyệt khi gõ (không gửi cả văn bản lên máy chủ mỗi lần gõ phím).
_LENGTH_JS = """(t) => {
    const s = (t || "").trim();
    if (!s) return "";
    const words = s.split(/\\s+/).length;
    const secs = Math.max(1, Math.round(words / 4));
    const m = Math.floor(secs / 60), r = secs % 60;
    return "📏 " + words + " chữ · đọc ra khoảng " + (m > 0 ? m + " phút " + r + " giây" : r + " giây");
}"""


def _length_info(text: str) -> str:
    words = len((text or "").split())
    if not words:
        return ""
    secs = max(1, round(words / 4))
    m, r = divmod(secs, 60)
    return f"📏 {words} chữ · đọc ra khoảng " + (f"{m} phút {r} giây" if m else f"{r} giây")


def device_choices(devices: list) -> list:
    """Lựa chọn cho ô Thiết bị: ``[(nhãn hiển thị, giá trị)]``."""
    return [(DEVICE_LABELS.get(d, d), d) for d in devices]


def backbone_choices(names: list) -> list:
    """Lựa chọn cho ô Mô hình: ``[(nhãn hiển thị, giá trị)]``."""
    return [(BACKBONE_LABELS.get(n, n), n) for n in names]


def _has_nvidia_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _running_on() -> str:
    """Model đang chạy trên thiết bị nào, nói theo cách người dùng phổ thông hiểu."""
    if getattr(tts, "backend", None) == "onnx" or "(CPU)" in (current_backbone or ""):
        return "CPU của máy chủ"
    if using_lmdeploy:
        return "card đồ hoạ NVIDIA"
    dev = getattr(getattr(tts, "engine", None), "device", None)
    dev_type = getattr(dev, "type", None) or str(dev or "").split(":")[0].lower()
    if dev_type == "cuda":
        return "card đồ hoạ NVIDIA"
    if dev_type == "mps":
        return "chip Apple (GPU)"
    if dev_type == "cpu":
        return "CPU của máy chủ"
    try:
        import torch
        if torch.cuda.is_available():
            return "card đồ hoạ NVIDIA"
        if torch.backends.mps.is_available():
            return "chip Apple (GPU)"
    except Exception:
        pass
    return "CPU của máy chủ"


def _supports_cloning(backbone_choice: str) -> bool:
    """Voice Cloning availability by model.

    v3+ clones directly from a sample audio; VieNeu-TTS-v2 (GPU) clones from
    audio plus a reference transcript. v1 and the CPU/Turbo builds stay
    preset-only.
    """
    c = (backbone_choice or "").lower()
    return "v3" in c or c == "vieneu-tts-v2 (gpu)"

def get_model_status_message() -> str:
    """Thông báo trạng thái model bằng lời dễ hiểu (không có thuật ngữ backend/codec)."""
    global model_loaded, tts, using_lmdeploy, current_backbone, current_codec
    if not model_loaded or tts is None:
        return NOT_LOADED_MSG

    model_name = BACKBONE_LABELS.get(current_backbone, current_backbone or "model")
    return (
        f"✅ **Sẵn sàng!** Đã tải **{model_name}**, đang chạy trên **{_running_on()}**.\n\n"
        "Nhập văn bản, chọn giọng rồi bấm **🔊 Đọc thành tiếng**."
    )

def restore_ui_state():
    """Update UI components based on persistence"""
    global model_loaded
    msg = get_model_status_message()
    return (
        msg, 
        gr.update(interactive=model_loaded), # btn_generate
        gr.update(interactive=model_loaded), # btn_generate_conv
        gr.update(interactive=False)         # btn_stop
    )

def should_use_lmdeploy(backbone_choice: str, device_choice: str) -> bool:
    """Determine if we should use LMDeploy backend."""
    # LMDeploy not supported on macOS
    if sys.platform == "darwin":
        return False

    # GGUF, v2-Turbo và v3 Turbo đều KHÔNG dùng LMDeploy (v3 là PyTorch, có engine riêng).
    bc = backbone_choice.lower()
    if "gguf" in bc or "v2-turbo" in bc or "v3" in bc:
        return False
    
    try:
        import torch
        if device_choice == "Auto":
            has_gpu = torch.cuda.is_available()
        elif device_choice == "CUDA":
            has_gpu = torch.cuda.is_available()
        else:
            has_gpu = False
        return has_gpu
    except ImportError:
        return False

_STYLE_VI = {
    "Phong cách tin tức": "giọng đọc tin tức",
    "Phong cách tự nhiên": "giọng tự nhiên",
    "Phong cách kể chuyện": "giọng kể chuyện",
    "Phong cách đọc truyện": "giọng đọc truyện",
    "Giọng đọc tự nhiên": "giọng tự nhiên",
}


def _friendly_voice_label(label):
    """'Trúc Ly — Nữ · Bắc · Phong cách tự nhiên' → '👩 Trúc Ly — nữ, miền Bắc, giọng tự nhiên'.

    Nhãn gốc "Nam · Nam" (giới tính · miền) rất dễ hiểu nhầm; nhãn không đúng mẫu giữ nguyên."""
    if not isinstance(label, str) or " — " not in label:
        return label
    name, desc = label.split(" — ", 1)
    parts = [p.strip() for p in desc.split("·")]
    if len(parts) < 2 or parts[0] not in ("Nam", "Nữ") or parts[1] not in ("Bắc", "Nam", "Trung"):
        return label
    bits = [parts[0].lower(), f"miền {parts[1]}"]
    if len(parts) > 2 and parts[2]:
        style = _STYLE_VI.get(parts[2], parts[2])
        bits.append(style[:1].lower() + style[1:])
    return f"{'👩' if parts[0] == 'Nữ' else '👨'} {name} — {', '.join(bits)}"


def _voice_choice_updates(selected=None) -> tuple:
    """Dựng lại danh sách giọng cho ô "Giọng đọc" và các ô giọng ở tab hội thoại
    → ``(voice_update, slot_update)``. ``selected``: giọng chọn sẵn (mặc định: giọng mặc định)."""
    global PRESET_VOICES_CACHE, CONV_VOICES_CACHE
    try:
        voices = tts.list_preset_voices()
    except Exception:
        voices = []

    is_user = getattr(tts, "is_user_voice", lambda _n: False)
    if voices and _HIDE_USER_VOICES:
        # Link chia sẻ cho người ngoài: ẩn các giọng clone đã lưu (giọng người thật).
        voices = [v for v in voices if not is_user(v[1] if isinstance(v, tuple) else v)]

    if voices:
        default_v = tts._default_voice
        is_tuple = isinstance(voices[0], tuple)
        if is_tuple:
            voices = [(("🎙️ " + lbl) if is_user(vid) else _friendly_voice_label(lbl), vid)
                      for lbl, vid in voices]
        voice_values = [v[1] for v in voices] if is_tuple else voices
        if not default_v and voice_values:
            default_v = voice_values[0]
        if default_v and default_v not in voice_values:
            voices.append((default_v, default_v) if is_tuple else default_v)
        if is_tuple:
            voices.sort(key=lambda x: str(x[0]))
        else:
            voices.sort()
        if selected and selected in voice_values:
            default_v = selected
        voice_update = gr.update(choices=voices, value=default_v, interactive=True)
        PRESET_VOICES_CACHE = voices

        # Giọng cho tab hội thoại (podcast=True; chấp nhận cả "True"/"False" dạng chuỗi).
        def _check_podcast(v_id):
            val = tts._preset_voices.get(v_id, {}).get('podcast', True)
            if isinstance(val, str):
                return val.strip().lower() == "true"
            return bool(val)

        CONV_VOICES_CACHE = [v for v in voices if _check_podcast(v[1] if is_tuple else v)]
        slot_dd_update = gr.update(choices=CONV_VOICES_CACHE)
    else:
        msg = "⚠️ Không có giọng có sẵn. Hãy dùng mục Giọng của bạn (clone)."
        voice_update = gr.update(choices=[msg], value=msg, interactive=False)
        slot_dd_update = gr.update(choices=[])
    return voice_update, slot_dd_update


def _ready_ui_updates(backbone_choice: str) -> tuple:
    """Cập nhật giao diện sau khi model đã sẵn sàng (mọi output của nút Tải Model, trừ dòng trạng thái):
    bật nút, điền danh sách giọng, hiện tab clone/hội thoại. Dùng cả khi một phiên mới mở trang."""
    voice_update, slot_dd_update = _voice_choice_updates()
    is_v2 = backbone_choice in ("VieNeu-TTS-v2 (GPU)", "VieNeu-TTS-v2 (CPU)")
    is_v3 = "v3" in (backbone_choice or "").lower()
    return (
        gr.update(interactive=True),   # btn_generate
        gr.update(interactive=True),   # btn_generate_conv
        gr.update(interactive=True),   # btn_load
        gr.update(interactive=False),  # btn_stop
        voice_update,
        gr.update(visible=True),                               # tab_preset
        gr.update(visible=_supports_cloning(backbone_choice)),  # tab_custom
        gr.update(selected="preset_mode"),                     # tabs
        "preset_mode",                                         # current_mode_state
        gr.update(visible=is_v2 or is_v3),                     # conv_tab
        *([slot_dd_update] * MAX_SPEAKERS),
    )


# --- Tải sẵn model khi khởi động: người dùng mở trang là dùng được, khỏi bấm "Tải Model" ---
_MODEL_LOAD_LOCK = threading.Lock()
_PRELOAD_DONE = threading.Event()
PRELOADING_MSG = "⏳ **Đang chuẩn bị giọng đọc...** Lần đầu mất khoảng 10–30 giây, bạn cứ nhập văn bản trong lúc chờ."
PRELOAD_FAILED_MSG = ("⚠️ Chưa chuẩn bị được giọng đọc. Mở mục **🛠️ Cài đặt mô hình** ở trên "
                      "rồi bấm **🔄 Tải Model**.")


def _preload_default_model(backbone_choice: str, codec_choice: str) -> None:
    try:
        for _ in load_model(backbone_choice, codec_choice, "Auto", False, "", "", ""):
            pass
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Không tải sẵn được model: {e}")
    finally:
        _PRELOAD_DONE.set()


_PRELOAD_START_LOCK = threading.Lock()
_PRELOAD_STARTED = False


def _ensure_preload_started() -> None:
    """Bắt đầu tải sẵn model ở nền (một lần duy nhất)."""
    global _PRELOAD_STARTED
    with _PRELOAD_START_LOCK:
        if _PRELOAD_STARTED:
            return
        _PRELOAD_STARTED = True
    threading.Thread(target=_preload_default_model, args=(default_backbone, default_codec),
                     daemon=True, name="vieneu-preload").start()


def on_page_load():
    """Mỗi lần có người mở trang: chờ model tải xong rồi điền danh sách giọng cho phiên đó
    (khỏi phải bấm "Tải Model")."""
    if _PRELOAD_ENABLED:
        _ensure_preload_started()
    waiting_shown = False
    deadline = time.time() + 600
    while not model_loaded and (_MODEL_LOAD_LOCK.locked()
                                or (_PRELOAD_ENABLED and not _PRELOAD_DONE.is_set())):
        if not waiting_shown:
            yield (PRELOADING_MSG, gr.update(interactive=False), gr.update(interactive=False),
                   *[gr.update()] * (8 + MAX_SPEAKERS))
            waiting_shown = True
        if time.time() > deadline:
            break
        time.sleep(0.5)
    if model_loaded and tts is not None:
        # Giọng vừa lưu từ phiên khác / voice_tool / API cũng hiện ra, không cần khởi động lại.
        if hasattr(tts, "reload_user_voices"):
            try:
                tts.reload_user_voices()
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ Không nạp lại được giọng người dùng: {e}")
        yield (get_model_status_message(), *_ready_ui_updates(current_backbone))
    else:
        yield (PRELOAD_FAILED_MSG if _PRELOAD_ENABLED else NOT_LOADED_MSG,
               gr.update(interactive=False), gr.update(interactive=False),
               *[gr.update()] * (8 + MAX_SPEAKERS))


def validate_clone_audio(audio_path):
    """Nhận xét độ dài giọng mẫu ngay khi người dùng ghi âm/tải lên (model dùng tối đa 8 giây đầu)."""
    if not audio_path:
        return gr.update(visible=False)
    try:
        dur = _probe_duration(audio_path)
    except Exception:
        return gr.update(value="⚠️ Không đọc được file này. Hãy dùng file wav, mp3, m4a hoặc ghi âm trực tiếp.",
                         visible=True)
    if dur < 3.0:
        return gr.update(value=f"⚠️ Giọng mẫu hơi ngắn ({dur:.1f} giây). Nên nói **5–8 giây** để giống hơn.",
                         visible=True)
    if dur > 8.5:
        return gr.update(value=(f"ℹ️ Giọng mẫu dài {dur:.0f} giây, app chỉ dùng **8 giây đầu**. "
                                "Hãy để đoạn nói rõ nhất ở đầu."), visible=True)
    return gr.update(value=f"✅ Giọng mẫu dài {dur:.1f} giây, vừa đẹp.", visible=True)


_CLONE_DIRECT_EXTS = (".wav", ".mp3", ".flac", ".ogg")


def on_clone_file_upload(path):
    """Nút "📁 Tải file ghi âm / video lên" (dễ thấy hơn icon ⬆ nhỏ của khung audio).

    File không phải wav/mp3/flac/ogg (m4a của iPhone, video mp4...) được ffmpeg tách tiếng ra wav
    (tối đa 2 phút đầu, để máy ít RAM không bị đầy) cho khung audio phát và cắt được."""
    import shutil
    import subprocess
    if not path:
        return gr.update()
    if os.path.splitext(str(path))[1].lower() in _CLONE_DIRECT_EXTS:
        return path
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        gr.Warning("Máy chủ chưa cài ffmpeg nên chỉ nhận file wav, mp3, flac hoặc ogg.")
        return gr.update()
    fd, out = tempfile.mkstemp(prefix="vieneu_clone_", suffix=".wav")
    os.close(fd)
    p = subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "48000",
                        "-t", "120", out], capture_output=True)
    if p.returncode != 0 or os.path.getsize(out) < 1000:
        gr.Warning("Không đọc được tiếng trong file này. Hãy thử file khác.")
        return gr.update()
    return out


_PREVIEW_CACHE: dict = {}


def preview_voice(voice_choice):
    """▶ Nghe thử: đọc một câu ngắn bằng giọng đang chọn (lưu lại để lần sau phát ngay)."""
    if not model_loaded or tts is None:
        gr.Warning("Giọng đọc đang được chuẩn bị, vui lòng đợi vài giây rồi bấm lại.")
        return gr.update(visible=False)
    if not voice_choice or "⚠️" in str(voice_choice):
        gr.Warning("Hãy chọn một giọng trước.")
        return gr.update(visible=False)
    if "v3" not in (current_backbone or "").lower():
        gr.Warning("Nghe thử chỉ có với mô hình VieNeu-TTS v3.")
        return gr.update(visible=False)
    try:
        v_id = resolve_voice_id(voice_choice)
    except ValueError as e:
        gr.Warning(str(e))
        return gr.update(visible=False)
    import hashlib
    try:
        emb = np.asarray(tts.get_preset_voice(v_id)["speaker_emb"], dtype=np.float32)
        fp = hashlib.sha1(emb.tobytes()).hexdigest()[:16]   # giọng làm lại cùng tên → bản nghe thử mới
    except Exception:  # noqa: BLE001
        fp = None
    key = (current_backbone, v_id, fp)
    path = _PREVIEW_CACHE.get(key)
    if not path or not os.path.exists(path):
        wav = tts.infer(PREVIEW_TEXT.format(name=str(v_id).replace("_", " ")), voice=v_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, wav, getattr(tts, "sample_rate", 48000))
            path = tmp.name
        _PREVIEW_CACHE[key] = path
    return gr.update(value=path, visible=True)


def load_model(backbone_choice: str, codec_choice: str, device_choice: str,
               force_lmdeploy: bool, custom_model_id: str = "", custom_base_model: str = "",
               custom_hf_token: str = ""):
    """Tải model; chỉ một lượt tải chạy tại một thời điểm (tránh tải chồng, tốn gấp đôi RAM).
    Cùng cấu hình với model đang chạy thì không tải lại (chỉ làm mới giao diện)."""
    global _LOADED_SETTINGS
    key = (backbone_choice, codec_choice, device_choice,
           bool(force_lmdeploy) and "v3" not in (backbone_choice or "").lower(),
           (custom_model_id or "").strip(), custom_base_model or "")
    if model_loaded and tts is not None and key == _LOADED_SETTINGS:
        if hasattr(tts, "reload_user_voices"):
            try:
                tts.reload_user_voices()   # giọng vừa thêm/đổi bằng voice_tool hiện ra ngay
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ Không nạp lại được giọng người dùng: {e}")
        yield (get_model_status_message() + "\n\n(Model này đang chạy sẵn, không cần tải lại.)",
               *_ready_ui_updates(backbone_choice))
        return
    if not _MODEL_LOAD_LOCK.acquire(blocking=False):
        yield ("⏳ Model đang được tải ở một phiên khác, vui lòng đợi rồi tải lại trang.",
               *[gr.update()] * (10 + MAX_SPEAKERS))
        return
    try:
        yield from _load_model_impl(backbone_choice, codec_choice, device_choice, force_lmdeploy,
                                    custom_model_id, custom_base_model, custom_hf_token)
        if model_loaded:
            _LOADED_SETTINGS = key
    finally:
        _MODEL_LOAD_LOCK.release()


_LOADED_SETTINGS = None  # cấu hình của lần tải model thành công gần nhất


def _load_model_impl(backbone_choice: str, codec_choice: str, device_choice: str,
               force_lmdeploy: bool, custom_model_id: str = "", custom_base_model: str = "",
               custom_hf_token: str = ""):
    """Load model with optimizations and max batch size control"""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    lmdeploy_error_reason = None
    model_loaded = False # Ensure we don't try to use a half-loaded model
    
    # Helper for slot updates (initially no change)
    slot_no_updates = [gr.update()] * MAX_SPEAKERS

    yield (
        "⏳ Đang tải model, vui lòng chờ khoảng 10–30 giây...",
        gr.update(interactive=False), # btn_generate
        gr.update(interactive=False), # btn_generate_conv
        gr.update(interactive=False), # btn_load
        gr.update(interactive=False), # btn_stop
        gr.update(), # voice_select
        gr.update(), gr.update(), gr.update(), gr.update(), # tab_p, tab_c, tab_sel, mode_state
        gr.update(), # conv_tab
        *slot_no_updates
    )
    
    try:
        # Cleanup before loading new model
        if tts is not None:
            tts = None # Reset instead of del to avoid NameError if load fails
            cleanup_gpu_memory()
        
        # Prepare Backbone Config/Repo
        custom_loading = False
        is_merged_lora = False

        if backbone_choice == "Custom Model":
            custom_loading = True
            if not custom_model_id or not custom_model_id.strip():
                yield (
                    "❌ Lỗi: Vui lòng nhập Model ID cho Custom Model.",
                    gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), # conv_tab
                    *slot_no_updates
                )
                return

            # Check if it is a LoRA to merge
            if "lora" in custom_model_id.lower():
                # Merging mode
                print(f"🔄 Detected LoRA in name. preparing merge with base: {custom_base_model}")
                if custom_base_model not in BACKBONE_CONFIGS:
                    yield (
                        f"❌ Lỗi: Base Model '{custom_base_model}' không hợp lệ.",
                        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), # conv_tab
                        *slot_no_updates
                    )
                    return
                
                base_config = BACKBONE_CONFIGS[custom_base_model]
                backbone_config = {
                    "repo": base_config["repo"], # Load base first
                    "supports_streaming": base_config["supports_streaming"],
                    "description": f"Custom Merged: {custom_model_id} + {custom_base_model}"
                }
                is_merged_lora = True
            else:
                # Normal custom model
                backbone_config = {
                    "repo": custom_model_id.strip(),
                    "supports_streaming": False, # Assume false for unknown
                    "description": f"Custom Model: {custom_model_id}"
                }
        else:
            backbone_config = BACKBONE_CONFIGS[backbone_choice]
            
        codec_config = CODEC_CONFIGS[codec_choice]
        use_lmdeploy = False
        
        # Override LMDeploy if custom
        if custom_loading:
             if "gguf" in backbone_config['repo'].lower() or "v2-turbo" in backbone_config['repo'].lower():
                 # GGUF must use Standard/Turbo backend
                 use_lmdeploy = False
             elif is_merged_lora:
                 # LoRA can use LMDeploy if we merge first (checked logic below) or Standard
                 use_lmdeploy = force_lmdeploy and should_use_lmdeploy(custom_base_model, device_choice)
             else:
                 # Full custom model (e.g. finetune)
                 use_lmdeploy = force_lmdeploy and should_use_lmdeploy("VieNeu-TTS (GPU)", device_choice) # Assume GPU compatible?
        # Use LMDeploy only if Force LMDeploy is set and the model is compatible
        # NOTE: For VieNeu-v2-Turbo, we handle LMDeploy inside TurboGPUVieNeuTTS class, 
        # so we set use_lmdeploy = False here to avoid generic FastVieNeuTTS loading.
        # NOTE: For custom_loading, the block above already decided use_lmdeploy correctly
        # (e.g. False for GGUF repos). Do NOT override that decision here.
        if "v2-Turbo" in backbone_choice or "v3" in backbone_choice.lower():
             # v2-Turbo có LMDeploy riêng trong class; v3 Turbo là PyTorch (engine riêng,
             # không bao giờ dùng LMDeploy) — bỏ qua FastVieNeuTTS generic.
             should_use_generic_fast = False
        elif custom_loading:
             should_use_generic_fast = False  # already handled above per repo name
        else:
             should_use_generic_fast = force_lmdeploy and should_use_lmdeploy(backbone_choice, device_choice)
             
        if should_use_generic_fast:
            use_lmdeploy = True
        
        if use_lmdeploy:
            lmdeploy_error_reason = None
            print(f"🚀 Using LMDeploy backend with optimizations")
            
            backbone_device = "cuda"
            
            if "ONNX" in codec_choice:
                codec_device = "cpu"
            else:
                try:
                    import torch
                    codec_device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    codec_device = "cpu"
            
            # Special handling for Custom LoRA + LMDeploy -> Merge & Save
            target_backbone_repo = backbone_config["repo"]
            
            if custom_loading and is_merged_lora:
                safe_name = custom_model_id.strip().replace("/", "_").replace("\\", "_").replace(":", "")
                cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "merged_models_cache", safe_name)
                target_backbone_repo = os.path.abspath(cache_dir)
                
                # Check if already merged (and voices.json exists)
                if not os.path.exists(cache_dir) or not os.path.exists(os.path.join(cache_dir, "vocab.json")):
                    print(f"🔄 Merging LoRA for LMDeploy optimization: {cache_dir}")
                    if os.path.exists(cache_dir):
                        print("   ⚠️ Detected incomplete cache, rebuilding...")
                    yield (
                         f"⏳ Đang merge và lưu model LoRA để tối ưu cho LMDeploy (thao tác này chỉ chạy một lần)...",
                         gr.update(interactive=False),
                         gr.update(interactive=False),
                         gr.update(interactive=False),
                         gr.update(interactive=False),
                         gr.update(),
                         gr.update(), gr.update(), gr.update(), gr.update(),
                         gr.update(), # conv_tab
                         *slot_no_updates
                    )
                    
                    try:
                        # Use GPU for merging if available for speed
                        # We use the Base Model specified
                        from vieneu.standard import VieNeuTTS
                        base_repo = BACKBONE_CONFIGS[custom_base_model]["repo"]
                        merge_device = "cuda" if torch.cuda.is_available() else "cpu"
                        
                        print(f"   • Loading base: {base_repo} ({merge_device})")
                        temp_tts = VieNeuTTS(
                            backbone_repo=base_repo,
                            backbone_device=merge_device, 
                            codec_repo=codec_config["repo"],
                            codec_device="cpu", # Codec unused for merging, keep on CPU
                            hf_token=custom_hf_token
                        )
                        
                        print(f"   • Loading Adapter: {custom_model_id}")
                        temp_tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)
                        
                        print(f"   • Merging...")
                        if hasattr(temp_tts.backbone, "merge_and_unload"):
                            temp_tts.backbone = temp_tts.backbone.merge_and_unload()
                        
                        print(f"   • Saving to cache: {cache_dir}")
                        temp_tts.backbone.save_pretrained(cache_dir)
                        temp_tts.tokenizer.save_pretrained(cache_dir)
                        
                        # Fix for LMDeploy: Explicitly save legacy tokenizer files (vocab.json, merges.txt)
                        # because LMDeploy/Transformers might default to slow tokenizer if fast one has issues,
                        # and save_pretrained on fast tokenizer sometimes omits legacy files.
                        try:
                            print("   • Ensuring legacy tokenizer files...")
                            from transformers import AutoTokenizer
                            slow_tokenizer = AutoTokenizer.from_pretrained(base_repo, use_fast=False)
                            slow_tokenizer.save_pretrained(cache_dir)
                        except Exception as e:
                            print(f"   ⚠️ Warning: Could not save slow tokenizer files: {e}")

                        # Save voices.json to cache directory so FastVieNeuTTS can find it
                        print(f"   • Saving voices definition...")
                        import json
                        voices_json_path = os.path.join(cache_dir, "voices.json")
                        voices_content = {
                             "meta": { "note": "Automatically generated during LoRA merge" },
                             "default_voice": temp_tts._default_voice,
                             "presets": temp_tts._preset_voices
                        }
                        with open(voices_json_path, 'w', encoding='utf-8') as f:
                             json.dump(voices_content, f, ensure_ascii=False, indent=2)

                        del temp_tts
                        cleanup_gpu_memory()
                        print("   ✅ Merge & Save successfully!")
                        
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise RuntimeError(f"Failed to merge & save LoRA for LMDeploy: {e}")

            print(f"📦 Loading optimized model...")
            print(f"   Backbone: {target_backbone_repo} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")
            print(f"   Triton: Enabled")
            
            try:
                from vieneu.fast import FastVieNeuTTS
                tts = FastVieNeuTTS(
                    backbone_repo=target_backbone_repo,
                    backbone_device=backbone_device,
                    codec_repo=codec_config["repo"],
                    codec_device=codec_device,
                    memory_util=0.3,
                    tp=1,
                    enable_prefix_caching=False,
                    enable_triton=True,
                    hf_token=custom_hf_token
                )
                using_lmdeploy = True
                
                # Legacy caching removed
                print(f"   ✅ Optimized backend initialized")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                
                error_str = str(e)
                if "$env:CUDA_PATH" in error_str:
                    lmdeploy_error_reason = "Không tìm thấy biến môi trường CUDA_PATH. Vui lòng cài đặt NVIDIA GPU Computing Toolkit."
                else:
                    lmdeploy_error_reason = f"{error_str}"
                
                yield (
                    f"⚠️ LMDeploy Init Error: {lmdeploy_error_reason}. Đang loading model với backend mặc định - tốc độ chậm hơn so với lmdeploy...",
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), # conv_tab
                    *slot_no_updates
                )
                time.sleep(1)
                use_lmdeploy = False
                using_lmdeploy = False
        
        if not use_lmdeploy:
            print(f"📦 Using original backend")

            if device_choice == "Auto":
                repo_lower = backbone_config['repo'].lower()
                is_gguf_backbone = "gguf" in repo_lower

                if is_gguf_backbone:
                    # GGUF backbones (llama-cpp-python): Metal on Mac, CUDA on Windows/Linux
                    if sys.platform == "darwin":
                        backbone_device = "gpu"  # llama-cpp-python uses Metal via n_gpu_layers
                    else:
                        try:
                            import torch
                            backbone_device = "gpu" if torch.cuda.is_available() else "cpu"
                        except ImportError:
                            backbone_device = "cpu"
                else:
                    # PyTorch backbones (Standard, Turbo GPU): use native torch device
                    try:
                        import torch
                        if sys.platform == "darwin":
                            backbone_device = "mps" if torch.backends.mps.is_available() else "cpu"
                        else:
                            backbone_device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        backbone_device = "cpu"

                # Codec device
                if "ONNX" in codec_choice:
                    codec_device = "cpu"
                else:
                    try:
                        import torch
                        if sys.platform == "darwin":
                            codec_device = "mps" if torch.backends.mps.is_available() else "cpu"
                        else:
                            codec_device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        codec_device = "cpu"

            elif device_choice == "MPS":
                backbone_device = "mps"
                codec_device = "mps" if "ONNX" not in codec_choice else "cpu"

            else:
                backbone_device = device_choice.lower()
                codec_device = device_choice.lower()

                if "ONNX" in codec_choice:
                    codec_device = "cpu"

            if "gguf" in backbone_config['repo'].lower() and backbone_device == "cuda":
                # Only Llama-cpp (GGUF) uses the 'gpu' string for CUDA
                backbone_device = "gpu"
            
            print(f"📦 Loading model...")
            print(f"   Backbone: {backbone_config['repo']} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")
            
            if "v3-Turbo" in backbone_choice:
                # VieNeu v3 Turbo. CPU → ONNX Runtime; GPU → PyTorch. The backend is
                # auto-selected from the device inside Vieneu(mode="v3turbo"); ONNX
                # graphs are fetched from the model repo's onnx/ subfolder.
                print("   🆕 Mode: v3 Turbo (CPU=ONNX / GPU=PyTorch)")
                # Map the app's device string to what the v3 engine understands.
                v3_device = "cpu" if str(backbone_device).lower() == "cpu" else "auto"
                # precision: "int8" (mặc định, subfolder onnx_int8) | "fp32" (onnx_update).
                # Chỉ ảnh hưởng đường CPU/ONNX; trên GPU dùng PyTorch nên bỏ qua.
                v3_precision = backbone_config.get("precision", "int8")
                print(f"   🎚️  Precision: {v3_precision}")
                tts = Vieneu(
                    mode="v3turbo",
                    backbone_repo=backbone_config["repo"],
                    device=v3_device,
                    precision=v3_precision,
                    hf_token=custom_hf_token,
                )
            elif "v2-Turbo" in backbone_choice:
                # VieNeu v2 Turbo uses the dedicated backend
                print("   ⚡ Mode: Turbo")
                mode = "turbo_gpu" if "GPU" in backbone_choice else "turbo"
                tts = Vieneu(
                    mode=mode,
                    backbone_repo=backbone_config["repo"],
                    decoder_repo=codec_config["repo"],
                    device=backbone_device,
                    backend="lmdeploy" if force_lmdeploy and "GPU" in backbone_choice else "standard",
                    hf_token=custom_hf_token
                )
            else:
                from vieneu.standard import VieNeuTTS
                tts = VieNeuTTS(
                    backbone_repo=backbone_config["repo"],
                    backbone_device=backbone_device,
                    codec_repo=codec_config["repo"],
                    codec_device=codec_device,
                    hf_token=custom_hf_token,
                    gguf_filename=backbone_config.get("gguf_filename")
                )

            # Perform LoRA Merge if needed (ONLY for Standard Backend)
            # For LMDeploy, we handled it above by saving to disk
            if is_merged_lora and custom_loading and not using_lmdeploy:
                yield (
                    f"🔄 Đang tải và merge LoRA adapter: {custom_model_id}...",
                    gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), # conv_tab
                    *slot_no_updates
                )
                try:
                    # 1. Load Adapter
                    tts.load_lora_adapter(custom_model_id.strip(), hf_token=custom_hf_token)
                    
                    # 2. Merge and Unload
                    # Check if backbone matches expected type for merge
                    if hasattr(tts, 'backbone') and hasattr(tts.backbone, 'merge_and_unload'):
                        print("   🔄 Merging LoRA into backbone...")
                        tts.backbone = tts.backbone.merge_and_unload()
                        
                        # Reset LoRA state so it behaves like a normal model
                        tts._lora_loaded = False 
                        tts._current_lora_repo = None
                        print("   ✅ Merged successfully!")
                    else:
                        print("   ⚠️ Warning: Model does not support merge_and_unload, keeping adapter active.")
                        
                except Exception as e:
                     raise RuntimeError(f"Failed to merge LoRA: {e}")

            using_lmdeploy = False
        
        current_backbone = backbone_choice
        current_codec = codec_choice
        model_loaded = True
        
        # Success message with optimization info
        backend_name = "🚀 LMDeploy (Optimized)" if using_lmdeploy else "📦 Standard"
        device_info = "cuda" if use_lmdeploy else (backbone_device if not use_lmdeploy else "N/A")
        
        streaming_support = "✅ Có" if backbone_config['supports_streaming'] else "❌ Không"
        preencoded_note = "\n⚠️ Codec này cần sử dụng pre-encoded codes (.pt files)" if codec_config['use_preencoded'] else ""
        
        opt_info = ""
        if using_lmdeploy and hasattr(tts, 'get_optimization_stats'):
            stats = tts.get_optimization_stats()
            opt_info = (
                f"\n\n🔧 Tối ưu hóa:"
                f"\n  • Triton: {'✅' if stats['triton_enabled'] else '❌'}"
                f"\n  • Max Batch Size (Default): {stats.get('max_batch_size', 'N/A')}"
                f"\n  • Reference Cache: {stats['cached_references']} voices"
                f"\n  • Prefix Caching: ❌"
            )
        
        warning_msg = ""
        if lmdeploy_error_reason:
             warning_msg = (
                 f"\n\n⚠️ **Cảnh báo:** Không thể kích hoạt LMDeploy (Optimized Backend) do lỗi sau:\n"
                 f"👉 {lmdeploy_error_reason}\n"
                 f"💡 Hệ thống đã tự động chuyển về chế độ Standard (chậm hơn)."
             )

        success_msg = get_model_status_message()
        if warning_msg:
            success_msg += warning_msg

        yield (success_msg, *_ready_ui_updates(backbone_choice))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        model_loaded = False
        using_lmdeploy = False

        if "$env:CUDA_PATH" in str(e):
            yield (
                "❌ Lỗi khi tải model: Không tìm thấy biến môi trường CUDA_PATH. Vui lòng cài đặt NVIDIA GPU Computing Toolkit (https://developer.nvidia.com/cuda/toolkit)",
                gr.update(interactive=False),
                gr.update(interactive=False), # btn_generate_conv
                gr.update(interactive=True), # btn_load
                gr.update(interactive=False), # btn_stop
                gr.update(), # voice_select
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), # conv_tab
                *slot_no_updates
            )
        else: 
            yield (
                f"❌ Lỗi khi tải model: {str(e)}",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), # conv_tab
                *slot_no_updates
            )


def resolve_voice_id(v_id: str) -> str:
    """Robustly resolve voice ID, handling both display labels and internal IDs."""
    if not v_id:
        return v_id
    # Chế độ ẩn giọng (link chia sẻ): không cho dùng giọng clone đã lưu kể cả khi gõ tay tên giọng.
    if _HIDE_USER_VOICES and tts is not None and getattr(tts, "is_user_voice", lambda _n: False)(str(v_id)):
        raise ValueError("Giọng này không dùng được trên link chia sẻ.")
    
    global PRESET_VOICES_CACHE
    if not PRESET_VOICES_CACHE:
        return v_id
        
    for item in PRESET_VOICES_CACHE:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            label, value = item[0], item[1]
            if v_id == value or v_id == label:
                return value
        else:
            if v_id == item:
                return item
            
    return v_id

# --- 2. DATA & HELPERS ---

# Phong cách đọc (style) đã bỏ trên v3 Turbo: style nằm sẵn trong reference
# (speaker embedding + ref codes) nên mọi lần sinh đều là giọng tự nhiên.


# Giọng mẫu dài từ ngần này giây trở lên thì lấy dấu vân giọng trung bình trên cả file.
LONG_REF_MIN_SECONDS = 12.0
_REF_EMB_CACHE: dict = {}


def _file_digest(path) -> str:
    """Mã băm nội dung file: cùng một file tải lên lại (đường dẫn/giờ sửa khác) vẫn trúng bộ nhớ đệm."""
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _probe_duration(path):
    """Thời lượng (giây) mà không giải mã cả file: soundfile cho wav/mp3/flac/ogg, ffprobe cho m4a/video."""
    try:
        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001
        pass
    import shutil
    import subprocess
    probe = shutil.which("ffprobe")
    if probe:
        try:
            out = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration",
                                  "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, timeout=30)
            return float(out.stdout.strip())
        except Exception:  # noqa: BLE001
            pass
    from vieneu.v3turbo import load_audio_mono
    wav, sr = load_audio_mono(path)
    return len(wav) / sr


def _reference_key_and_duration(path):
    """(khoá nhớ theo nội dung file, thời lượng giây) của file giọng mẫu; lỗi → (None, 0)."""
    try:
        return _file_digest(path), _probe_duration(path)
    except Exception:
        return None, 0.0


_CLIP_FILES: dict = {}


def _cut_reference_clip(path, key, start: float, end: float) -> str:
    """Cắt đoạn [start, end] giây của file giọng mẫu ra file wav tạm (nhớ lại để khỏi cắt lần nữa)."""
    ck = (key, start, end)
    p = _CLIP_FILES.get(ck)
    if p and os.path.exists(p):
        return p
    fd, p = tempfile.mkstemp(prefix="vieneu_refclip_", suffix=".wav")
    os.close(fd)
    try:
        sr = sf.info(str(path)).samplerate      # chỉ đọc đúng đoạn cần, không nạp cả file dài
        wav, sr = sf.read(str(path), start=int(start * sr), stop=int(end * sr), dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        sf.write(p, wav, sr)
    except Exception:  # noqa: BLE001 — m4a/video: để ffmpeg cắt thẳng đoạn cần
        import shutil
        import subprocess
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise
        subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}",
                        "-i", str(path), "-vn", "-ac", "1", "-ar", "48000", p], check=True, capture_output=True)
    _CLIP_FILES[ck] = p
    return p


def _remember_reference_emb(key, emb) -> None:
    if key is None or (isinstance(key, tuple) and None in key[:2]):
        return
    if len(_REF_EMB_CACHE) >= 32:
        _REF_EMB_CACHE.pop(next(iter(_REF_EMB_CACHE)))
    _REF_EMB_CACHE[key] = emb


def _save_output_audio(wav, sr: int, prefix: str = "giong-doc") -> str:
    """Lưu kết quả thành MP3 (nhẹ, dễ gửi Zalo/Messenger) với tên dễ đọc; không ghi được MP3 thì lưu WAV."""
    out_dir = tempfile.mkdtemp(prefix="vieneu_out_")
    stem = os.path.join(out_dir, f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}")
    try:
        sf.write(stem + ".mp3", wav, sr, format="MP3")
        return stem + ".mp3"
    except Exception:
        sf.write(stem + ".wav", wav, sr)
        return stem + ".wav"


def _clone_reference_steps(custom_audio, denoise_ref, long_profile=True):
    """Chuẩn bị giọng mẫu để clone — dùng chung cho nút Đọc và nút Lưu giọng (giọng lưu = giọng nghe thử).

    File dài: tìm đoạn mẫu + dấu vân giọng trung bình trên cả file; mã giọng lấy từ đoạn mẫu (cắt lặng,
    chuẩn hoá âm lượng, tuỳ chọn lọc tạp âm). Có nhớ đệm theo nội dung file. Là generator: yield
    ``(None, thông báo)`` khi phải chờ; trả về ``(speaker_emb, ref_codes, file_mẫu, ghi_chú)``."""
    note = ""
    ref_key, ref_dur = _reference_key_and_duration(custom_audio)
    enc_src, clip_info, avg_emb = custom_audio, None, None
    # File mẫu dài (video/stream tải lên): phân tích một lượt → đoạn 8 giây rõ nhất làm mã
    # giọng mẫu (thay vì luôn lấy 8 giây đầu) + dấu vân giọng trung bình trên cả file, bỏ
    # đoạn nhạc/người khác. Nhớ theo nội dung file nên chỉ chậm ở lần đầu.
    # long_profile=False (giọng cảm xúc): chỉ dùng đúng đoạn mẫu, không trộn cảm xúc của cả file dài.
    if long_profile and ref_key is not None and ref_dur >= LONG_REF_MIN_SECONDS and hasattr(tts, "reference_profile"):
        prof_key = ("prof", ref_key, current_backbone)
        prof = _REF_EMB_CACHE.get(prof_key)
        if prof is None:
            yield None, ("⏳ Đang phân tích giọng mẫu: tìm đoạn rõ nhất và lấy dấu vân giọng trên "
                         "cả file (chỉ lần đầu, khoảng 10–20 giây; các lần đọc sau sẽ nhanh)...")
            try:
                prof = tts.reference_profile(custom_audio)
                _remember_reference_emb(prof_key, prof)
            except Exception as e:  # noqa: BLE001 — lỗi thì dùng 8 giây đầu như cũ
                print(f"⚠️ Không phân tích được giọng mẫu dài: {e}")
        if prof is not None:
            avg_emb, c_start, c_end = prof
            enc_src = _cut_reference_clip(custom_audio, ref_key, c_start, c_end)
            clip_info = (c_start, c_end)
            note = f" · 🎯 Đoạn mẫu tự chọn: giây {c_start:.0f}–{c_end:.0f}"
    # Enroll: (tuỳ chọn lọc tạp âm) + cắt lặng + chuẩn hoá âm lượng → speaker emb + mã giọng.
    # Nhớ theo nội dung file + đoạn mẫu + lựa chọn lọc tạp âm để bấm đọc lại không phải làm lại.
    enc_key = None if ref_key is None else ("enc", ref_key, clip_info, bool(denoise_ref), current_backbone)
    cached = _REF_EMB_CACHE.get(enc_key) if enc_key else None
    if cached is None:
        cached = tts.encode_reference(enc_src, denoise=denoise_ref)
        _remember_reference_emb(enc_key, cached)
    # Bản sao để các bước sau (đẩy giọng miền...) không sửa vào bộ nhớ đệm.
    v3_speaker_emb, ref_codes = np.array(cached[0], copy=True), np.array(cached[1], copy=True)
    if avg_emb is not None:
        v3_speaker_emb = np.array(avg_emb, copy=True)
    return v3_speaker_emb, ref_codes, enc_src, note


SAVE_NAME_MAX = 60


def _save_voice_core(custom_audio, name, description, denoise_ref, accent, accent_strength, *,
                     allow_overwrite=False, long_profile=True):
    """Lõi lưu một giọng người dùng từ giọng mẫu (dùng chung cho 💾 Lưu giọng và ➕ Lưu cảm xúc).

    Xử lý giống hệt khi bấm Đọc (cùng _clone_reference_steps), nên giọng lưu = giọng vừa nghe thử.
    Generator: yield ``("status", thông báo)`` khi chờ/lỗi; cuối cùng yield ``("done", (thành_công, ghi_chú))``."""
    def fail(msg):
        return [("status", msg), ("done", (False, ""))]

    if not model_loaded or tts is None:
        yield from fail("⏳ Giọng đọc đang được chuẩn bị, vui lòng đợi vài giây rồi bấm lại.")
        return
    if _HIDE_USER_VOICES:
        yield from fail("❌ Chức năng lưu giọng đang tắt trên link chia sẻ này.")
        return
    if "v3" not in (current_backbone or "").lower() or not hasattr(tts, "add_voice"):
        yield from fail("❌ Chỉ lưu được giọng khi dùng mô hình VieNeu-TTS v3.")
        return
    if not custom_audio:
        yield from fail("❌ Bạn chưa ghi âm hoặc tải lên giọng mẫu.")
        return

    def taken():
        """Tên đã có → thông báo lỗi; None nếu được dùng (chưa có, hoặc cho ghi đè giọng người dùng)."""
        if name not in getattr(tts, "_preset_voices", {}):
            return None
        if tts.is_user_voice(name):
            return None if allow_overwrite else f"❌ Đã có giọng đã lưu tên **{name}**. Hãy đặt tên khác."
        return f"❌ Đã có giọng có sẵn tên **{name}**. Hãy đặt tên khác."

    if hasattr(tts, "reload_user_voices"):
        try:
            tts.reload_user_voices()   # đồng bộ với giọng vừa lưu/đổi ở nơi khác trước khi ghi
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Không nạp lại được giọng người dùng: {e}")
    err = taken()
    if err:
        yield from fail(err)
        return

    yield ("status", "⏳ Đang phân tích và lưu giọng (khoảng 10–30 giây)...")
    try:
        steps = _clone_reference_steps(custom_audio, denoise_ref, long_profile=long_profile)
        while True:
            try:
                _, msg = next(steps)
                yield ("status", msg)
            except StopIteration as stop:
                emb, _codes, src, note = stop.value
                break
        # Phân tích mất 10–30 giây: trong lúc đó tên có thể vừa bị dùng ở nơi khác → kiểm tra lại
        # ngay trước khi lưu để không ghi đè giọng của người khác.
        if hasattr(tts, "reload_user_voices"):
            tts.reload_user_voices()
        err = taken()
        if err:
            yield from fail(err.replace("Đã có", "Vừa có") + " (vừa được lưu ở nơi khác)")
            return
        if accent in ACCENTS and hasattr(tts, "steer_accent"):
            emb = tts.steer_accent(emb, accent, 0.5 if accent_strength is None else float(accent_strength))
        tts.add_voice(name, src, denoise=bool(denoise_ref), description=description,
                      speaker_emb=emb, save=True)
    except Exception as e:  # noqa: BLE001
        yield from fail(f"❌ Không lưu được giọng: {e}")
        return
    extra = f", đã chỉnh về giọng miền {ACCENTS[accent]}" if accent in ACCENTS else ""
    yield ("done", (True, extra + note))


def save_cloned_voice(custom_audio, name, desc, consent, denoise_ref, accent, accent_strength):
    """💾 Lưu giọng clone thành giọng riêng (ghi vào file giọng người dùng): lần sau chọn ở "Giọng có sẵn"
    hoặc gọi API (``voice=tên``), không cần tải mẫu lại."""
    keep = [gr.update()] * (1 + MAX_SPEAKERS)   # ô "Giọng đọc" + các ô giọng ở tab hội thoại

    def status(msg):
        return (gr.update(value=msg, visible=True), *keep)

    name = " ".join(str(name or "").split())
    if not name:
        yield status("❌ Hãy đặt tên cho giọng (vd: Giọng anh Tiến).")
        return
    if len(name) > SAVE_NAME_MAX or "⚠️" in name:
        yield status(f"❌ Tên giọng tối đa {SAVE_NAME_MAX} ký tự.")
        return
    if not consent:
        yield status("❌ Hãy tích ô xác nhận được phép dùng giọng này.")
        return
    description = (desc or "").strip() or f"Giọng clone lưu từ web ({time.strftime('%d/%m/%Y')})"
    result = (False, "")
    for kind, val in _save_voice_core(custom_audio, name, description, denoise_ref, accent, accent_strength):
        if kind == "status":
            yield status(val)
        else:
            result = val
    if not result[0]:
        return
    voice_update, slot_update = _voice_choice_updates(selected=name)
    yield (gr.update(value=(f"✅ Đã lưu giọng **{name}**{result[1]}. Giọng nằm trong mục **👤 Giọng có sẵn** "
                            f"(có dấu 🎙️): chọn rồi bấm **▶ Nghe thử** hoặc **🔊 Đọc thành tiếng**. "
                            f"App ngoài gọi API thì dùng `\"voice\": \"{name}\"`."), visible=True),
           voice_update, *[slot_update] * MAX_SPEAKERS)


# --- 🎭 Bộ giọng cảm xúc: nhiều giọng cảm xúc của cùng một người, đọc theo thẻ [cảm xúc] ---
EMOTION_GUIDE = (
    "**Đọc có cảm xúc bằng chính giọng của bạn, 2 bước:**\n"
    "1. **Thêm cảm xúc:** mỗi lần ghi âm/tải lên một đoạn **5–8 giây** bạn nói với **một** cảm xúc "
    "(vd ấm áp, vui, hỏi nhẹ), chọn tên cảm xúc rồi bấm **➕ Lưu cảm xúc này**. Làm lại cho từng cảm xúc: "
    "cùng một người, cùng micro, cùng chỗ thu.\n"
    "2. **Đọc theo cảm xúc:** chọn bộ giọng ở dưới, rồi ở bước ① viết thẻ trước mỗi câu, vd "
    "`[ấm áp] Chào bạn. [dừng 0.5s] [vui] Học thử miễn phí nhé!` rồi bấm **🔊 Đọc thành tiếng**.\n\n"
    "Ghi chú khác trong ngoặc vuông (vd `[đọc chậm, nhấn chữ này]`) được bỏ qua, không đọc to."
)
EMOTION_EMPTY_INFO = "Chưa có bộ giọng cảm xúc nào. Thêm ít nhất một cảm xúc ở mục **➕ Thêm cảm xúc** phía trên."


def _user_voice_ids() -> list:
    if tts is None:
        return []
    if hasattr(tts, "reload_user_voices"):
        try:
            tts.reload_user_voices()
        except Exception:  # noqa: BLE001
            pass
    try:
        return [v["id"] for v in tts.list_voices() if v.get("custom")]
    except Exception:  # noqa: BLE001
        return []


def _current_emotion_sets() -> dict:
    return {} if _HIDE_USER_VOICES else emotion_sets(_user_voice_ids())


def _emotion_default_and_info(sets: dict, sel):
    if not sel or sel not in sets:
        return gr.update(choices=[], value=None), EMOTION_EMPTY_INFO
    emos = list(sets[sel])
    default = next((e for e in ("tự nhiên", "ấm áp") if e in emos), emos[0])
    tags = " ".join(f"`[{e}]`" for e in emos)
    info = (f"Bộ **{sel}** có {len(emos)} cảm xúc. Thẻ dùng được: {tags}. Thêm `[dừng 0.5s]` để ngừng, "
            f"`[cười]` để cười.")
    return gr.update(choices=emos, value=default), info


def emotion_set_updates(selected=None):
    """Cập nhật: ô tên bộ (thêm cảm xúc), ô chọn bộ, ô cảm xúc mặc định, dòng mô tả bộ."""
    sets = _current_emotion_sets()
    names = list(sets)
    sel = selected if selected in sets else (names[0] if names else None)
    default_update, info = _emotion_default_and_info(sets, sel)
    return gr.update(choices=names), gr.update(choices=names, value=sel), default_update, info


def on_emotion_set_change(sel):
    return _emotion_default_and_info(_current_emotion_sets(), sel)


def emotion_example_text(sel):
    sets = _current_emotion_sets()
    if not sel or sel not in sets:
        gr.Warning("Hãy thêm cảm xúc và chọn bộ giọng trước.")
        return gr.update()
    return example_script(list(sets[sel]))


def save_emotion_voice(emo_audio, set_name, emotion, consent, overwrite, denoise_ref, accent):
    """➕ Lưu một cảm xúc cho bộ giọng: giọng người dùng tên ``"<bộ> · <cảm xúc>"``."""
    keep = [gr.update()] * 5   # ô tên bộ, ô chọn bộ, ô cảm xúc mặc định, mô tả bộ, ô "Giọng đọc"

    def status(msg):
        return (gr.update(value=msg, visible=True), *keep)

    err = validate_set_name(set_name) or validate_emotion(emotion)
    if err:
        yield status("❌ " + err)
        return
    if not consent:
        yield status("❌ Hãy tích ô xác nhận được phép dùng giọng này.")
        return
    set_name = " ".join(str(set_name).split())
    emo = norm_label(emotion)
    name = emotion_voice_name(set_name, emo)
    result = (False, "")
    for kind, val in _save_voice_core(emo_audio, name, f"Cảm xúc: {emo} (bộ {set_name})", denoise_ref,
                                      accent, 0.5, allow_overwrite=bool(overwrite), long_profile=False):
        if kind == "status":
            if "Đã có giọng đã lưu" in val:
                val += " Muốn thay đoạn mẫu cũ thì tích **Ghi đè nếu cảm xúc này đã có**."
            yield status(val)
        else:
            result = val
    if not result[0]:
        return
    _voice_choice_updates()   # làm mới danh sách giọng (giọng cảm xúc cũng hiện ở "Giọng có sẵn")
    yield (gr.update(value=(f"✅ Đã lưu cảm xúc **{emo}** cho bộ **{set_name}** (giọng `{name}`){result[1]}. "
                            f"Thêm cảm xúc khác cho cùng bộ, rồi dùng mục **Đọc theo cảm xúc** bên dưới."),
                      visible=True),
           *emotion_set_updates(selected=set_name), gr.update(choices=PRESET_VOICES_CACHE))


def _synthesize_emotion(text, set_name, default_emotion, temperature, max_chars_chunk):
    """Đọc văn bản có thẻ [cảm xúc]/[dừng Xs]: mỗi đoạn đọc bằng giọng cảm xúc tương ứng của bộ, rồi ghép."""
    if _HIDE_USER_VOICES:
        yield None, "❌ Giọng cảm xúc đang tắt trên link chia sẻ này."
        return
    if "v3" not in (current_backbone or "").lower():
        yield None, "❌ Đọc theo cảm xúc chỉ có với mô hình VieNeu-TTS v3."
        return
    sets = _current_emotion_sets()
    if not set_name or set_name not in sets:
        yield None, "⚠️ Hãy chọn bộ giọng cảm xúc (bước ②, tab 🎭 Giọng cảm xúc)."
        return
    emotions = sets[set_name]
    default_voice = emotions.get(norm_label(default_emotion or "")) or next(iter(emotions.values()))
    items, ignored = parse_emotion_script(text, emotions, default_voice)
    n = sum(1 for it in items if it[0] == "speak")
    if n == 0:
        yield None, "⚠️ Không có chữ nào để đọc (văn bản chỉ có thẻ?)."
        return
    sr = getattr(tts, "sample_rate", 48000)
    parts, i, last, t0 = [], 0, None, time.time()
    for it in items:
        if _STOP_EVENT.is_set():
            yield None, "⏹️ Đã dừng đọc."
            return
        if it[0] == "pause":
            parts.append(np.zeros(int(it[1] * sr), dtype=np.float32))
            last = "pause"
            continue
        i += 1
        emo = (split_emotion_voice_name(it[1]) or ("", ""))[1]
        # Giữ cụm "Đang xử lý đoạn i/n" để wrap_with_estimate ước tính thời gian.
        yield None, f"⏳ Đang đọc ({emo})... Đang xử lý đoạn {i}/{n}"
        try:
            wav = tts.infer(it[2], voice=it[1], temperature=temperature, max_chars=int(max_chars_chunk),
                            apply_watermark=False)
        except Exception as e:  # noqa: BLE001
            yield None, f"❌ Có lỗi khi đọc đoạn {i} ({emo}): {e}"
            return
        if last == "speak":
            parts.append(np.zeros(int(0.25 * sr), dtype=np.float32))   # nghỉ ngắn khi đổi cảm xúc
        parts.append(np.asarray(wav, dtype=np.float32))
        last = "speak"
    final = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    if hasattr(tts, "_apply_watermark"):
        try:
            final = tts._apply_watermark(final)
        except Exception:  # noqa: BLE001
            pass
    out = _save_output_audio(final, sr, prefix="giong-cam-xuc")
    skipped = ""
    if ignored:
        shown = ", ".join(f"[{x}]" for x in ignored[:4]) + (" ..." if len(ignored) > 4 else "")
        skipped = f" · Đã bỏ qua ghi chú: {shown}"
    yield out, (f"✅ **Xong!** Đã đọc {n} đoạn theo cảm xúc. Bấm ▶ ở trên để nghe, hoặc bấm "
                f"**⬇️ Tải file về máy**. (mất {time.time() - t0:.1f} giây){skipped}")


def synthesize_speech(text: str, voice_choice: str, custom_audio, custom_text: str,
                      mode_tab: str, generation_mode: str, use_batch: bool, max_batch_size_run: int,
                      temperature: float, max_chars_chunk: int,
                      denoise_ref: bool = True, session_id: str = None,
                      accent: str = "auto", accent_strength: float = 0.5,
                      emotion_set: str = None, emotion_default: str = None):
    """Synthesis with optimization support and max batch size control.

    ``accent`` ("auto" | "bac" | "nam") chỉ dùng khi clone bằng v3: "auto" để model tự
    đoán giọng miền từ audio mẫu, "bac"/"nam" đẩy speaker embedding về miền đó với mức
    ``accent_strength``.
    """
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    global _RUNNING_SESSION

    _STOP_EVENT.clear()  # Reset for new generation
    _RUNNING_SESSION = session_id  # nút Dừng chỉ dừng lượt của đúng người đang đọc
    
    if not model_loaded or tts is None:
        yield None, "⏳ Giọng đọc đang được chuẩn bị, vui lòng đợi vài giây rồi bấm lại."
        return

    if not text or text.strip() == "":
        yield None, "⚠️ Bạn chưa nhập văn bản cần đọc (bước ①)."
        return
    
    raw_text = text.strip()
    if mode_tab == "emotion_mode":   # 🎭 tab Giọng cảm xúc: đọc theo thẻ [cảm xúc]
        yield from _synthesize_emotion(raw_text, emotion_set, emotion_default, temperature, max_chars_chunk)
        return
    
    codec_config = CODEC_CONFIGS[current_codec]
    use_preencoded = codec_config['use_preencoded']
    
    
    # Setup Reference
    yield None, "⏳ Đang chuẩn bị giọng đọc..."
    
    is_v3 = "v3" in (current_backbone or "").lower()
    v3_speaker_emb = None
    clone_note = ""  # vd "· 🎯 Đoạn mẫu tự chọn: giây 19–27", hiện kèm thông báo Xong
    try:
        ref_codes = None
        ref_text_raw = ""
        v3_voice_token_id = None
        v_id = None

        if mode_tab == "preset_mode":
            if not voice_choice:
                raise ValueError("Bạn chưa chọn giọng đọc (bước ②).")
            if "⚠️" in voice_choice:
                raise ValueError("Không có giọng có sẵn, hãy dùng mục Giọng của bạn (clone).")

            # Use SDK method - handles caching and JSON internally
            v_id = resolve_voice_id(voice_choice)
            voice_data = tts.get_preset_voice(v_id)
            if is_v3:
                # v3 presets carry a speaker embedding + reference codes.
                v3_speaker_emb = voice_data['speaker_emb']
                ref_codes = voice_data.get('codes')
            else:
                ref_codes = voice_data['codes']
                ref_text_raw = voice_data.get('text', '')
                v3_voice_token_id = voice_data.get('reserved_id')

        elif mode_tab == "custom_mode":
            if custom_audio is None:
                raise ValueError("Bạn chưa ghi âm hoặc tải lên giọng mẫu (bước ②).")

            cb_lower = (current_backbone or "").lower()
            # Turbo v2 and v3 clone purely from audio → no reference transcript needed.
            needs_ref_text = "v2-turbo" not in cb_lower and "v3" not in cb_lower
            if needs_ref_text and (not custom_text or not custom_text.strip()):
                raise ValueError("Vui lòng nhập nội dung văn bản của Audio mẫu (Reference Text)!")

            ref_text_raw = custom_text.strip() if custom_text else ""
            if is_v3:
                v3_speaker_emb, ref_codes, _ref_src, clone_note = yield from _clone_reference_steps(
                    custom_audio, denoise_ref)
                # Model tự đoán giọng miền từ speaker embedding và hay đọc người Bắc thành
                # giọng Nam → người dùng chọn Bắc/Nam thì đẩy embedding về miền đó.
                if accent in ACCENTS:
                    if not hasattr(tts, "steer_accent"):
                        raise ValueError("Model đang tải không hỗ trợ chọn giọng miền, hãy để 'Tự động'.")
                    strength = 0.5 if accent_strength is None else float(accent_strength)
                    v3_speaker_emb = tts.steer_accent(v3_speaker_emb, accent, strength)
                    yield None, f"🗺️ Đẩy giọng về miền {ACCENTS[accent]} (mức {strength:.2f})..."
            else:
                ref_codes = tts.encode_reference(custom_audio)

        # Ensure numpy for inference
        if 'torch' in sys.modules:
            import torch
            if isinstance(ref_codes, torch.Tensor):
                ref_codes = ref_codes.cpu().numpy()

    except Exception as e:
        yield None, f"❌ {str(e)}" if isinstance(e, ValueError) else f"❌ Chưa dùng được giọng mẫu: {str(e)}"
        return
    
    # === STANDARD MODE ===
    if generation_mode == "Standard (Một lần)":
        # ============================ v3 TURBO BRANCH ========================
        # VieNeu-TTS v3 Turbo: split the text into chunks and run them through the
        # batched serving engine (vieneu.v3_turbo_serve) so multiple chunks share
        # each forward step (big GPU throughput win at Batch Size 32). Falls back
        # to single-utterance generation on CPU / 1 chunk / batching disabled.
        if "v3" in (current_backbone or "").lower():
            _t0 = time.time()
            yield None, "⏳ Đang đọc..."
            sr_v3 = getattr(tts, "sample_rate", 48000)
            try:
                from vieneu_utils.phonemize_text import phonemize_text_with_emotions

                # Chia chunk theo TEXT đã normalize (giống v2-gpu, không vụn), giữ
                # inline cues; phonemize TỪNG chunk khi dựng request.
                v3_chunks, v3_gaps = normalize_to_chunks_v3_with_gaps(raw_text, max_chars=max_chars_chunk)
                v3_bs = max(1, int(max_batch_size_run)) if use_batch else 1
                v3_engine_dev = getattr(getattr(tts, "engine", None), "device", None)
                v3_can_batch = (
                    v3_bs > 1 and len(v3_chunks) > 1
                    and v3_engine_dev is not None and v3_engine_dev.type == "cuda"
                )

                if v3_can_batch:
                    from vieneu.v3_turbo_serve import V3TurboBatchEngine
                    if getattr(tts, "_v3_batch_engine", None) is None:
                        tts._v3_batch_engine = V3TurboBatchEngine(tts.engine)
                    # Phonemize từng TEXT chunk (giữ inline cues) rồi gộp lô theo ĐỘ DÀI
                    # phoneme (length bucketing): các prompt cùng cỡ vào chung lô để
                    # giảm left-padding — đỡ tốn prefill và đỡ nhiễu số học do pad.
                    # Kết quả được trả về đúng vị trí gốc nên khớp v3_gaps khi join.
                    v3_phs = [phonemize_text_with_emotions(c) for c in v3_chunks]
                    v3_order = sorted(range(len(v3_chunks)), key=lambda k: len(v3_phs[k]))
                    v3_wavs = [None] * len(v3_chunks)
                    for bi, i in enumerate(range(0, len(v3_order), v3_bs)):
                        if _STOP_EVENT.is_set():
                            yield None, "⏹️ Đã dừng tạo giọng nói."
                            return
                        idxs = v3_order[i:i + v3_bs]
                        yield None, f"⚡ v3 Turbo: lô {bi + 1} ({len(idxs)} đoạn, batch size {v3_bs})..."
                        reqs = [{"phonemes": v3_phs[j], "speaker_emb": v3_speaker_emb,
                                 "ref_codes": ref_codes, "use_ref_codes": True} for j in idxs]
                        for j, w in zip(idxs, tts._v3_batch_engine.generate_batch(
                                reqs, temperature=temperature, max_new_frames=300)):
                            v3_wavs[j] = w
                    wav = join_audio_chunks(v3_wavs, sr=sr_v3, silence_ps=gaps_to_silence(v3_gaps))
                else:
                    # CPU (ONNX) hoặc GPU khi tắt batch: xử lý TUẦN TỰ từng đoạn.
                    # Gọi engine 1 lần / đoạn (giống nhánh batch nhưng tuần tự) để có
                    # ĐÚNG 1 wav / đoạn -> khớp với v3_gaps khi join, đồng thời báo cho
                    # người dùng đang xử lý đến đoạn nào + ước tính thời gian còn lại —
                    # quan trọng trên CPU vì mỗi đoạn có thể mất nhiều giây.
                    # (KHÔNG enumerate tts.infer(): infer() trả về 1 mảng đã ghép sẵn,
                    #  lặp qua nó sẽ ra từng sample numpy.float32 -> len() ném lỗi.)
                    total_v3 = len(v3_chunks)
                    v3_wavs = []
                    chunk_durations = []
                    last_t = time.time()
                    for i, chunk in enumerate(v3_chunks):
                        if _STOP_EVENT.is_set():
                            yield None, "⏹️ Đã dừng tạo giọng nói."
                            return
                        # Giữ cụm "Đang xử lý đoạn i/n": wrap_with_estimate dựa vào nó để ước tính thời gian.
                        yield None, f"⏳ Đang đọc... Đang xử lý đoạn {i + 1}/{total_v3}"
                        ph = phonemize_text_with_emotions(chunk)
                        chunk_wav = tts.engine.infer(
                            phonemes=ph, speaker_emb=v3_speaker_emb, ref_codes=ref_codes,
                            use_ref_codes=True,
                            temperature=temperature, max_new_frames=300)
                        now = time.time()
                        chunk_durations.append(now - last_t)
                        last_t = now
                        # Luôn append (kể cả rỗng) để v3_wavs khớp 1-1 với v3_chunks —
                        # lọc bớt ở đây sẽ làm lệch v3_gaps khi join.
                        v3_wavs.append(chunk_wav if chunk_wav is not None
                                       else np.zeros(0, dtype=np.float32))
                        done = i + 1
                        if done < total_v3:
                            avg = sum(chunk_durations) / len(chunk_durations)
                            eta = avg * (total_v3 - done)
                            yield None, (
                                f"⏳ Đã đọc xong {done}/{total_v3} đoạn, "
                                f"còn khoảng {_format_duration(eta)}..."
                            )
                    wav = join_audio_chunks(v3_wavs, sr=sr_v3, silence_ps=gaps_to_silence(v3_gaps))
            except Exception as e:
                yield None, f"❌ Có lỗi khi đọc: {str(e)}"
                return
            if wav is None or len(wav) == 0:
                yield None, "❌ Chưa tạo được âm thanh. Hãy thử lại, hoặc dùng đoạn văn ngắn hơn."
                return
            out_path_v3 = _save_output_audio(wav, sr_v3)
            _dt = time.time() - _t0
            yield out_path_v3, (f"✅ **Xong!** Bấm ▶ ở trên để nghe, hoặc bấm **⬇️ Tải file về máy**. "
                                f"(mất {_dt:.1f} giây){clone_note}")
            cleanup_gpu_memory()
            return
        # ========================== end v3 TURBO BRANCH ======================

        backend_name = "LMDeploy" if using_lmdeploy else "Standard"

        is_v2_turbo = "v2-Turbo" in (current_backbone or "")
        
        if is_v2_turbo:
            text_chunks = phonemize_to_chunks(raw_text, max_chars=max_chars_chunk)
        else:
            # Chia chunk SAU normalize: normalize cả văn bản trước rồi mới cắt theo
            # độ dài ĐÃ chuẩn hóa (tránh chunk phình quá max_chars khi norm mở rộng).
            text_chunks = normalize_to_chunks(raw_text, max_chars=max_chars_chunk)
            
        total_chunks = len(text_chunks)

        batch_info = " (Batch Mode)" if use_batch and using_lmdeploy and total_chunks > 1 else ""
        
        # Show batch size info
        batch_size_info = ""
        if use_batch and using_lmdeploy and hasattr(tts, 'max_batch_size'):
            batch_size_info = f" [Max batch: {tts.max_batch_size}]"
        
        yield None, f"🚀 Bắt đầu tổng hợp {backend_name}{batch_info}{batch_size_info} ({total_chunks} đoạn)..."
        
        all_wavs = []
        sr = 24000
        
        start_time = time.time()
        
        try:
            if is_v2_turbo:
                # Sequential processing with progress updates
                total_chunks = len(text_chunks)
                for i, chunk in enumerate(text_chunks):
                    if _STOP_EVENT.is_set():
                        yield None, "⏹️ Đã dừng tạo giọng nói."
                        return
                    yield None, f"⚡ Turbo v2: Đang xử lý đoạn {i+1}/{total_chunks}..."
                    
                    chunk_wav = tts.infer(
                        chunk.text, 
                        ref_codes=ref_codes, 
                        temperature=temperature,
                        max_chars=max_chars_chunk,
                        skip_normalize=True,
                        skip_phonemize=True
                    )
                    
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_wavs.append(chunk_wav)
                        # Add silence between Gradio-level chunks for Turbo
                        if i < total_chunks - 1:
                            sil_dur = get_silence_duration_v2(chunk)
                            sil_wav = np.zeros(int(sr * sil_dur), dtype=np.float32)
                            all_wavs.append(sil_wav)
            
            # Use batch processing if enabled and using LMDeploy (for v1)
            elif use_batch and using_lmdeploy and hasattr(tts, 'infer_batch') and total_chunks > 1:
                # Process in mini-batches to allow cancellation between batches
                num_batches = (total_chunks + max_batch_size_run - 1) // max_batch_size_run
                total_batch_duration = 0.0
                completed_batches = 0
                
                for i in range(0, total_chunks, max_batch_size_run):
                    if _STOP_EVENT.is_set():
                        print("🛑 Synthesis stopped during batch processing.")
                        yield None, "⏹️ Đã dừng tạo giọng nói."
                        return
                    
                    batch_idx = i // max_batch_size_run
                    estimate_info = ""
                    if completed_batches > 0:
                        average_batch_duration = total_batch_duration / completed_batches
                        estimated_total = average_batch_duration * num_batches
                        estimated_remaining = average_batch_duration * max(0, num_batches - batch_idx)
                        estimate_info = (
                            f" | Ước tính còn lại: {_format_duration(estimated_remaining)}"
                            f" / tổng: {_format_duration(estimated_total)}"
                        )
                    yield None, f"⚡ Đang xử lý batch {batch_idx+1}/{num_batches} (đoạn {i+1}-{min(i+max_batch_size_run, total_chunks)}){estimate_info}..."
                    
                    current_batch = text_chunks[i : i + max_batch_size_run]
                    batch_start_time = time.time()
                    batch_wavs = tts.infer_batch(
                        current_batch, 
                        ref_codes=ref_codes, 
                        ref_text=ref_text_raw,
                        max_batch_size=max_batch_size_run,
                        temperature=temperature,
                        skip_normalize=True
                    )
                    batch_duration = time.time() - batch_start_time
                    total_batch_duration += batch_duration
                    completed_batches += 1
                    average_batch_duration = total_batch_duration / completed_batches
                    estimated_total = average_batch_duration * num_batches
                    estimated_remaining = average_batch_duration * max(0, num_batches - completed_batches)
                    for chunk_wav in batch_wavs:
                        if chunk_wav is not None and len(chunk_wav) > 0:
                            all_wavs.append(chunk_wav)
                    yield None, (
                        f"✅ Xong batch {batch_idx+1}/{num_batches} "
                        f"(trung bình batch: {_format_duration(average_batch_duration)}, "
                        f"ước tính còn lại: {_format_duration(estimated_remaining)}, "
                        f"tổng: {_format_duration(estimated_total)})"
                    )

            else:
                # Sequential processing (PyTorch or GGUF v1)
                for i, chunk in enumerate(text_chunks):
                    if _STOP_EVENT.is_set():
                        yield None, "⏹️ Đã dừng tạo giọng nói."
                        return
                    yield None, f"⏳ Đang xử lý đoạn {i+1}/{total_chunks}..."
                    chunk_wav = tts.infer(
                        chunk, 
                        ref_codes=ref_codes, 
                        ref_text=ref_text_raw,
                        temperature=temperature,
                        max_chars=max_chars_chunk,
                        skip_normalize=True
                    )
                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_wavs.append(chunk_wav)
            
            if not all_wavs:
                yield None, "❌ Không sinh được audio nào."
                return
            
            yield None, "💾 Đang ghép file và lưu..."
            
            # Use utility function for joining with silence/crossfade
            # Default silence=0.15s to match SDK
            silence_p = 0.15 if not is_v2_turbo else 0.0 # Turbo adds silence internally
            final_wav = join_audio_chunks(all_wavs, sr=sr, silence_p=silence_p)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                output_path = tmp.name
            
            process_time = time.time() - start_time
            backend_info = f" (Backend: {'LMDeploy 🚀' if using_lmdeploy else 'Standard 📦'})"
            speed_info = f", Tốc độ: {len(final_wav)/sr/process_time:.2f}x realtime" if process_time > 0 else ""
            
            
            yield output_path, f"✅ Hoàn tất! (Thời gian: {process_time:.2f}s{speed_info}){backend_info}"
            
            # Cleanup memory
            if using_lmdeploy and hasattr(tts, 'cleanup_memory'):
                tts.cleanup_memory()
            
            cleanup_gpu_memory()
            
        except Exception as e:
            # Check for CUDA OOM specifically if torch is loaded
            if 'torch' in sys.modules:
                import torch
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    cleanup_gpu_memory()
                    yield None, (
                        f"❌ GPU hết VRAM! Hãy thử:\n"
                        f"• Giảm Max Batch Size (hiện tại: {tts.max_batch_size if hasattr(tts, 'max_batch_size') else 'N/A'})\n"
                        f"• Giảm độ dài văn bản\n\n"
                        f"Chi tiết: {str(e)}"
                    )
                    return
            
            import traceback
            traceback.print_exc()
            cleanup_gpu_memory()
            yield None, f"❌ Lỗi Standard Mode: {str(e)}"
            return
    
    # === STREAMING MODE ===
    else:
        sr = 24000
        crossfade_samples = int(sr * 0.03)
        audio_queue = queue.Queue(maxsize=100)
        PRE_BUFFER_SIZE = 3
        
        end_event = threading.Event()
        error_event = threading.Event()
        error_msg = ""
        
        is_v2_turbo = "v2-Turbo" in (current_backbone or "")
        if is_v2_turbo:
            text_chunks = phonemize_to_chunks(raw_text, max_chars=max_chars_chunk)
        else:
            # Chia chunk SAU normalize: normalize cả văn bản trước rồi mới cắt theo
            # độ dài ĐÃ chuẩn hóa (tránh chunk phình quá max_chars khi norm mở rộng).
            text_chunks = normalize_to_chunks(raw_text, max_chars=max_chars_chunk)
        
        def producer_thread():
            nonlocal error_msg
            try:
                previous_tail = None
                
                for i, chunk_text in enumerate(text_chunks):
                    if _STOP_EVENT.is_set():
                        break
                    
                    if is_v2_turbo:
                        stream_gen = tts.infer_stream(
                            chunk_text.text,
                            ref_codes=ref_codes, 
                            temperature=temperature,
                            max_chars=max_chars_chunk,
                            skip_normalize=True,
                            skip_phonemize=True,
                            emotion_tag=""
                        )
                    else:
                        stream_gen = tts.infer_stream(
                            chunk_text, 
                            ref_codes=ref_codes, 
                            ref_text=ref_text_raw,
                            temperature=temperature,
                            max_chars=max_chars_chunk,
                            skip_normalize=True,
                            emotion_tag=""
                        )
                    
                    for part_idx, audio_part in enumerate(stream_gen):
                        if _STOP_EVENT.is_set():
                            break
                        if audio_part is None or len(audio_part) == 0:
                            continue
                        
                        if previous_tail is not None and len(previous_tail) > 0:
                            overlap = min(len(previous_tail), len(audio_part), crossfade_samples)
                            if overlap > 0:
                                fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                                fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                                
                                blended = (audio_part[:overlap] * fade_in + 
                                         previous_tail[-overlap:] * fade_out)
                                
                                processed = np.concatenate([
                                    previous_tail[:-overlap] if len(previous_tail) > overlap else np.array([]),
                                    blended,
                                    audio_part[overlap:]
                                ])
                            else:
                                processed = np.concatenate([previous_tail, audio_part])
                            
                            tail_size = min(crossfade_samples, len(processed))
                            previous_tail = processed[-tail_size:].copy()
                            output_chunk = processed[:-tail_size] if len(processed) > tail_size else processed
                        else:
                            tail_size = min(crossfade_samples, len(audio_part))
                            previous_tail = audio_part[-tail_size:].copy()
                            output_chunk = audio_part[:-tail_size] if len(audio_part) > tail_size else audio_part
                        
                        if len(output_chunk) > 0:
                            audio_queue.put((sr, output_chunk))
                            
                    # Add silence between chunks for Turbo v2
                    if is_v2_turbo and i < len(text_chunks) - 1:
                        sil_dur = get_silence_duration_v2(chunk_text)
                        sil_wav = np.zeros(int(sr * sil_dur), dtype=np.float32)
                        audio_queue.put((sr, sil_wav))
                
                if previous_tail is not None and len(previous_tail) > 0:
                    audio_queue.put((sr, previous_tail))
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                error_event.set()
            finally:
                end_event.set()
                audio_queue.put(None)
        
        threading.Thread(target=producer_thread, daemon=True).start()
        
        yield (sr, np.zeros(int(sr * 0.05))), "📄 Đang buffering..."
        
        pre_buffer = []
        while len(pre_buffer) < PRE_BUFFER_SIZE:
            try:
                item = audio_queue.get(timeout=5.0)
                if item is None:
                    break
                pre_buffer.append(item)
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    return
                break
        
        full_audio_buffer = []
        backend_info = "🚀 LMDeploy" if using_lmdeploy else "📦 Standard"
        for sr, audio_data in pre_buffer:
            full_audio_buffer.append(audio_data)
            yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."
        
        while True:
            try:
                item = audio_queue.get(timeout=0.05)
                if item is None:
                    break
                sr, audio_data = item
                full_audio_buffer.append(audio_data)
                yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    break
                if end_event.is_set() and audio_queue.empty():
                    break
                continue
        
        if full_audio_buffer:
            final_wav = np.concatenate(full_audio_buffer)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                
                yield tmp.name, f"✅ Hoàn tất Streaming! ({backend_info})"
            
            # Cleanup memory
            if using_lmdeploy and hasattr(tts, 'cleanup_memory'):
                tts.cleanup_memory()
            
            cleanup_gpu_memory()

synthesize_speech_with_estimate = wrap_with_estimate(synthesize_speech)

def synthesize_conversation_with_empty_estimate(*args):
    for audio_path, status in synthesize_conversation(*args):
        yield audio_path, status, ""

# --- CANCELLATION ---
# threading.Event is a mutable object: never reassigned, always the same reference.
# All threads share the exact same object — no scoping/serialization issues.
_STOP_EVENT = threading.Event()
# Phiên đang đọc (các lượt đọc xếp hàng một lượt một lúc qua concurrency_id "tts"): nút Dừng
# của người khác không được dừng lượt này.
_RUNNING_SESSION = None

# --- 3. CONVERSATION LOGIC ---

def _synthesize_conversation_v3(lines, mapping, temperature, max_chars_chunk, silence_duration):
    """v3 Turbo conversation: batch the WHOLE conversation at batch size 32.

    Unlike the v2 path (sequential, one turn at a time), every turn is split into
    chunks and ALL chunks across ALL turns are flattened into one request list, then
    fed to the V3 batched engine in groups of 32 — regardless of which speaker each
    chunk belongs to (each request carries its own speaker ref_codes + reserved
    token). Wavs are reassembled per turn (in order) and joined with inter-turn
    silence. Yields ``(audio_path, status)`` like the v2 path. 48 kHz.
    """
    global tts
    from collections import defaultdict
    from vieneu_utils.core_utils import join_audio_chunks, gaps_to_silence
    from vieneu_utils.phonemize_text import phonemize_text_with_emotions, normalize_to_chunks_v3_with_gaps
    # NOTE: KHÔNG import vieneu.v3_turbo_serve ở đây — module đó import torch ở cấp
    # module, nên trên bản cài CPU/macOS không-torch (ONNX) sẽ lỗi "No module named
    # 'torch'". Chỉ import bên trong nhánh CUDA bên dưới (nơi thực sự cần batch engine).

    sr = getattr(tts, "sample_rate", 48000)
    t0 = time.time()

    # Resolve each speaker → (speaker_emb, ref_codes), cached per speaker.
    def _voice_for(spk_name):
        cfg = mapping.get(spk_name.lower())
        v_id = (cfg or {}).get('voice') or tts._default_voice
        try:
            vd = tts.get_preset_voice(v_id)
        except Exception:
            vd = tts.get_preset_voice(tts._default_voice)
        emb = vd.get('speaker_emb')
        rc = vd.get('codes')
        if 'torch' in sys.modules:
            import torch
            if isinstance(rc, torch.Tensor):
                rc = rc.cpu().numpy()
        return (np.asarray(emb, dtype=np.float32) if emb is not None else None,
                np.asarray(rc) if rc is not None else None)

    # CPU (ONNX) has no batched engine → run sequentially, one turn at a time.
    dev = getattr(getattr(tts, "engine", None), "device", None)
    is_cuda = dev is not None and getattr(dev, "type", None) == "cuda"
    if not is_cuda:
        all_wavs = []
        for li, line in enumerate(lines):
            if _STOP_EVENT.is_set():
                yield None, "⏹️ Đã dừng hội thoại."
                return
            cfg = mapping.get(line['speaker'].lower())
            v_id = (cfg or {}).get('voice') or tts._default_voice
            yield None, f"⏳ [{li+1}/{len(lines)}] {line['speaker']}: {line['text'][:30]}..."
            try:
                wav = tts.infer(line['text'], voice=v_id,
                                temperature=temperature, max_chars=max_chars_chunk)
            except Exception as e:
                print(f"❌ Lỗi câu {li+1}: {e}")
                continue
            if wav is not None and len(wav):
                all_wavs.append(wav)
                if li < len(lines) - 1 and silence_duration > 0:
                    all_wavs.append(np.zeros(int(sr * silence_duration), dtype=np.float32))
        if not all_wavs:
            yield None, "❌ Không thể tạo được âm thanh nào!"
            return
        yield None, "🪄 Đang ghép nối âm thanh..."
        final_wav = np.concatenate(all_wavs)
        yield (_save_output_audio(final_wav, sr, prefix="hoi-thoai"),
               f"✅ **Xong!** Đã đọc {len(lines)} câu hội thoại. Bấm ▶ để nghe hoặc **⬇️ Tải file về máy**. "
               f"(mất {time.time() - t0:.1f} giây)")
        cleanup_gpu_memory()
        return

    voice_cache = {}
    reqs, req_line = [], []
    line_gaps = {}
    for li, line in enumerate(lines):
        key = line['speaker'].lower()
        if key not in voice_cache:
            voice_cache[key] = _voice_for(line['speaker'])
        spk_emb, ref_codes = voice_cache[key]
        # Chia chunk theo TEXT đã normalize (giống v2-gpu), giữ inline cues, rồi
        # phonemize từng chunk. Hội thoại luôn dùng phong cách Tự nhiên.
        line_chunks, line_gaps[li] = normalize_to_chunks_v3_with_gaps(line['text'], max_chars=max_chars_chunk)
        for chunk in line_chunks:
            reqs.append({"phonemes": phonemize_text_with_emotions(chunk),
                         "speaker_emb": spk_emb, "ref_codes": ref_codes,
                         "use_ref_codes": True})
            req_line.append(li)

    if not reqs:
        yield None, "❌ Không có lời thoại để tổng hợp."
        return

    if getattr(tts, "_v3_batch_engine", None) is None:
        # Chỉ tới đây khi chạy trên CUDA → torch chắc chắn có sẵn.
        from vieneu.v3_turbo_serve import V3TurboBatchEngine
        tts._v3_batch_engine = V3TurboBatchEngine(tts.engine)

    BS = 32
    total_batches = (len(reqs) + BS - 1) // BS
    # Length bucketing: gộp lô theo độ dài phoneme để giảm left-padding; kết quả
    # đặt về đúng vị trí gốc nên vẫn khớp req_line khi gom theo lời thoại.
    req_order = sorted(range(len(reqs)), key=lambda k: len(reqs[k]["phonemes"]))
    wavs_flat = [None] * len(reqs)
    for bi, i in enumerate(range(0, len(req_order), BS)):
        if _STOP_EVENT.is_set():
            yield None, "⏹️ Đã dừng hội thoại."
            return
        idxs = req_order[i:i + BS]
        yield None, f"⚡ v3 Turbo hội thoại: lô {bi + 1}/{total_batches} ({len(idxs)} đoạn, batch 32)..."
        for j, w in zip(idxs, tts._v3_batch_engine.generate_batch(
                [reqs[k] for k in idxs], temperature=temperature, max_new_frames=300)):
            wavs_flat[j] = w

    # Reassemble per turn (in order), then join turns with inter-turn silence.
    by_line = defaultdict(list)
    for w, li in zip(wavs_flat, req_line):
        by_line[li].append(w)

    all_wavs = []
    for li in range(len(lines)):
        lw = join_audio_chunks(by_line[li], sr=sr, silence_ps=gaps_to_silence(line_gaps.get(li, []))) if by_line[li] else None
        if lw is None or len(lw) == 0:
            continue
        all_wavs.append(lw)
        if li < len(lines) - 1 and silence_duration > 0:
            all_wavs.append(np.zeros(int(sr * silence_duration), dtype=np.float32))

    if not all_wavs:
        yield None, "❌ Không thể tạo được âm thanh nào!"
        return

    yield None, "🪄 Đang ghép nối âm thanh..."
    final_wav = np.concatenate(all_wavs)
    elapsed = time.time() - t0
    yield (_save_output_audio(final_wav, sr, prefix="hoi-thoai"),
           f"✅ **Xong!** Đã đọc {len(lines)} câu hội thoại. Bấm ▶ để nghe hoặc **⬇️ Tải file về máy**. "
           f"(mất {elapsed:.1f} giây)")
    cleanup_gpu_memory()


def synthesize_conversation(
    script_text: str,
    *args
):
    """
    Synthesizes multi-speaker conversation from a script.

    Gradio passes speaker name boxes and voice dropdowns as individual positional args.
    Layout: args[0..MAX_SPEAKERS-1] = speaker names, args[MAX_SPEAKERS..2*MAX_SPEAKERS-1] = voice IDs,
    args[2*MAX_SPEAKERS] = silence_duration, args[2*MAX_SPEAKERS+1] = temperature,
    args[2*MAX_SPEAKERS+2] = max_chars_chunk, args[2*MAX_SPEAKERS+3] = session_id
    """
    speaker_names     = list(args[:MAX_SPEAKERS])
    speaker_voices    = list(args[MAX_SPEAKERS:MAX_SPEAKERS*2])
    silence_duration  = args[MAX_SPEAKERS * 2]
    temperature       = args[MAX_SPEAKERS * 2 + 1]
    max_chars_chunk   = args[MAX_SPEAKERS * 2 + 2]
    session_id        = args[MAX_SPEAKERS * 2 + 3] if len(args) > MAX_SPEAKERS * 2 + 3 else None

    global tts, model_loaded, using_lmdeploy, _RUNNING_SESSION

    _STOP_EVENT.clear()
    _RUNNING_SESSION = session_id

    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return
        
    if not script_text or script_text.strip() == "":
        yield None, "⚠️ Vui lòng nhập kịch bản hội thoại!"
        return

    # 1. Parse Script
    lines = []
    for line in script_text.strip().split('\n'):
        if not line.strip(): continue
        if ':' in line:
            parts = line.split(':', 1)
            lines.append({'speaker': parts[0].strip(), 'text': parts[1].strip()})
        else:
            if lines:
                lines[-1]['text'] += " " + line.strip()
            else:
                lines.append({'speaker': 'Narrator', 'text': line.strip()})

    if not lines:
        yield None, "⚠️ Không tìm thấy lời thoại hợp lệ (định dạng Nhân vật: Lời thoại)!"
        return

    # 2. Build Speaker Mapping from individual slot components
    mapping = {}
    for name, voice in zip(speaker_names, speaker_voices):
        name = str(name).strip() if name else ""
        if not name: continue
        # Use lowercase key for robust matching
        v_id = resolve_voice_id(str(voice)) if voice else ""
        mapping[name.lower()] = {
            'type': 'Preset',
            'voice': v_id,
            'ref_text': ''
        }

    # 2b. v3 Turbo: batch toàn bộ hội thoại (bs=32), bất kể speaker thay đổi.
    if "v3" in (current_backbone or "").lower():
        yield from _synthesize_conversation_v3(
            lines, mapping, temperature, max_chars_chunk, silence_duration
        )
        return

    # 3. Process Each Line
    all_wavs = []
    sr = 24000
    total_lines = len(lines)
    
    yield None, f"🎭 Đang khởi tạo hội thoại ({total_lines} câu)..."
    
    start_time = time.time()
    
    try:
        for i, line in enumerate(lines):
            if _STOP_EVENT.is_set():
                yield None, "⏹️ Đã dừng hội thoại."
                return
            spk_name = line['speaker']
            text = line['text']
            
            yield None, f"⏳ [{i+1}/{total_lines}] {spk_name}: {text[:30]}..."
            
            # Determine voice
            ref_codes = None
            ref_text_val = None
            current_voice_obj = None
            
            # Case-insensitive lookup
            config = mapping.get(spk_name.lower())
            
            if not config:
                print(f"  ⚠️ Character '{spk_name}' not found in mapping. Fallback to default.")
                # Fallback to default if speaker not mapped
                try:
                    # Get default voice data
                    default_v_id = tts._default_voice
                    if not default_v_id:
                        dv_list = tts.list_preset_voices()
                        if dv_list:
                            first = dv_list[0]
                            default_v_id = first[1] if isinstance(first, tuple) else first
                    
                    if default_v_id:
                        current_voice_obj = tts.get_preset_voice(default_v_id)
                        ref_codes = current_voice_obj['codes']
                        ref_text_val = current_voice_obj['text']
                except Exception as e:
                    print(f"  ❌ Fallback failed: {e}")
            else:
                try:
                    v_id = config['voice']
                    if config['type'] == "Preset":
                        current_voice_obj = tts.get_preset_voice(v_id)
                        if current_voice_obj and 'codes' in current_voice_obj:
                            ref_codes = current_voice_obj['codes']
                            ref_text_val = current_voice_obj['text']
                        else:
                            print(f"  ❌ Could not find codes for voice '{v_id}'")
                    else: # Custom
                        if v_id and os.path.exists(v_id):
                            ref_codes = tts.encode_reference(v_id)
                            ref_text_val = config.get('ref_text', '')
                            current_voice_obj = {'codes': ref_codes, 'text': ref_text_val}
                            print(f"  🦜 Using custom voice for '{spk_name}'")
                except Exception as e:
                    print(f"  ❌ Lỗi nạp giọng cho {spk_name} (ID: {config.get('voice')}): {e}")
            
            # Ensure numpy for inference
            if 'torch' in sys.modules:
                import torch
                if isinstance(ref_codes, torch.Tensor):
                    ref_codes = ref_codes.cpu().numpy()

            # Infer audio
            try:
                wav = tts.infer(
                    text,
                    voice=current_voice_obj, # Use full voice object
                    ref_codes=ref_codes,     # Fallback if object not supported
                    ref_text=ref_text_val,
                    temperature=temperature,
                    max_chars=max_chars_chunk,
                    emotion_tag="<|emotion_0|>" # Emotion tag for conversation
                )
                
                all_wavs.append(wav)
                
                # Add silence between turns
                if i < total_lines - 1 and silence_duration > 0:
                    silence_len = int(sr * silence_duration)
                    silence = np.zeros(silence_len)
                    all_wavs.append(silence)
                    
            except Exception as e:
                print(f"❌ Lỗi tổng hợp câu {i+1}: {e}")
                continue

        if not all_wavs:
            yield None, "❌ Không thể tạo được âm thanh nào!"
            return

        # 4. Merge and Output
        yield None, "🪄 Đang ghép nối âm thanh..."
        final_wav = np.concatenate(all_wavs)
        
        elapsed = time.time() - start_time
        yield (_save_output_audio(final_wav, sr, prefix="hoi-thoai"),
               f"✅ **Xong!** Đã đọc {total_lines} câu hội thoại. Bấm ▶ để nghe hoặc **⬇️ Tải file về máy**. "
               f"(mất {elapsed:.1f} giây)")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, f"❌ Lỗi hệ thống: {str(e)}"

def extract_speakers_from_script(script):
    """Find unique speakers and return gr.update() lists for the 8 slot components."""
    global CONV_VOICES_CACHE
    if not script:
        # Hide all slots
        name_updates = [gr.update(value="", visible=False)] * MAX_SPEAKERS
        dd_updates   = [gr.update(value=None, visible=False)] * MAX_SPEAKERS
        row_updates  = [gr.update(visible=False)] * MAX_SPEAKERS
        return name_updates + dd_updates + row_updates

    speakers = []
    seen = set()
    for line in script.strip().split('\n'):
        if ':' in line:
            s = line.split(':', 1)[0].strip()
            if s and s not in seen:
                seen.add(s)
                speakers.append(s)

    used_voices = set()

    # Auto-match each speaker name to a preset voice
    def _best_match(name):
        if not CONV_VOICES_CACHE:
            return None
        
        name_l = name.lower()
        
        # 0. Manual overrides for specific common names
        overrides = {
            "phương": "Trúc Ly",
            "dũng": "Thanh Bình",
            "hùng": "Thái Sơn",
            "lan": "Trúc Ly",
            "minh": "Minh Đức",
        }
        if name_l in overrides:
            target = overrides[name_l].lower()
            for v in CONV_VOICES_CACHE:
                label, value = (v[0], v[1]) if isinstance(v, tuple) else (v, v)
                if target in label.lower() or target in value.lower():
                    return value

        # 1. Try to find name in labels or values
        for v in CONV_VOICES_CACHE:
            label, value = (v[0], v[1]) if isinstance(v, tuple) else (v, v)
            if name_l == label.lower() or name_l == value.lower():
                return value
        
        # 2. Fuzzy match (contains)
        for v in CONV_VOICES_CACHE:
            label, value = (v[0], v[1]) if isinstance(v, tuple) else (v, v)
            if name_l in label.lower() or name_l in value.lower() or label.lower() in name_l or value.lower() in name_l:
                return value
        
        # 3. Không khớp tên: lấy giọng chưa ai dùng để mỗi người một giọng khác nhau.
        for v in CONV_VOICES_CACHE:
            value = v[1] if isinstance(v, tuple) else v
            if value not in used_voices:
                return value
        first_voice = CONV_VOICES_CACHE[0]
        return first_voice[1] if isinstance(first_voice, tuple) else first_voice

    name_updates, dd_updates, row_updates = [], [], []
    for i in range(MAX_SPEAKERS):
        if i < len(speakers):
            name_updates.append(gr.update(value=speakers[i], visible=True))
            _voice = _best_match(speakers[i])
            used_voices.add(_voice)
            dd_updates.append(gr.update(value=_voice, choices=CONV_VOICES_CACHE, visible=True))
            row_updates.append(gr.update(visible=True))
        else:
            name_updates.append(gr.update(value="", visible=False))
            dd_updates.append(gr.update(value=None, choices=CONV_VOICES_CACHE, visible=False))
            row_updates.append(gr.update(visible=False))

    return name_updates + dd_updates + row_updates

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using PyMuPDF (fitz)."""
    global HAS_FITZ, fitz
    if not HAS_FITZ:
        return "⚠️ PyMuPDF chưa được cài đặt. Vui lòng chạy: pip install PyMuPDF"
    try:
        doc = fitz.open(pdf_path)
        # Ghép theo khối văn bản: mỗi dòng in trong PDF kết thúc bằng xuống dòng, nếu giữ nguyên
        # thì bộ chia đoạn coi mỗi dòng là một đoạn → đọc lên bị ngắt vụn. Bỏ số trang đứng riêng.
        paras = []
        for page in doc:
            for block in page.get_text("blocks", sort=True):
                if len(block) > 6 and block[6] != 0:  # khối ảnh
                    continue
                t = " ".join(str(block[4]).split())
                if t and not t.isdigit():
                    paras.append(t)
        doc.close()
        if not paras:
            return "⚠️ Không tìm thấy chữ trong file PDF (có thể là file ảnh scan)."
        return "\n\n".join(paras)
    except Exception as e:
        return f"⚠️ Không đọc được file PDF: {str(e)}"

EXAMPLES_LIST = [
    ["Về miền Tây không chỉ để ngắm nhìn sông nước hữu tình, mà còn để cảm nhận tấm chân tình của người dân nơi đây.", "Vĩnh (nam miền Nam)"],
    ["Hà Nội những ngày vào thu mang một vẻ đẹp trầm mặc và cổ kính đến lạ thường.", "Bình (nam miền Bắc)"],
]

with gr.Blocks(theme=theme, css=css, title="VieNeu-TTS", head=head_html) as demo:
    # Session ID for cancellation tracking
    session_id_state = gr.State("")

    with gr.Column(elem_classes="container"):
        gr.HTML("""
<div class="header-box">
    <h1 class="header-title">
        <span class="header-icon">🦜</span>
        <span class="gradient-text">VieNeu-TTS Studio</span>
    </h1>
    <p class="header-subtitle">Biến chữ thành giọng nói tiếng Việt: <b>nhập văn bản</b> → <b>chọn giọng</b> → <b>bấm Đọc</b>.
    Có thể dùng cả <b>giọng của chính bạn</b>.</p>
    <div class="model-card-content">
        <div class="model-card-item">
            <strong>Phát triển bởi:</strong>
            <span>Trần Trọng Thanh Tùng &amp; Hữu Tiến</span>
        </div>
        <div class="model-card-item">
            <strong>Dựa trên:</strong>
            <a href="https://github.com/pnnbao97/VieNeu-TTS" target="_blank" class="model-card-link">VieNeu-TTS của Phạm Nguyễn Ngọc Bảo</a>
        </div>
    </div>
</div>
        """)
        
        model_status = gr.Markdown(
            PRELOADING_MSG if _PRELOAD_ENABLED else NOT_LOADED_MSG, elem_classes="status-bar")

        # Phần chọn mô hình/thiết bị chỉ dành cho người rành kỹ thuật: model mặc định được tải sẵn.
        with gr.Accordion("🛠️ Cài đặt mô hình (dành cho người rành kỹ thuật)", open=False):
            # --- CONFIGURATION ---
            with gr.Group():
                with gr.Row():
                    # --- BACKBONE & CODEC DEFAULT LOGIC ---
                    # GPU users default to VieNeu-TTS-v3-Turbo (GPU); CPU-only users get v3 Turbo
                    # (the only CPU backbone). v3 (GPU) is registered solely when HAS_GPU.
                    # int8 là entry đầu tiên → mặc định trên cả CPU lẫn GPU (trên GPU dùng
                    # PyTorch nên int8/fp32 như nhau; trên CPU int8 nhanh nhất).
                    default_backbone = list(BACKBONE_CONFIGS.keys())[0]
                
                    # Default parameters based on backbone
                    if "v3" in default_backbone.lower():
                        default_codec = "VieNeu-Codec"
                        default_temp = 0.8
                        default_text = DEFAULT_TEXT_V3
                    elif "Turbo" in default_backbone:
                        default_codec = "VieNeu-Codec"
                        default_temp = 0.4
                        default_text = DEFAULT_TEXT_TURBO
                    elif "(CPU)" in default_backbone:
                        default_codec = "NeuCodec (ONNX)"
                        default_temp = 0.7
                        default_text = DEFAULT_TEXT_GPU
                    else:
                        default_codec = "NeuCodec (Distill)" if "NeuCodec (Distill)" in CODEC_CONFIGS else list(CODEC_CONFIGS.keys())[0]
                        default_temp = 0.7
                        default_text = DEFAULT_TEXT_GPU

                    # v3 Turbo batches chunks through the serving engine → default 32.
                    # Must be set at creation: v3 is the default backbone, so the
                    # on_backbone_change handler (which also sets 32) never fires on load.
                    default_batch_size = 32 if "v3" in default_backbone.lower() else 4

                    backbone_select = gr.Dropdown(
                        backbone_choices(list(BACKBONE_CONFIGS.keys()) + ["Custom Model"]),
                        value=default_backbone,
                        label="🦜 Mô hình giọng đọc",
                        info="Bản Nhanh chạy tốt trên máy thường. Bản Chất lượng cao nhất chậm hơn.",
                    )
                    # Codec (bộ giải mã biến kết quả của model thành âm thanh) luôn được chọn tự
                    # động theo mô hình và không sửa được → ẩn để khách không bị rối.
                    codec_select = gr.Dropdown(
                        list(CODEC_CONFIGS.keys()),
                        value=default_codec,
                        label="🎵 Codec",
                        interactive=False,
                        visible=False,
                    )
                    device_choice = gr.Radio(
                        device_choices(get_available_devices()), value="Auto",
                        label="💻 Thiết bị của bạn",
                        info="Tự động: dùng card đồ hoạ nếu máy có, không thì dùng CPU.",
                    )
            
                with gr.Row(visible=False) as custom_model_group:
                    custom_backbone_model_id = gr.Textbox(
                        label="📦 Custom Model ID",
                        placeholder="pnnbao-ump/VieNeu-TTS-0.3B-lora-ngoc-huyen",
                        info="Nhập HuggingFace Repo ID hoặc đường dẫn local",
                        scale=2
                    )
                    custom_backbone_hf_token = gr.Textbox(
                        label="🔑 HF Token (nếu private)",
                        placeholder="Để trống nếu repo public",
                        type="password",
                        info="Token để truy cập repo private",
                        scale=1
                    )
                    base_model_choices = [k for k in BACKBONE_CONFIGS.keys() if "turbo" not in k.lower() and k != "Custom Model"]
                    custom_backbone_base_model = gr.Dropdown(
                        base_model_choices,
                        label="🔗 Base Model (cho LoRA)",
                        value=base_model_choices[0] if base_model_choices else None,
                        visible=False,
                        info="Model gốc để merge với LoRA (GPU Only)",
                        scale=1
                    )
            
                with gr.Row():
                    use_lmdeploy_cb = gr.Checkbox(
                        value=True,
                        label="🚀 Optimize with LMDeploy (Khuyên dùng cho NVIDIA GPU)",
                        info="Tick nếu bạn dùng GPU để tăng tốc độ tổng hợp đáng kể.",
                        visible="v3" not in default_backbone.lower(),  # v3 Turbo (PyTorch) không dùng LMDeploy
                    )
            
            
                gr.Markdown("""
                💡 **Sử dụng Custom Model:** Chọn "Custom Model" để tải LoRA adapter hoặc bất kỳ model nào được finetune từ **VieNeu-TTS** hoặc **VieNeu-TTS-0.3B**.
                """)
            
                gr.HTML("""
                <div class="warning-banner">
                    <div class="warning-banner-title">
                        🦜 Gợi ý tối ưu hiệu năng
                    </div>
                    <div class="warning-banner-grid">
                        <div class="warning-banner-item">
                            <strong>🐆 Hệ máy GPU</strong>
                            <div class="warning-banner-content">
                                <b>VieNeu-TTS-v3-Turbo (early access)</b> đã được phát hành để dùng thử trước, đã hỗ trợ các tag cảm xúc `[cười]` `[hắng giọng]` `[thở dài]`, tuy nhiên những tính năng này vẫn đang được thử nghiệm và chưa thực sự ổn định, có thể sẽ xảy ra lỗi không mong muốn, nếu có lỗi các bạn hãy thông báo với chúng tôi tại: https://discord.com/invite/yJt8kzjzWZ. Trong trường hợp bạn cần sự ổn định hãy sử dụng <b>VieNeu-TTS-v2 (GPU)</b>. 
                            </div>
                        </div>
                        <div class="warning-banner-item" style="background: #dcfce7; border-color: #86efac;">
                            <strong style="color: #15803d;">🐢 Hệ máy CPU</strong>
                            <div class="warning-banner-content" style="color: #166534;">
                                Máy <b>CPU</b> nên dùng bản <b>VieNeu-TTS-v3-Turbo (int8)</b> để tốc độ tối đa. Chuyển sang <b>VieNeu-TTS-v3-Turbo</b> nếu cần chất lượng cao hơn (nhưng chậm hơn trên CPU).
                            </div>
                        </div>
                    </div>
                    <div style="margin-top: 12px; font-size: 0.85rem; color: #92400e; border-top: 1px dashed #fcd34d; padding-top: 8px;">
                        💡 <b>Mẹo:</b> Nếu máy bạn có GPU mà không thấy các phiên bản GPU hãy xem lại cách cài đặt uv sync --group gpu
                    </div>
                </div>
                """)

                gr.Markdown(
                    "🆕 **VieNeu-TTS-v3-Turbo (early access)** đã được phát hành để **dùng thử trước** — "
                    "48kHz, **hỗ trợ Voice Cloning** (tính năng clone chỉ có từ **v3** trở lên; v1/v2 không hỗ trợ). "
                    "Bản **v3 đầy đủ** sẽ ra mắt trong **vài tuần tới**.\n\n"
                    "🎭 v3 Turbo đã **hỗ trợ các tag cảm xúc** `[cười]` `[hắng giọng]` `[thở dài]` "
                    "(chèn trực tiếp vào văn bản) — nhưng tính năng này **đang thử nghiệm và chưa thực sự ổn định**."
                )

                btn_load = gr.Button("🔄 Tải Model", variant="primary")
        
        # Một cột cho dễ theo dõi (cả trên điện thoại): ① văn bản → ② giọng → ③ bấm đọc → kết quả.
        with gr.Column(elem_classes="container main-flow"):
            # --- INPUT ---
            with gr.Column(scale=3):
                with gr.Tabs() as main_input_tabs:
                    # --- TAB 1: SINGLE SPEAKER ---
                    with gr.Tab("📖 Đọc văn bản", id="single_tab") as single_tab:
                        gr.Markdown("### ① Nhập văn bản cần đọc", elem_classes="step-title")
                        text_input = gr.Textbox(
                            label="Văn bản cần đọc",
                            show_label=False,
                            lines=7,
                            value=default_text,
                            placeholder="Chạm vào đây để gõ, hoặc giữ ngón tay rồi chọn «Dán» để dán đoạn văn.",
                        )
                        with gr.Row(equal_height=True):
                            btn_clear_text = gr.ClearButton([text_input], value="🗑️ Xoá chữ", size="sm",
                                                            scale=0, min_width=110)
                            # Lấy chữ từ PDF: chọn file xong chữ tự điền vào ô ở trên.
                            # Ẩn hẳn nếu máy chưa cài PyMuPDF (pip install vieneu[pdf]).
                            pdf_upload = gr.UploadButton(
                                "📄 Lấy chữ từ file PDF",
                                file_types=[".pdf"],
                                file_count="single",
                                type="filepath",
                                variant="secondary",
                                size="sm",
                                scale=0,
                                min_width=190,
                                visible=HAS_FITZ,
                            )
                            length_info = gr.Markdown(_length_info(default_text), elem_classes="hint")
                        pdf_status = gr.Markdown(visible=False)
                        gr.Markdown(
                            "💡 Văn bản dài cũng được, app tự chia thành từng đoạn. Muốn thêm cảm xúc thì chèn "
                            "`[cười]` hoặc `[thở dài]` vào câu (đang thử nghiệm).",
                            elem_classes="hint",
                        )

                        gr.Markdown("### ② Chọn giọng đọc", elem_classes="step-title")
                        with gr.Tabs() as tabs:
                            with gr.TabItem("👤 Giọng có sẵn", id="preset_mode") as tab_preset:
                                with gr.Row(equal_height=True):
                                    # min_width lớn để trên điện thoại nút Nghe thử tự xuống dòng,
                                    # tên giọng không bị cắt.
                                    voice_select = gr.Dropdown(choices=[], value=None, label="Giọng đọc",
                                                               allow_custom_value=True, scale=3, min_width=300)
                                    btn_preview = gr.Button("▶ Nghe thử giọng này", variant="secondary",
                                                            scale=1, min_width=170)
                                preview_audio = gr.Audio(label="Nghe thử", type="filepath", interactive=False,
                                                         autoplay=True, visible=False)

                            # Voice cloning is only available on v3+ models. Hidden by
                            # default and toggled on by on_backbone_change when a v3
                            # model is selected.
                            with gr.TabItem("🎙️ Giọng của bạn (clone)", id="custom_mode", visible=False) as tab_custom:
                                # Initial clone-tab state must match the DEFAULT backbone:
                                # on_backbone_change only fires when the dropdown changes, so a
                                # v2-GPU default would otherwise keep v3's "no transcript" copy
                                # and a hidden reference-text box (-> false "missing ref text").
                                _default_is_v2_gpu = (default_backbone == "VieNeu-TTS-v2 (GPU)")
                                _default_is_v3 = "v3" in (default_backbone or "").lower()
                                clone_info_md = gr.Markdown(
                                    "ℹ️ **Voice Cloning (VieNeu-TTS v2).** Tải lên audio mẫu 3–5 giây "
                                    "và **nhập đúng nội dung** của audio đó (kể cả dấu câu) — v2 cần "
                                    "reference transcript để clone giọng."
                                    if _default_is_v2_gpu else CLONE_GUIDE_V3
                                )
                                with gr.Group(visible=True) as cloning_elements_group:
                                    # Ghi âm là cách dễ nhất (nhất là trên điện thoại) → mở sẵn chế độ ghi âm.
                                    custom_audio = gr.Audio(
                                        label="🎙️ Giọng mẫu của bạn (nói 5–8 giây)",
                                        type="filepath",
                                        sources=["microphone", "upload"],
                                    )
                                    # Icon ⬆ của khung audio rất nhỏ → thêm nút tải file to, rõ ràng
                                    # (máy tính không có micro thì dùng nút này).
                                    clone_upload_btn = gr.UploadButton(
                                        "📁 Tải file ghi âm / video lên (wav, mp3, m4a, mp4)",
                                        file_types=["audio", "video"],
                                        file_count="single",
                                        type="filepath",
                                        variant="secondary",
                                    )
                                    cloning_warning_msg = gr.Markdown(visible=False, elem_id="cloning-warning")
                                    gr.Markdown("🔒 Chỉ dùng giọng của chính bạn hoặc của người đã đồng ý.",
                                                elem_classes="hint")
                                    # Giọng miền (chỉ v3): model tự đoán Bắc/Nam từ giọng mẫu và hay
                                    # đọc người Bắc thành giọng Nam → cho người dùng chọn để đẩy về miền đó.
                                    with gr.Row(visible=_default_is_v3) as accent_row:
                                        accent_select = gr.Dropdown(
                                            choices=[("Để máy tự đoán", "auto"), ("Miền Bắc", "bac"), ("Miền Nam", "nam")],
                                            value="auto",
                                            label="🗺️ Bạn nói giọng miền nào?",
                                            info="Chọn đúng miền để giọng tạo ra giống bạn hơn (máy hay đoán người Bắc thành giọng Nam).",
                                            scale=2,
                                        )
                                        accent_strength = gr.Slider(
                                            minimum=0.2, maximum=0.8, value=0.5, step=0.05,
                                            label="Mức đẩy giọng miền",
                                            info="0.5 là vừa. Tăng lên nếu vẫn chưa đúng miền.",
                                            visible=False,
                                            scale=1,
                                        )
                                    # Mặc định TẮT: đo trên nhiều giọng, lọc tạp âm làm giọng clone kém
                                    # giống hơn (vd Mixi 0,63 → 0,68 khi tắt) trừ khi bản ghi rất ồn.
                                    denoise_checkbox = gr.Checkbox(
                                        value=False, label="🔇 Lọc tạp âm cho giọng mẫu",
                                        info="Để tắt thì giọng giống hơn. Chỉ bật khi bản ghi có nhiều tiếng ồn (quạt, xe, đám đông).",
                                    )
                                    # 💾 Lưu thành giọng riêng (lưu bền): chọn lại ở "Giọng có sẵn", gọi qua API.
                                    with gr.Column(variant="panel", visible=_default_is_v3 and not _HIDE_USER_VOICES) as save_voice_panel:
                                        gr.Markdown("#### 💾 Lưu thành giọng riêng để dùng lại")
                                        gr.Markdown(
                                            "Lưu xong, giọng nằm trong mục **👤 Giọng có sẵn** (có dấu 🎙️): lần sau "
                                            "chọn là đọc được ngay, không cần tải mẫu lại. App ngoài gọi API cũng dùng được.",
                                            elem_classes="hint",
                                        )
                                        with gr.Row():
                                            save_voice_name = gr.Textbox(label="Tên giọng", placeholder="vd: Giọng anh Tiến",
                                                                         max_length=SAVE_NAME_MAX, scale=2)
                                            save_voice_desc = gr.Textbox(label="Mô tả (không bắt buộc)",
                                                                         placeholder="vd: nam, miền Bắc, đọc tin tức",
                                                                         max_length=120, scale=3)
                                        save_voice_consent = gr.Checkbox(
                                            value=False,
                                            label="Tôi xác nhận đây là giọng của tôi, hoặc người có giọng này đã đồng ý cho dùng.",
                                        )
                                        btn_save_voice = gr.Button("💾 Lưu giọng", variant="primary")
                                        save_voice_status = gr.Markdown(visible=False)
                                    # v3 clones from audio only — the reference transcript box
                                    # is hidden for v3 (toggled by on_backbone_change).
                                    custom_text = gr.Textbox(label="Nội dung audio mẫu - vui lòng gõ đúng nội dung của audio mẫu - kể cả dấu câu vì model rất nhạy cảm với dấu câu (.,?!)", visible=_default_is_v2_gpu)
                                    _ex_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "audio_ref")
                                    _ex_files = [os.path.join(_ex_dir, f) for f in
                                                 ("example.wav", "example_2.wav", "example_3.wav", "example_4.wav")]
                                    _ex_texts = ["Ví dụ 2. Tính trung bình của dãy số.",
                                                 "Trên thực tế, các nghi ngờ đã bắt đầu xuất hiện.",
                                                 "Cậu có nhìn thấy không?",
                                                 "Tết là dịp mọi người háo hức đón chào một năm mới với nhiều hy vọng và mong ước."]
                                    with gr.Accordion("🎧 Chưa ghi âm được? Thử bằng giọng mẫu có sẵn", open=False):
                                        if _default_is_v2_gpu:
                                            # v2 cần cả nội dung audio mẫu.
                                            gr.Examples(examples=[[f, t] for f, t in zip(_ex_files, _ex_texts)],
                                                        inputs=[custom_audio, custom_text],
                                                        label="Bấm vào một dòng để dùng làm giọng mẫu")
                                        else:
                                            # v3 chỉ cần audio → bảng gọn, không hiện cột nội dung.
                                            gr.Examples(examples=[[f] for f in _ex_files], inputs=[custom_audio],
                                                        example_labels=[f"Giọng mẫu {i}" for i in range(1, 5)],
                                                        label="Bấm một giọng mẫu để thử")
                        
                            # 🎭 Bộ giọng cảm xúc: thêm các đoạn mẫu cảm xúc của cùng một người, rồi đọc theo thẻ.
                            with gr.TabItem("🎭 Giọng cảm xúc", id="emotion_mode",
                                            visible=_default_is_v3 and not _HIDE_USER_VOICES) as tab_emotion:
                                gr.Markdown(EMOTION_GUIDE)
                                with gr.Accordion("➕ Thêm cảm xúc cho giọng của bạn", open=True):
                                    with gr.Row():
                                        emo_set_name = gr.Dropdown(
                                            choices=[], value=None, allow_custom_value=True,
                                            label="Tên người / bộ giọng",
                                            info="Gõ tên mới (vd: Hoa) hoặc chọn bộ đã có để thêm cảm xúc.",
                                        )
                                        emo_label = gr.Dropdown(
                                            choices=PRESET_EMOTIONS, value="ấm áp", allow_custom_value=True,
                                            label="Cảm xúc", info="Chọn trong danh sách hoặc tự gõ tên cảm xúc.",
                                        )
                                    emo_audio = gr.Audio(
                                        label="🎙️ Đoạn giọng mang cảm xúc này (5–8 giây, chỉ một cảm xúc)",
                                        type="filepath", sources=["microphone", "upload"],
                                    )
                                    emo_upload_btn = gr.UploadButton(
                                        "📁 Tải file ghi âm / video lên (wav, mp3, m4a, mp4)",
                                        file_types=["audio", "video"], file_count="single", type="filepath",
                                        variant="secondary",
                                    )
                                    emo_audio_msg = gr.Markdown(visible=False)
                                    with gr.Row():
                                        emo_accent = gr.Dropdown(
                                            choices=[("Để máy tự đoán", "auto"), ("Miền Bắc", "bac"), ("Miền Nam", "nam")],
                                            value="auto", label="🗺️ Giọng miền", scale=2,
                                        )
                                        emo_denoise = gr.Checkbox(value=False, label="🔇 Lọc tạp âm",
                                                                  info="Chỉ bật khi bản ghi ồn.", scale=1)
                                    emo_overwrite = gr.Checkbox(value=False, label="Ghi đè nếu cảm xúc này đã có trong bộ")
                                    emo_consent = gr.Checkbox(
                                        value=False,
                                        label="Tôi xác nhận đây là giọng của tôi, hoặc người có giọng này đã đồng ý cho dùng.",
                                    )
                                    btn_add_emotion = gr.Button("➕ Lưu cảm xúc này", variant="primary")
                                    emo_add_status = gr.Markdown(visible=False)
                                gr.Markdown("#### 🎭 Đọc theo cảm xúc", elem_classes="step-title")
                                with gr.Row():
                                    emotion_set_select = gr.Dropdown(choices=[], value=None, label="Bộ giọng cảm xúc")
                                    emotion_default_select = gr.Dropdown(
                                        choices=[], value=None, label="Cảm xúc mặc định",
                                        info="Dùng cho phần văn bản chưa gắn thẻ.",
                                    )
                                emotion_set_info = gr.Markdown(EMOTION_EMPTY_INFO, elem_classes="hint")
                                btn_emotion_example = gr.Button("📝 Điền văn bản mẫu có thẻ cảm xúc vào bước ①",
                                                                variant="secondary", size="sm")

                        # Chỉ còn một chế độ nên ẩn đi (giá trị vẫn truyền vào synthesize_speech).
                        generation_mode = gr.Radio(
                            ["Standard (Một lần)"],
                            value="Standard (Một lần)",
                            label="Chế độ sinh",
                            visible=False,
                        )
                        gr.Markdown("### ③ Bấm để nghe", elem_classes="step-title")
                        btn_generate = gr.Button("🔊 Đọc thành tiếng", variant="primary", size="lg",
                                                 interactive=False, elem_classes="big-btn")

                    # --- TAB 2: MULTI-SPEAKER CONVERSATION ---
                    with gr.Tab("🎭 Hội thoại nhiều người", id="conv_tab", visible=False) as conv_tab:
                        gr.Markdown(
                            "**Cách dùng:** ① viết kịch bản, mỗi đoạn theo dạng **Tên: lời nói** (xem ví dụ bên dưới) "
                            "→ ② bấm **🔍 Tìm người nói** rồi chọn giọng cho từng người → ③ bấm **🎭 Đọc hội thoại**.",
                            elem_classes="hint",
                        )
                        conv_script_input = gr.Textbox(
                            label="Kịch bản hội thoại",
                            placeholder="Phương: Chào mọi người, mình là Phương...",
                            lines=10,
                            elem_classes="script-box",
                            value=('Lan: Chào anh Minh, chiều nay anh có rảnh không?\n\n'
                                   'Minh: Chào Lan, anh rảnh. Có chuyện gì vậy em?\n\n'
                                   'Lan: Em muốn nhờ anh xem giúp bản báo cáo tháng này.\n\n'
                                   'Minh: Được thôi, em gửi cho anh nhé.'),
                        )
                        
                        with gr.Row():
                            btn_detect_speakers = gr.Button("🔍 Tìm người nói", size="sm", variant="secondary")
                            silence_slider = gr.Slider(minimum=0, maximum=3, value=0.1, step=0.1,
                                                       label="⏱️ Nghỉ giữa các câu (giây)")

                        gr.Markdown("### 🎤 Chọn giọng cho từng người", elem_classes="step-title")

                        # Pre-build MAX_SPEAKERS speaker slot rows
                        speaker_name_boxes = []
                        speaker_voice_dds  = []
                        speaker_slot_rows  = []

                        for _i in range(MAX_SPEAKERS):
                            # Mặc định cho 3 nhân vật đầu tiên theo yêu cầu
                            _default_name = ""
                            _default_voice = None
                            _row_visible = False
                            
                            # Khớp kịch bản mẫu (Lan, Minh) với giọng v3 có thật, để bấm đọc được ngay.
                            if _i == 0:
                                _default_name = "Lan"
                                _default_voice = "Trúc Ly"
                                _row_visible = True
                            elif _i == 1:
                                _default_name = "Minh"
                                _default_voice = "Minh Đức"
                                _row_visible = True

                            with gr.Row(visible=_row_visible) as _row:
                                _name = gr.Textbox(
                                    value=_default_name,
                                    label="👤 Nhân vật",
                                    interactive=False,
                                    scale=1,
                                    min_width=120
                                )
                                _dd = gr.Dropdown(
                                    choices=PRESET_VOICES_CACHE,
                                    value=_default_voice,
                                    label="🎤 Giọng đọc",
                                    interactive=True,
                                    scale=3,
                                    allow_custom_value=True
                                )
                            speaker_slot_rows.append(_row)
                            speaker_name_boxes.append(_name)
                            speaker_voice_dds.append(_dd)
                        
                        btn_generate_conv = gr.Button("🎭 Đọc hội thoại", variant="primary", size="lg",
                                                      interactive=False, elem_classes="big-btn")

                with gr.Accordion("⚙️ Tuỳ chỉnh giọng đọc (không bắt buộc)", open=False):
                    # Đọc nhiều đoạn cùng lúc chỉ có tác dụng trên card đồ hoạ NVIDIA
                    # (v3: batch engine chỉ chạy khi model ở CUDA; bản cũ: cần LMDeploy)
                    # → máy không có card NVIDIA thì ẩn hai mục này.
                    with gr.Row(visible=_has_nvidia_gpu()):
                        use_batch = gr.Checkbox(
                            value=True,
                            label="⚡ Đọc nhiều đoạn cùng lúc",
                            info="Văn bản dài được chia thành nhiều đoạn. Bật mục này để card đồ hoạ NVIDIA đọc nhiều đoạn một lúc, nhanh hơn. Nên để bật."
                        )
                        max_batch_size_run = gr.Slider(
                            minimum=1,
                            maximum=32,
                            value=default_batch_size,
                            step=1,
                            label="📊 Số đoạn đọc cùng lúc",
                            info="Số càng lớn đọc càng nhanh nhưng tốn bộ nhớ card đồ hoạ hơn. Nếu báo lỗi hết bộ nhớ (Out of Memory), hãy giảm số này xuống."
                        )
                    with gr.Row():
                        temperature_slider = gr.Slider(
                            minimum=0.1, maximum=1.5, value=default_temp, step=0.1,
                            label="🌡️ Độ biến hoá của giọng",
                            info="Cao: giọng biểu cảm, đa dạng hơn nhưng dễ đọc lỗi. Thấp: giọng đều, ổn định hơn."
                        )
                        max_chars_chunk_slider = gr.Slider(
                            minimum=128, maximum=512,
                            value=256, step=32,
                            label="📝 Độ dài mỗi đoạn (ký tự)",
                            info="Văn bản dài được chia thành các đoạn ngắn, mỗi đoạn tối đa ngần này ký tự (mặc định 256)."
                        )
                
                # State to track current mode
                current_mode_state = gr.State("preset_mode")

            # --- OUTPUT --- (ngay dưới nút đọc, để trên điện thoại bấm xong là thấy kết quả)
            with gr.Column(scale=2, elem_classes="result-card"):
                gr.Markdown("### 🎧 Kết quả", elem_classes="step-title")
                audio_output = gr.Audio(
                    label="Giọng đọc đã tạo",
                    type="filepath",
                    autoplay=True,
                    interactive=False,
                )
                status_output = gr.Markdown(
                    "Kết quả sẽ hiện ở đây sau khi bạn bấm **🔊 Đọc thành tiếng**.",
                    elem_classes="status-md",
                )
                estimate_output = gr.Markdown(elem_classes="estimate-md")
                with gr.Row():
                    download_btn = gr.DownloadButton(
                        "⬇️ Tải file về máy",
                        variant="primary",
                        visible=False,
                        elem_classes="download-btn"
                    )
                    btn_stop = gr.Button("⏹️ Dừng đọc", variant="stop", interactive=False)
                gr.Markdown("<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>🔒 File có gắn dấu ẩn để nhận biết đây là giọng do máy tạo.</div>")
        
        codec_select.change(
            on_codec_change, 
            inputs=[codec_select, current_mode_state], 
            outputs=[tab_custom, tabs, current_mode_state]
        )
        
        # Bind tab events to update state
        tab_preset.select(lambda: "preset_mode", outputs=current_mode_state)
        tab_custom.select(lambda: "custom_mode", outputs=current_mode_state)
        tab_emotion.select(lambda: "emotion_mode", outputs=current_mode_state)
        
        custom_audio.change(validate_clone_audio, inputs=[custom_audio], outputs=[cloning_warning_msg])
        clone_upload_btn.upload(on_clone_file_upload, inputs=[clone_upload_btn], outputs=[custom_audio])
        btn_save_voice.click(
            save_cloned_voice,
            inputs=[custom_audio, save_voice_name, save_voice_desc, save_voice_consent,
                    denoise_checkbox, accent_select, accent_strength],
            outputs=[save_voice_status, voice_select, *speaker_voice_dds],
            concurrency_id="tts",
        )
        # 🎭 Bộ giọng cảm xúc
        emotion_set_outputs = [emo_set_name, emotion_set_select, emotion_default_select, emotion_set_info]
        emo_upload_btn.upload(on_clone_file_upload, inputs=[emo_upload_btn], outputs=[emo_audio])
        emo_audio.change(validate_clone_audio, inputs=[emo_audio], outputs=[emo_audio_msg])
        btn_add_emotion.click(
            save_emotion_voice,
            inputs=[emo_audio, emo_set_name, emo_label, emo_consent, emo_overwrite, emo_denoise, emo_accent],
            outputs=[emo_add_status, *emotion_set_outputs, voice_select],
            concurrency_id="tts",
        )
        emotion_set_select.change(on_emotion_set_change, inputs=[emotion_set_select],
                                  outputs=[emotion_default_select, emotion_set_info])
        # Mở tab thì làm mới danh sách bộ (bộ vừa tạo ở phiên khác cũng hiện), giữ bộ đang chọn.
        tab_emotion.select(emotion_set_updates, inputs=[emotion_set_select], outputs=emotion_set_outputs)
        btn_emotion_example.click(emotion_example_text, inputs=[emotion_set_select], outputs=[text_input])
        # ▶ Nghe thử giọng có sẵn; đổi giọng thì ẩn bản nghe thử cũ. Chung hàng đợi "tts" với nút
        # đọc để máy yếu không phải chạy hai lượt đọc cùng lúc.
        btn_preview.click(preview_voice, inputs=[voice_select], outputs=[preview_audio],
                          concurrency_id="tts")
        voice_select.change(lambda _v: gr.update(value=None, visible=False),
                            inputs=[voice_select], outputs=[preview_audio])
        # Thanh "Mức đẩy" chỉ hiện khi đã chọn Bắc/Nam.
        accent_select.change(lambda a: gr.update(visible=a in ACCENTS),
                             inputs=[accent_select], outputs=[accent_strength])

        # --- Custom Model Event Handlers ---

        def on_backbone_change(choice):
            is_custom = (choice == "Custom Model")
            is_v3 = "v3" in (choice or "").lower()
            is_v2_gpu = (choice == "VieNeu-TTS-v2 (GPU)")
            # Voice Cloning: v3 clones from audio only; v2 (GPU) clones from
            # audio + a reference transcript. Both expose the cloning tab.
            clone_ok = is_v3 or is_v2_gpu
            print(f"   🔄 Backbone changed to: {choice}")
            
            # 1. Device logic
            # Allow hardware acceleration (MPS/CUDA/Auto) for all GPU models AND Turbo (GGUF) models
            is_hw_accel_supported = "(GPU)" in choice or "v2-Turbo" in choice or "v3" in choice.lower() or is_custom
            
            if is_hw_accel_supported:
                dev_choices = get_available_devices()
                initial_dev = "Auto"
            else:
                dev_choices = ["CPU"]
                initial_dev = "CPU"
            
            # 2. Parameter logic
            if is_v3:
                # v3 Turbo uses its own MOSS codec (PyTorch); 0.8 khớp bản tham chiếu.
                codec_update = gr.update(value="VieNeu-Codec", interactive=False)
                text_update = gr.update(value=DEFAULT_TEXT_V3)
                temp_update = gr.update(value=0.8)
            elif "Turbo" in choice:
                codec_update = gr.update(value="VieNeu-Codec", interactive=False)
                text_update = gr.update(value=DEFAULT_TEXT_TURBO)
                temp_update = gr.update(value=0.4)
            elif "(CPU)" in choice:
                codec_update = gr.update(value="NeuCodec (ONNX)", interactive=False)
                text_update = gr.update(value=DEFAULT_TEXT_GPU)
                temp_update = gr.update(value=0.7)
            else:
                codec_update = gr.update(value="NeuCodec (Distill)", interactive=False)
                text_update = gr.update(value=DEFAULT_TEXT_GPU)
                temp_update = gr.update(value=0.7)
            # Không ghi đè văn bản người dùng đã gõ/dán khi đổi mô hình.
            text_update = gr.update()

            # Reference-transcript box + info text differ between v2 and v3 clone.
            if is_v2_gpu:
                clone_info_update = gr.update(value=(
                    "ℹ️ **Voice Cloning (VieNeu-TTS v2).** Tải lên audio mẫu 3–5 giây "
                    "và **nhập đúng nội dung** của audio đó (kể cả dấu câu) — v2 cần "
                    "reference transcript để clone giọng."
                ))
            else:
                clone_info_update = gr.update(value=CLONE_GUIDE_V3)

            return (
                gr.update(visible=is_custom),
                codec_update,
                text_update,
                temp_update,
                gr.update(choices=device_choices(dev_choices), value=initial_dev),
                gr.update(visible=clone_ok),   # cloning_elements_group
                gr.update(visible=clone_ok),   # tab_custom — clone tab (v3 + v2 GPU)
                gr.update(value=32 if is_v3 else 4),  # max_batch_size_run — v3 batches chunks
                gr.update(visible=not is_v3),  # use_lmdeploy_cb — irrelevant for v3 (PyTorch, no LMDeploy)
                gr.update(visible=is_v2_gpu),  # custom_text — only v2 needs a reference transcript
                clone_info_update,             # clone_info_md
                gr.update(value=256),  # max_chars_chunk_slider
                gr.update(visible=is_v3),      # accent_row — chọn giọng miền chỉ có ở v3
                gr.update(visible=is_v3 and not _HIDE_USER_VOICES),  # save_voice_panel — lưu giọng chỉ có ở v3
                gr.update(visible=is_v3 and not _HIDE_USER_VOICES),  # tab_emotion — giọng cảm xúc chỉ có ở v3
            )

        backbone_select.change(
            on_backbone_change,
            inputs=[backbone_select],
            outputs=[
                custom_model_group,
                codec_select,
                text_input,
                temperature_slider,
                device_choice,
                cloning_elements_group,
                tab_custom,
                max_batch_size_run,
                use_lmdeploy_cb,
                custom_text,
                clone_info_md,
                max_chars_chunk_slider,
                accent_row,
                save_voice_panel,
                tab_emotion,
            ]
        )
        
        custom_backbone_model_id.change(
            on_custom_id_change,
            inputs=[custom_backbone_model_id],
            outputs=[custom_backbone_base_model, custom_audio, custom_text]
        )

        btn_load.click(
            fn=load_model,
            inputs=[backbone_select, codec_select, device_choice, use_lmdeploy_cb,
                    custom_backbone_model_id, custom_backbone_base_model, custom_backbone_hf_token],
            outputs=[model_status, btn_generate, btn_generate_conv, btn_load, btn_stop, voice_select,
                     tab_preset, tab_custom, tabs, current_mode_state,
                     conv_tab,
                     *speaker_voice_dds],
            # Chờ lượt đọc đang chạy xong rồi mới tải lại model (không làm hỏng lượt của người khác).
            concurrency_id="tts",
        )

        # Số chữ + thời lượng ước tính, tính ngay trên trình duyệt khi gõ.
        text_input.change(None, inputs=[text_input], outputs=[length_info], js=_LENGTH_JS)

        # --- PDF Upload Event Handlers ---
        def on_pdf_upload(pdf_file):
            """Extract text from uploaded PDF and populate text input."""
            if not pdf_file:
                return gr.update(), gr.update(visible=False)
            extracted = extract_text_from_pdf(pdf_file)
            if extracted.startswith("⚠️"):
                return gr.update(), gr.update(value=extracted, visible=True)
            char_count = len(extracted)
            return (
                gr.update(value=extracted),
                gr.update(
                    value=f"✅ Đã trích xuất **{char_count:,}** ký tự từ file PDF. Bạn xem lại trong ô văn bản ở trên.",
                    visible=True
                )
            )

        pdf_upload.upload(
            fn=on_pdf_upload,
            inputs=[pdf_upload],
            outputs=[text_input, pdf_status]
        )

        # --- Conversation Event Handlers ---
        # Scan speakers → update all 8 slot rows/names/dropdowns
        btn_detect_speakers.click(
            fn=extract_speakers_from_script,
            inputs=[conv_script_input],
            outputs=speaker_name_boxes + speaker_voice_dds + speaker_slot_rows
        )
        
        conv_gen_event = btn_generate_conv.click(
            fn=synthesize_conversation_with_empty_estimate,
            inputs=[conv_script_input,
                    *speaker_name_boxes,
                    *speaker_voice_dds,
                    silence_slider, temperature_slider, max_chars_chunk_slider,
                    session_id_state],
            outputs=[audio_output, status_output, estimate_output],
            concurrency_id="tts",
            scroll_to_output=True,
        )
        btn_generate_conv.click(lambda: QUEUED_MSG, outputs=[status_output], queue=False)
        btn_generate_conv.click(lambda: gr.update(visible=False), outputs=[download_btn])
        btn_generate_conv.click(lambda: gr.update(interactive=True), outputs=btn_stop)
        conv_gen_event.then(lambda: gr.update(interactive=False), outputs=btn_stop)

        # --- Auto-adjust Temperature on Tab Switch ---
        conv_tab.select(
            fn=lambda bb: gr.update(value=0.8 if "v3" in (bb or "").lower() else 1.0),
            inputs=backbone_select,
            outputs=temperature_slider
        )
        single_tab.select(
            fn=lambda bb: gr.update(value=0.8 if "v3" in (bb or "").lower() else default_temp),
            inputs=backbone_select,
            outputs=temperature_slider
        )
        
        # --- Standard Generation Handlers ---
        gen_event = btn_generate.click(
            fn=synthesize_speech_with_estimate,
            inputs=[text_input, voice_select, custom_audio, custom_text, current_mode_state,
                    generation_mode, use_batch, max_batch_size_run,
                    temperature_slider, max_chars_chunk_slider,
                    denoise_checkbox, session_id_state,
                    accent_select, accent_strength,
                    emotion_set_select, emotion_default_select],
            outputs=[audio_output, status_output, estimate_output],
            concurrency_id="tts",
            scroll_to_output=True,
        )
        # Báo ngay khi bấm (không xếp hàng): máy chỉ đọc một lượt một lúc nên có thể phải chờ.
        btn_generate.click(lambda: QUEUED_MSG, outputs=[status_output], queue=False)
        btn_generate.click(lambda: gr.update(visible=False), outputs=[download_btn])
        btn_generate.click(lambda: gr.update(interactive=True), outputs=btn_stop)
        gen_event.then(lambda: gr.update(interactive=False), outputs=btn_stop)

        # --- Stop Button ---
        def request_stop(session_id):
            # Chỉ dừng khi lượt đang chạy là của chính phiên này (link chia sẻ cho nhiều người).
            if session_id and session_id != _RUNNING_SESSION:
                return (gr.update(), "ℹ️ Lượt đọc của bạn chưa bắt đầu (đang chờ người khác đọc xong) hoặc đã xong.",
                        gr.update(), gr.update(), gr.update())
            print("🛑 STOP REQUESTED via button click.")
            _STOP_EVENT.set()
            return None, "⏹️ Đã dừng đọc.", "", gr.update(interactive=False), gr.update(visible=False)

        # Handler: set stop event + update UI
        # Note: We avoid cancels= here to prevent internal Gradio KeyError crashes,
        # relying instead on the frequent _STOP_EVENT.is_set() checks in the code.
        btn_stop.click(fn=request_stop, inputs=[session_id_state],
                       outputs=[audio_output, status_output, estimate_output, btn_stop, download_btn])

        # --- Download Button Event Handlers ---
        def on_audio_generated(audio_path):
            """Show download button when audio is generated."""
            if audio_path and os.path.exists(audio_path):
                return gr.update(value=audio_path, visible=True)
            return gr.update(visible=False)

        # Connect to generation events (must be after gen_event/conv_gen_event are defined)
        gen_event.then(
            fn=on_audio_generated,
            inputs=[audio_output],
            outputs=[download_btn]
        )
        conv_gen_event.then(
            fn=on_audio_generated,
            inputs=[audio_output],
            outputs=[download_btn]
        )
        # Mỗi phiên trình duyệt một mã riêng (dùng cho nút Dừng).
        demo.load(lambda: uuid.uuid4().hex, outputs=[session_id_state])

        # Mở trang: chờ model (tải sẵn ở nền) rồi điền danh sách giọng, hiện các tab cho phiên này.
        page_load_event = demo.load(
            fn=on_page_load,
            outputs=[model_status, btn_generate, btn_generate_conv, btn_load, btn_stop, voice_select,
                     tab_preset, tab_custom, tabs, current_mode_state,
                     conv_tab,
                     *speaker_voice_dds]
        )
        # Model sẵn sàng rồi mới điền danh sách bộ giọng cảm xúc.
        page_load_event.then(emotion_set_updates, outputs=emotion_set_outputs)

def main():
    # Cho phép override từ biến môi trường (hữu ích cho Docker)
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    # Check running in Colab
    is_on_colab = os.getenv("COLAB_RELEASE_TAG") is not None

    # Default:
    # - Colab: share=True (convenient)
    # - Docker/local: share=False (safe)
    share = env_bool("GRADIO_SHARE", default=is_on_colab)
    
    # If server_name is "0.0.0.0" and GRADIO_SHARE is not set, disable sharing
    if server_name == "0.0.0.0" and os.getenv("GRADIO_SHARE") is None:
        share = False

    # Tải sẵn model mặc định ở nền ngay khi khởi động → người dùng mở trang là dùng được.
    if _PRELOAD_ENABLED:
        _ensure_preload_started()

    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)

if __name__ == "__main__":
    main()
