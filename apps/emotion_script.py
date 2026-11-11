"""Đọc văn bản theo cảm xúc bằng "bộ giọng cảm xúc".

Một bộ giọng = nhiều giọng người dùng đã lưu của CÙNG một người, mỗi giọng clone từ một đoạn mẫu
mang một cảm xúc, đặt tên ``"<tên bộ> · <cảm xúc>"`` (vd ``Hoa · vui``, ``Hoa · ấm áp``). Model v3
bắt chước cách nói của đoạn mẫu, nên đổi giọng cảm xúc theo từng đoạn văn = đổi ngữ điệu.

Cú pháp văn bản::

    [ấm áp] Chào bố mẹ, mình là Hoa. [dừng 0.5s] [vui] Đăng kí học thử miễn phí nhé!

- ``[cảm xúc]``: từ đây đọc bằng giọng cảm xúc đó của bộ đang chọn.
- ``[dừng 0.5s]`` / ``[nghỉ 1 giây]``: chèn khoảng lặng (tối đa 5 giây).
- ``[cười]``, ``[thở dài]``, ``[hắng giọng]``: giữ nguyên cho model (âm thanh cảm xúc).
- Chữ khác trong ngoặc vuông (ghi chú đạo diễn như ``[đọc chậm, nhấn chữ này]``): bỏ qua, không đọc to.
- Dấu ``*`` và số thứ tự đầu dòng (``1.``) bị bỏ để không bị đọc thành "sao", "một".
"""
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

SEP = " · "
PRESET_EMOTIONS = ["tự nhiên", "ấm áp", "vui", "hào hứng", "tự hào", "hỏi nhẹ", "thân mật",
                   "nghiêm túc", "buồn", "thì thầm"]
# Thẻ âm thanh cảm xúc mà model hiểu sẵn (xem vieneu_utils.phonemize_text) — không dùng làm tên cảm xúc.
MODEL_CUES = {"cười", "cuoi", "chuckle", "thở dài", "tho dai", "sigh", "hắng giọng", "hang giong", "clear throat"}
MAX_PAUSE = 5.0
SET_NAME_MAX = 40
EMOTION_MAX = 30

_TAG_RE = re.compile(r"\[([^\[\]\n]{1,80})\]")
_PAUSE_RE = re.compile(r"^(?:dừng|dung|nghỉ|nghi|ngừng|ngung|pause)\s*(\d+(?:[.,]\d+)?)\s*"
                       r"(?:s|giây|giay|sec|secs|second|seconds)?$")
_LIST_NUM_RE = re.compile(r"(?m)^[ \t]*\d{1,3}[.)][ \t]+")
_BAD_NAME_CHARS = set("[]·|\n")


def norm_label(s: str) -> str:
    """Chuẩn hoá nhãn để so khớp thẻ: NFC, chữ thường, gộp khoảng trắng."""
    return " ".join(unicodedata.normalize("NFC", str(s or "")).strip().lower().split())


def is_pause_tag(label: str) -> bool:
    return bool(_PAUSE_RE.match(norm_label(label)))


def validate_set_name(name: str) -> Optional[str]:
    """Trả về thông báo lỗi (tiếng Việt) nếu tên bộ giọng không hợp lệ, None nếu ổn."""
    name = " ".join(str(name or "").split())
    if not name:
        return "Hãy nhập tên người / bộ giọng (vd: Hoa)."
    if len(name) > SET_NAME_MAX:
        return f"Tên bộ giọng tối đa {SET_NAME_MAX} ký tự."
    if any(c in _BAD_NAME_CHARS for c in name) or "⚠️" in name:
        return "Tên bộ giọng không được chứa các ký tự [ ] · |"
    return None


def validate_emotion(emotion: str) -> Optional[str]:
    lab = norm_label(emotion)
    if not lab:
        return "Hãy chọn hoặc gõ tên cảm xúc (vd: vui, ấm áp)."
    if len(lab) > EMOTION_MAX:
        return f"Tên cảm xúc tối đa {EMOTION_MAX} ký tự."
    if any(c in _BAD_NAME_CHARS for c in lab):
        return "Tên cảm xúc không được chứa các ký tự [ ] · |"
    if lab in MODEL_CUES:
        return f"'{lab}' là thẻ âm thanh có sẵn của model, hãy đặt tên cảm xúc khác."
    if is_pause_tag(lab):
        return "Tên cảm xúc không được giống lệnh dừng (vd: dừng 0.5s)."
    return None


def emotion_voice_name(set_name: str, emotion: str) -> str:
    """Tên giọng người dùng của một cảm xúc trong bộ, vd ``Hoa · ấm áp``."""
    return f"{' '.join(str(set_name).split())}{SEP}{norm_label(emotion)}"


def split_emotion_voice_name(name: str) -> Optional[Tuple[str, str]]:
    """``"Hoa · vui"`` → ``("Hoa", "vui")``; tên không theo mẫu → None."""
    if not isinstance(name, str) or SEP not in name:
        return None
    base, emo = name.rsplit(SEP, 1)
    base, emo = base.strip(), norm_label(emo)
    return (base, emo) if base and emo else None


def emotion_sets(voice_names: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """Gom giọng theo bộ: ``{tên bộ: {cảm xúc (đã chuẩn hoá): tên giọng}}``, giữ thứ tự xuất hiện."""
    sets: Dict[str, Dict[str, str]] = {}
    for n in voice_names:
        parts = split_emotion_voice_name(n)
        if parts:
            sets.setdefault(parts[0], {})[parts[1]] = n
    return sets


def parse_emotion_script(text: str, emotions: Dict[str, str],
                         default_voice: str) -> Tuple[List[tuple], List[str]]:
    """Tách văn bản thành các bước đọc → ``(items, ghi_chú_bị_bỏ)``.

    ``items``: ``("speak", tên_giọng, đoạn_văn)`` hoặc ``("pause", giây)``, theo đúng thứ tự.
    ``emotions``: ``{cảm xúc đã chuẩn hoá: tên giọng}`` của bộ đang chọn; ``default_voice`` dùng cho
    phần văn bản trước thẻ cảm xúc đầu tiên.
    """
    text = (text or "").replace("*", "")
    text = _LIST_NUM_RE.sub("", text)
    items: List[tuple] = []
    ignored: List[str] = []
    buf: List[str] = []
    cur = default_voice

    def flush():
        seg = "".join(buf).strip()
        buf.clear()
        if seg and re.search(r"\w", seg):
            if items and items[-1][0] == "speak" and items[-1][1] == cur:
                items[-1] = ("speak", cur, items[-1][2] + " " + seg)   # cùng giọng liền nhau → gộp
            else:
                items.append(("speak", cur, seg))

    pos = 0
    for m in _TAG_RE.finditer(text):
        buf.append(text[pos:m.start()])
        pos = m.end()
        raw = m.group(1)
        lab = norm_label(raw)
        pm = _PAUSE_RE.match(lab)
        if pm:
            flush()
            sec = min(float(pm.group(1).replace(",", ".")), MAX_PAUSE)
            if sec > 0:
                items.append(("pause", sec))
            continue
        if lab in emotions:
            flush()
            cur = emotions[lab]
            continue
        if lab in MODEL_CUES:
            buf.append(m.group(0))       # để model tự tạo tiếng cười/thở dài
            continue
        ignored.append(raw.strip())      # ghi chú đạo diễn: bỏ qua, không đọc to
    buf.append(text[pos:])
    flush()
    return items, ignored


SAMPLE_LINES = [
    "Chào bạn, hôm nay mình muốn kể cho bạn nghe một chuyện nhỏ.",
    "Thật tuyệt vời, mọi thứ đều diễn ra đúng như mong đợi!",
    "Bạn có muốn thử cùng mình không?",
    "Cảm ơn bạn đã lắng nghe, hẹn gặp lại nhé.",
]


def example_script(emotions: List[str]) -> str:
    """Văn bản mẫu dùng đúng các cảm xúc của bộ đang chọn (tối đa 4 cảm xúc)."""
    emos = list(emotions)[:4] or ["tự nhiên"]
    lines = []
    for i, emo in enumerate(emos):
        lines.append(f"[{emo}] {SAMPLE_LINES[i % len(SAMPLE_LINES)]}")
        if i < len(emos) - 1:
            lines.append("[dừng 0.5s]")
    return "\n".join(lines)
