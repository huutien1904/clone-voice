"""Tách văn bản theo thẻ cảm xúc cho "bộ giọng cảm xúc" (apps/emotion_script.py)."""
import pytest

from apps.emotion_script import (
    emotion_sets,
    emotion_voice_name,
    example_script,
    parse_emotion_script,
    split_emotion_voice_name,
    validate_emotion,
    validate_set_name,
)

EMO = {"ấm áp": "Hoa · ấm áp", "vui": "Hoa · vui", "hỏi nhẹ": "Hoa · hỏi nhẹ"}


def test_voice_name_roundtrip_and_sets():
    assert emotion_voice_name(" Hoa ", "Ấm  Áp") == "Hoa · ấm áp"
    assert split_emotion_voice_name("Hoa · ấm áp") == ("Hoa", "ấm áp")
    assert split_emotion_voice_name("clone_admin_Tùng") is None
    sets = emotion_sets(["Hoa · ấm áp", "Minh Đức", "Hoa · vui", "Lan · buồn", "clone_bản tin vip"])
    assert sets == {"Hoa": {"ấm áp": "Hoa · ấm áp", "vui": "Hoa · vui"}, "Lan": {"buồn": "Lan · buồn"}}


def test_parse_switches_voice_pauses_and_ignores_directions():
    text = ("Mở đầu không thẻ.\n"
            "[hỏi nhẹ] Con nhà mình cứ rảnh là cầm điện thoại?\n"
            "[đọc chậm, nhấn chữ cuối] [dừng 0,5 giây]\n"
            "[Vui] Đăng kí học thử miễn phí nhé! [cười]")
    items, ignored = parse_emotion_script(text, EMO, "Hoa · ấm áp")
    assert items == [
        ("speak", "Hoa · ấm áp", "Mở đầu không thẻ."),
        ("speak", "Hoa · hỏi nhẹ", "Con nhà mình cứ rảnh là cầm điện thoại?"),
        ("pause", 0.5),
        ("speak", "Hoa · vui", "Đăng kí học thử miễn phí nhé! [cười]"),
    ]
    assert ignored == ["đọc chậm, nhấn chữ cuối"]


def test_parse_strips_markdown_stars_and_list_numbers():
    text = "1. *Câu một.*\n2) [vui] *Câu hai!*"
    items, _ = parse_emotion_script(text, EMO, "Hoa · ấm áp")
    assert items == [("speak", "Hoa · ấm áp", "Câu một."), ("speak", "Hoa · vui", "Câu hai!")]


def test_parse_merges_same_voice_and_caps_pause():
    items, _ = parse_emotion_script("[vui] A. [vui] B. [nghỉ 9s] C.", EMO, "Hoa · ấm áp")
    assert items == [("speak", "Hoa · vui", "A. B."), ("pause", 5.0), ("speak", "Hoa · vui", "C.")]


@pytest.mark.parametrize("bad", ["", "cười", "dừng 1s", "a[b", "x" * 31])
def test_validate_emotion_rejects(bad):
    assert validate_emotion(bad) is not None


def test_validate_names_ok_and_bad():
    assert validate_emotion("Ấm áp") is None
    assert validate_set_name("Hoa") is None
    assert validate_set_name("") is not None
    assert validate_set_name("Hoa · 2") is not None


def test_example_script_uses_set_emotions():
    ex = example_script(["ấm áp", "vui"])
    assert ex.startswith("[ấm áp] ") and "[dừng 0.5s]" in ex and "[vui] " in ex
    items, ignored = parse_emotion_script(ex, EMO, "Hoa · ấm áp")
    assert [i[0] for i in items] == ["speak", "pause", "speak"] and not ignored
