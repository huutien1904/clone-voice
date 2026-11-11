import importlib
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


class TestV3TurboClonePreclean:
    def test_preclean_reference_audio_trims_silence_and_returns_path(self):
        from vieneu.v3turbo import V3TurboVieNeuTTS

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "raw_ref.wav"
            sr = 16000
            wav = np.zeros(sr * 2, dtype=np.float32)
            wav[sr // 2: sr * 3 // 2] = 0.5
            sf.write(src, wav, sr)

            cleaned = V3TurboVieNeuTTS._preclean_reference_audio(src)

            try:
                assert cleaned is not None
                assert Path(cleaned).exists()
                clean_wav, clean_sr = sf.read(cleaned, dtype="float32", always_2d=False)
                assert clean_sr == sr
                assert len(clean_wav) < len(wav)
            finally:
                if cleaned and Path(cleaned).exists():
                    try:
                        Path(cleaned).unlink()
                    except Exception:
                        pass

    def test_preclean_reads_m4a_via_ffmpeg(self, tmp_path):
        """File ghi âm điện thoại (.m4a) soundfile không đọc được → giải mã qua ffmpeg."""
        import shutil
        import subprocess

        import pytest

        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg không có trong PATH")
        from vieneu.v3turbo import V3TurboVieNeuTTS

        sr = 48000
        wav = np.zeros(sr * 2, dtype=np.float32)
        wav[sr // 2: sr * 3 // 2] = 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
        src_wav = tmp_path / "rec.wav"
        sf.write(src_wav, wav, sr)
        m4a = tmp_path / "rec.m4a"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src_wav), str(m4a)], check=True)

        cleaned = V3TurboVieNeuTTS._preclean_reference_audio(m4a)
        try:
            clean_wav, clean_sr = sf.read(cleaned, dtype="float32", always_2d=False)
            assert clean_sr == sr
            assert 0.5 * sr < len(clean_wav) < 1.5 * sr  # phần im lặng hai đầu đã bị cắt
        finally:
            Path(cleaned).unlink(missing_ok=True)
