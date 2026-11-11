# Hướng dẫn cho Claude: cài và chạy Clone-voice-TTS

Phần lớn người nhờ bạn cài phần mềm này KHÔNG biết lập trình. Khi họ nhờ cài, chạy, mở lại hay cập nhật phần mềm:
trả lời bằng tiếng Việt, câu ngắn, không thuật ngữ; tự làm hết các bước; chỉ hỏi khi cần họ bấm cho phép hoặc tự quyết
định. Có lựa chọn kỹ thuật thì tự chọn cách an toàn, phổ biến nhất. Báo tiến độ bằng lời dễ hiểu ("đang tải mô hình giọng,
khoảng 5 phút") và nhắc họ để ý các nút cho phép, vì bạn phải chờ họ bấm.

**Tên phần mềm là Clone-voice-TTS.** Khi nói với người dùng, luôn gọi đúng tên này, kể cả khi README, giao diện web
hay mã nguồn còn ghi tên dự án gốc. Không đổi các tên kỹ thuật: lệnh `vieneu-web`, gói `vieneu`, thư mục `~/.cache/vieneu`.

## Cài đặt — chỉ dùng đúng các lệnh này

0. **Thư mục.** Repo nằm thẳng trong thư mục đang mở, không tạo thư mục con, để lần sau mở lại thư mục này là có ngay
   file CLAUDE.md. Thư mục trống: `git clone https://github.com/huutien1904/clone-voice .`; không trống thì clone vào thư mục
   tạm rồi chuyển hết file (kể cả file ẩn như `.claude`, `.git`) sang. Nếu đường dẫn nằm trong OneDrive hoặc iCloud
   (có chữ `OneDrive`, `Mobile Documents`), báo người dùng và đề nghị dùng `C:\Clone-voice-TTS` (Mac: `~/Clone-voice-TTS`),
   vì thư mục thư viện gần 1 GB sẽ bị đồng bộ lên mạng.
1. **Git** (nếu chưa có): Windows `winget install --id Git.Git -e --source winget` (Windows có thể hiện hộp hỏi quyền,
   dặn người dùng bấm Yes, hộp đó đôi khi chỉ nhấp nháy ở thanh Taskbar); macOS `xcode-select --install` (người dùng bấm Install).
   Phiên hiện tại chưa nhận lệnh `git` thì gọi bằng đường dẫn đầy đủ, ví dụ `"C:\Program Files\Git\cmd\git.exe"`.
2. **uv.** Kiểm tra `uv --version`. Chưa có thì cài:
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`

   Lệnh cài bị chặn (thường do ô chế độ quyền đang để Auto): nhờ người dùng chuyển ô đó sang **Manual** rồi thử lại.
   Cài xong mà lệnh `uv` chưa nhận, gọi bằng đường dẫn đầy đủ: Windows `$env:USERPROFILE\.local\bin\uv.exe`,
   macOS `~/.local/bin/uv`. Không cần cài Python: uv tự tải Python 3.12.
3. **ffmpeg** (để nhận file video và file ghi âm m4a của điện thoại): Windows `winget install --id Gyan.FFmpeg -e`;
   macOS `brew install ffmpeg` nếu máy có Homebrew, không có thì bỏ qua và báo người dùng chỉ tải lên được file wav/mp3.
   Phiên hiện tại chưa nhận lệnh `ffmpeg` thì thêm thư mục chứa `ffmpeg` vào PATH của lệnh chạy web ở bước 5.
4. **Thư viện** (trong thư mục repo): `uv sync --frozen --extra pdf`
5. **Chạy web** (chạy nền, tiến trình chạy lâu dài): `uv run --frozen --extra pdf vieneu-web`
   - Trang web lên (`http://127.0.0.1:7860` trả HTTP 200) trước khi mô hình giọng tải xong. Lần đầu mô hình (khoảng 0,3 GB)
     tải ngầm vài phút; trang hiện "⏳ Đang chuẩn bị giọng đọc..." rồi mới đến "✅ Sẵn sàng!".
     Kiểm tra HTTP 200 định kỳ, đừng chờ dòng log (log có thể bị giữ trong bộ đệm; đặt `PYTHONUNBUFFERED=1` nếu cần đọc).
   - Trang lên thì mở cho người dùng: Windows `start http://127.0.0.1:7860`, macOS `open http://127.0.0.1:7860`, kèm link
     trong khung chat. Dặn họ chờ dòng "✅ Sẵn sàng!", rồi nói ngắn gọn: gõ chữ ở ①, chọn giọng ở ②, bấm
     "Đọc thành tiếng" ở ③; muốn clone giọng thì vào tab "Giọng của bạn (clone)".
   - Dặn họ: lúc này phần mềm chạy bên trong Claude, cứ để Claude mở (thu nhỏ cũng được); tắt Claude là phần mềm tắt theo.

## Không được làm

- Không chạy `uv sync` hay `uv run` thiếu `--frozen`: sẽ lỗi 404 khi tải `llama_cpp_python-0.3.16-...whl`
  (file cũ đã bị gỡ khỏi GitHub; bản CPU không cần nó).
- Không dùng `--group gpu`, `--group dev`, `--extra legacy`: kéo PyTorch nhiều GB và trên Windows cũng dính lỗi 404 trên.
  Bản CPU là đủ, không cần card đồ hoạ.
- Không `pip install vieneu`: đó là bản gốc trên PyPI, không phải bản trong repo này.
- Không sửa `pyproject.toml`, `uv.lock` để "chữa" lỗi cài đặt.

## Lỗi thường gặp

- **Cổng 7860 đang bận** (app khác chiếm): đặt biến môi trường `GRADIO_SERVER_PORT=7861` khi chạy, rồi mở cổng đó.
- **`WinError 1314` khi tải mô hình** (Windows không cho tạo symlink): nhờ người dùng bật Developer Mode
  (Settings → System → For developers), hoặc chép file từ `blobs/` sang đúng đường dẫn trong `snapshots/<rev>/`
  của thư mục `~/.cache/huggingface/hub/models--...` rồi chạy lại.
- **Báo thiếu bộ nhớ / web rất chậm:** cần từ 8 GB RAM; nhờ người dùng đóng bớt Chrome, phần mềm dựng video rồi chạy lại.
- **Nút "Lấy chữ từ file PDF" báo thiếu thư viện:** chạy lại bước 4 (có `--extra pdf`). PDF dạng ảnh scan thì không lấy được chữ.
- **Tải file video/m4a lên báo chưa cài ffmpeg:** làm bước 3 rồi chạy lại web.
- **Giọng clone nghe chưa giống:** đoạn mẫu nên dài 5–8 giây, một người nói, ít tạp âm, chọn đúng giọng miền; chỉ bật
  "🔇 Lọc tạp âm cho giọng mẫu" khi bản ghi ồn.

## Mở lại, tắt, cập nhật, sao lưu

- **Mở lại:** chạy lại bước 5 trong thư mục repo (không cần cài lại).
- **Tắt:** dừng tiến trình đang chạy web.
- **Cập nhật:** tắt web, chạy `git pull` trong thư mục repo, rồi bước 4 và bước 5. Giọng đã lưu không mất.
- **Sao lưu giọng:** chép `~/.cache/vieneu/user_voices.json` ra Desktop. **Khôi phục trên máy mới:** cài xong thì chép file đó
  về `~/.cache/vieneu/` (máy mới đã có giọng riêng thì hỏi người dùng trước khi ghi đè).
- **Biểu tượng trên Desktop:** sau lần chạy đầu thành công, đề nghị tạo để lần sau họ bấm đúp là mở, không cần Claude.
  - Windows: file `Clone-voice-TTS.bat` trên Desktop, lưu dạng UTF-8 không BOM. Thay `<repo>` bằng đường dẫn thật của
    thư mục repo và `<uv>` bằng đường dẫn thật của uv (`(Get-Command uv).Source`, thường là `%USERPROFILE%\.local\bin\uv.exe`):
    ```bat
    @echo off
    chcp 65001 >nul
    cd /d "<repo>"
    start "" powershell -NoProfile -WindowStyle Hidden -Command "while (-not (Test-NetConnection 127.0.0.1 -Port 7860 -InformationLevel Quiet -WarningAction SilentlyContinue)) { Start-Sleep 2 }; Start-Process 'http://127.0.0.1:7860'"
    "<uv>" run --frozen --extra pdf vieneu-web
    ```
  - macOS: file `Clone-voice-TTS.command` trên Desktop (nhớ `chmod +x`), chạy `cd "<repo>"`, mở trình duyệt khi cổng 7860 lên,
    rồi `~/.local/bin/uv run --frozen --extra pdf vieneu-web`.
  - Dặn người dùng: bấm đúp sẽ mở một cửa sổ chạy nhiều chữ tiếng Anh (Windows: cửa sổ đen; Mac: Terminal), đó là phần mềm
    đang chạy, không phải lỗi; khoảng một phút sau web tự mở. Giữ cửa sổ đó mở trong lúc dùng; đóng nó là tắt phần mềm.

## Ghi chú

- Hướng dẫn bằng lời cho người dùng nằm ở `HUONG_DAN.md`. Khi họ hỏi cách dùng, trả lời theo đúng tên nút trong file đó.
- Giọng người dùng tự lưu nằm ở `~/.cache/vieneu/user_voices.json`, chỉ có trên máy đó.
- Văn bản và giọng được xử lý ngay trên máy người dùng; web chỉ mở ở `127.0.0.1` (không ai khác truy cập được).
  Dòng "đang chạy trên CPU của máy chủ" trên web nghĩa là chính máy của họ.
