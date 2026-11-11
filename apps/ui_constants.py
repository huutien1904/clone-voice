import gradio as gr

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    # Chữ to hơn cho dễ đọc; ô nhập 16px nên iPhone không tự phóng to trang khi chạm vào.
    text_size="lg",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui'],
).set(
    button_primary_background_fill="linear-gradient(90deg, #6366f1 0%, #0ea5e9 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #4f46e5 0%, #0284c7 100%)",
)

css = """
.container { max-width: 1000px; margin: auto; }
/* --- Giao diện cho người mới: các bước ①②③, dòng trạng thái, khung kết quả --- */
.header-subtitle { color: #cbd5e1 !important; font-size: 1.05rem; margin: 6px 0 12px; }
.header-subtitle b { color: #ffffff !important; }
.status-bar {
    padding: 10px 16px;
    border-radius: 10px;
    background: var(--color-accent-soft);
    border: 1px solid var(--border-color-accent);
    margin: 4px 0 10px;
    font-size: 1rem;
}
.step-title h3 { margin: 14px 0 2px !important; font-size: 1.15rem !important; }
.hint { font-size: 0.9rem; color: var(--body-text-color-subdued); }
.big-btn { font-size: 1.15rem !important; min-height: 52px; }
.result-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 12px 16px;
    background: var(--background-fill-secondary);
    margin-top: 12px;
}
.status-md { font-size: 1rem; }
.header-box, .model-card-content, .warning-banner, .status-bar { overflow-wrap: anywhere; }
/* Ẩn chân trang mặc định của Gradio ("Use via API · Built with Gradio") cho người dùng phổ thông. */
footer { display: none !important; }
@media (max-width: 640px) {
    .header-box { padding: 16px 12px; margin-bottom: 12px; }
    .header-title { font-size: 1.6rem; }
    .header-subtitle { font-size: 0.95rem; }
    .model-card-content { flex-direction: column; gap: 4px; font-size: 0.85rem; }
    .model-card-item { flex-wrap: wrap; }
    .warning-banner-item { min-width: 0; }
}
.header-box {
    text-align: center;
    margin-bottom: 25px;
    padding: 25px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 12px;
    color: white !important;
}
.header-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: white !important;
}
.gradient-text {
    background: -webkit-linear-gradient(45deg, #60A5FA, #22D3EE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-icon {
    color: white;
}
.status-box {
    font-weight: 500;
    border: 1px solid rgba(99, 102, 241, 0.1);
    background: rgba(99, 102, 241, 0.03);
    border-radius: 8px;
}
.status-box textarea {
    text-align: center;
    font-family: inherit;
}
.estimate-box {
    font-weight: 500;
    border: 1px solid rgba(99, 102, 241, 0.1);
    background: rgba(99, 102, 241, 0.03);
    border-radius: 8px;
}
.estimate-box textarea {
    text-align: center;
    font-family: inherit;
}
.model-card-content {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 15px;
    font-size: 0.9rem;
    text-align: center;
    color: white !important;
}
.model-card-item {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    color: white !important;
}
.model-card-item strong {
    color: white !important;
}
.model-card-item span {
    color: white !important;
}
.model-card-link {
    color: #60A5FA;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s;
}
.model-card-link:hover {
    color: #22D3EE;
    text-decoration: underline;
}
.warning-banner {
    background-color: #fffbeb;
    border: 1px solid #fef3c7;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
}
.warning-banner-title {
    color: #92400e;
    font-weight: 700;
    font-size: 1.1rem;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}
.warning-banner-grid {
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
}
.warning-banner-item {
    flex: 1;
    min-width: 240px;
    background: #fef3c7;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #fde68a;
}
.warning-banner-item strong {
    color: #b45309;
    display: block;
    margin-bottom: 4px;
    font-size: 0.95rem;
}
.warning-banner-content {
    color: #78350f;
    font-size: 0.9rem;
    line-height: 1.5;
}
.warning-banner-content b {
    color: #451a03;
    background: rgba(251, 191, 36, 0.2);
    padding: 1px 4px;
    border-radius: 4px;
}
.script-box textarea {
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
}
.speaker-table {
    margin-top: 10px;
}
"""

head_html = """
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🦜</text></svg>">
"""

DEFAULT_TEXT_GPU = "Hà Nội, trái tim của Việt Nam, là một thành phố ngàn năm văn hiến với bề dày lịch sử và văn hóa độc đáo. Bước chân trên những con phố cổ kính quanh Hồ Hoàn Kiếm, du khách như được du hành ngược thời gian, chiêm ngưỡng kiến trúc Pháp cổ điển hòa quyện với nét kiến trúc truyền thống Việt Nam. Mỗi con phố trong khu phố cổ mang một tên gọi đặc trưng, phản ánh nghề thủ công truyền thống từng thịnh hành nơi đây như phố Hàng Bạc, Hàng Đào, Hàng Mã. Ẩm thực Hà Nội cũng là một điểm nhấn đặc biệt, từ tô phở nóng hổi buổi sáng, bún chả thơm lừng trưa hè, đến chè Thái ngọt ngào chiều thu. Những món ăn dân dã này đã trở thành biểu tượng của văn hóa ẩm thực Việt, được cả thế giới yêu mến. Người Hà Nội nổi tiếng với tính cách hiền hòa, lịch thiệp nhưng cũng rất cầu toàn trong từng chi tiết nhỏ, từ cách pha trà sen cho đến cách chọn hoa sen tây để thưởng trà."
DEFAULT_TEXT_TURBO = (
    "Trước đây, hệ thống điện chủ yếu sử dụng direct current, nhưng Tesla đã chứng minh rằng alternating current is more efficient for long-distance transmission. Nhờ đó, điện có thể được truyền đi xa hơn với ít tổn thất năng lượng hơn. Đây là một bước tiến cực kỳ quan trọng trong ngành điện.\n\n"
    "Một trong những phát minh nổi tiếng của ông là Tesla coil, một thiết bị có thể tạo ra điện áp rất cao và những tia sét nhân tạo. This device is still used today in demonstrations và trong một số ứng dụng nghiên cứu. Khi nhìn thấy những tia điện này, nhiều người cảm thấy vừa ấn tượng vừa hơi đáng sợ."
)

# v3 Turbo: văn bản mẫu ngắn, không có tag cảm xúc để người mới không bị rối.
DEFAULT_TEXT_V3 = (
    "Xin chào! Đây là giọng đọc tiếng Việt được tạo bằng trí tuệ nhân tạo. "
    "Bạn hãy xoá đoạn này, dán nội dung của mình vào, chọn giọng rồi bấm Đọc thành tiếng nhé."
)
