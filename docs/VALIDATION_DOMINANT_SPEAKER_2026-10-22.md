# Kiểm chứng tự tìm người nói chính trên bản phỏng vấn dài

Ngày chạy: 2026-10-22

Thiết bị: NVIDIA RTX 5060 Ti 16 GB

File kiểm thử: `YTSave.to_YouTube_Media_RBoZG70Es3Y_Trường-Giang-Tôi-xin-lỗi-tôi-đã-hành-xử-chưa-tốt_008_128k.m4a`

SHA-256: `9bdf5a95e96d63907cc6148c4816174322434181a066ceaf6f21bc098e36ada3`

## Nhận diện cụm giọng

- Thời lượng file: 1.756,24 giây (29:16,24).
- Lời nói phát hiện được: 1.295,2 giây.
- Số đoạn được tạo embedding: 209/209.
- Số cụm giọng ước tính: 2.
- Silhouette cosine: 0,6188.
- Cosine giữa hai tâm cụm: 0,4623.
- Cụm chính: 88,92% phần lời nói, ước tính 1.151,6 giây.
- Mẫu đại diện: 06:27,06–06:34,06; cosine với tâm cụm 0,9353.

Các tỷ lệ trên mô tả cụm âm học trong phần lời nói do Whisper tìm thấy. Chúng không xác minh danh tính thật của người trong bản ghi.

## Tách toàn bộ file

- Pipeline: MossFormer2_SS_16K + VieNeu speaker encoder + FRCRN_SE_16K.
- Kết quả: 148/160 cửa sổ chọn được stem mục tiêu.
- Median cosine của stem được chọn với mẫu: 0,789.
- Median chênh lệch cosine với stem còn lại: 0,443.
- Cả `target.wav` và `without_target.wav` dài đủ 1.756,24 giây.
- Thời gian chạy: 8 phút 08,55 giây, nhanh khoảng 3,6 lần thời gian thực.
- RAM đỉnh: 2.163.596 KiB; không dùng swap.
- VRAM quan sát khi chạy: khoảng 10,8 GB.
- Phiên âm lại `target.wav`: 275 đoạn, tổng khoảng 1.266,8 giây theo mốc ASR.

Với cùng phần cứng và loại nội dung tương tự, file một giờ được ước tính mất khoảng 17 phút cho bước tách và giảm nhiễu. Đây là ngoại suy tuyến tính từ một lần chạy, chưa phải benchmark nhiều file.

## Tệp kết quả cục bộ

- `data/validation/dominant_truong_giang/dominant_speaker.json`
- `data/validation/dominant_truong_giang/dominant_reference.wav`
- `data/validation/target_truong_giang_full_v2/selection.json`
- `data/validation/target_truong_giang_full_v2/quality_metrics.json`
- `data/validation/target_truong_giang_full_v2/target.wav`
- `data/validation/target_truong_giang_full_v2/without_target.wav`

Không tạo hoặc kích hoạt adapter từ file kiểm thử này. Fine-tune chỉ nên chạy sau khi người dùng xác nhận quyền sử dụng giọng và nghe kiểm tra WAV đã lọc.
