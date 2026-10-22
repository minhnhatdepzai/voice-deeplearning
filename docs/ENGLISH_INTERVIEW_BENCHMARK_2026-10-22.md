# Benchmark giọng Anh từ bản phỏng vấn hai người

Ngày chạy: 2026-10-22

Thiết bị train: NVIDIA RTX 5060 Ti 16 GB

Model nền: VieNeu-TTS v3 Turbo, upstream commit `04044e1b8f822f946bafb2c88b716459f4164152`

## Phạm vi và dữ liệu

Không tải hoặc clone giọng người nổi tiếng có thể nhận dạng. Benchmark dùng hai người nói LibriTTS có giấy phép CC BY 4.0 để kiểm tra đúng bài toán kỹ thuật: người phỏng vấn xen kẽ khách mời, có đoạn chồng lời và tiếng đường phố tổng hợp.

- Khách mời: speaker `8297`, 80 lượt, 544,49 giây.
- Người phỏng vấn: speaker `3752`, 20 lượt, 101,73 giây.
- Toàn bản phỏng vấn: 669,54 giây, tức 11 phút 10 giây.
- Ba điều kiện: sạch, nhiễu vừa SNR 18 dB và nhiễu đường phố SNR 10 dB.
- Test: 22 câu, 380 từ, hoàn toàn tách khỏi dữ liệu train.

Script tái tạo benchmark là `training/benchmark_interview.py`. Manifest và WAV cục bộ nằm trong `data/validation/english_interview_benchmark/`.

## Tự tìm và tách người nói chính

Ngưỡng gộp cụm cũ `0,86` đã gộp nhầm hai giọng nam tương đối giống nhau. Ngưỡng được tăng lên `0,93` và có regression test. Sau sửa, cả ba điều kiện đều tìm được hai cụm; cụm chính khớp khách mời theo encoder speaker:

| Điều kiện | Tỷ lệ cụm chính | Cosine cụm chính → khách mời | Cosine → người phỏng vấn | Biên nhận dạng |
| --- | ---: | ---: | ---: | ---: |
| Sạch | 73,80% | 0,9565 | 0,6488 | 0,3076 |
| SNR 18 dB | 77,80% | 0,8898 | 0,6043 | 0,2856 |
| Đường phố SNR 10 dB | 71,12% | 0,8376 | 0,5860 | 0,2516 |

Bản khó nhất được chạy hết bằng MossFormer2 + encoder VieNeu + FRCRN: 61/61 cửa sổ, giữ nguyên timeline 669,54 giây, mất 3 phút 12 giây, RAM đỉnh khoảng 2,04 GB và VRAM khoảng 11,3 GB.

## Dữ liệu và hai lần train mới

Lần chuẩn bị đầu tiên bị chặn đúng bởi kiểm tra codec do nhiều clip có đuôi im lặng. Pipeline được sửa để cắt đuôi im lặng và căn theo frame codec 80 ms; không bỏ qua cổng kiểm tra.

| Ứng viên | Train | Dữ liệu | Cấu hình | VRAM đỉnh |
| --- | ---: | --- | --- | ---: |
| `interview_street_v3` | 80 clip / 8,87 phút | Chỉ lời đã tách từ SNR 10 | 100 bước, rank 32, target all, LR 1e-4 | 15.289 MiB |
| `interview_mix_v3` | 120 clip / 13,02 phút | 80 clip đã tách + 40 clip sạch | 100 bước, rank 32, target all, LR 1e-4 | 15.276 MiB |

Cosine giữa 80 clip đã tách và audio gốc có trung bình `0,8657`, thấp nhất `0,7844`. Pipeline export giờ chọn mẫu enrollment sạch/điểm cao nhất theo `reference_file` hoặc `reference_score`; nó không còn mặc định lấy dòng train đầu tiên.

## Kết quả công bằng trên cùng 22 câu

| Model | WER ↓ | CER ↓ | Câu không lỗi ASR | Cosine AI → WAV gốc ↑ | Tỷ lệ thời lượng AI/gốc | Pause nội bộ ≥350 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline `libritts_en_8297_v1` | **3,16%** | **0,75%** | **15/22** | 0,8553 | 0,972 | 23 |
| Train chỉ phỏng vấn nhiễu | 4,47% | 1,91% | 13/22 | 0,7615 | 0,953 | **20** |
| Phỏng vấn + dữ liệu sạch | 4,21% | **0,75%** | 14/22 | **0,8603** | **1,001** | 22 |

Pause dùng frame RMS 20 ms và chỉ là proxy cho độ trôi chảy. Không model nào clipping. Model trộn giữ thời lượng và âm sắc tốt hơn một chút, nhưng baseline ít sai từ nhất: 12 lỗi so với 16 và 17 lỗi. Vì ưu tiên phát âm rõ, ít vấp, baseline hiện tại được giữ nguyên; hai ứng viên không được đăng ký hoặc thay model đang dùng.

## Giới hạn kết luận

WER/CER đo bằng faster-whisper và cosine đo bằng encoder VieNeu. Chúng giúp phát hiện lỗi phát âm, lệch giọng và ngắt bất thường, nhưng không chứng minh âm thanh “giống người thật”. Cần bài nghe mù MOS/ABX với nhiều người nghe để đo tự nhiên. Kết quả chi tiết được lưu ở `data/validation/english_interview_benchmark/model_comparison.json`.
