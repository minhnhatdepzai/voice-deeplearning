# Báo cáo huấn luyện và đánh giá công khai

Ngày cập nhật: 22/10/2026. Model nền: VieNeu-TTS v3 Turbo. Các adapter công khai trong bảng được tạo bằng recipe upstream tại commit `bd0f88eb8de7d560753a6172ba9c868f309a101c`; hai ứng viên cục bộ mới dùng commit `04044e1b8f822f946bafb2c88b716459f4164152`. Mỗi run dùng một speaker. Các con số bên dưới lấy từ kết quả cục bộ đã lưu và được tóm tắt, không công bố audio nguồn hay transcript cá nhân.

## Cập nhật pipeline 22/10/2026

- Bổ sung tách theo giọng tham chiếu bằng MossFormer2_SS_16K, chọn stem với encoder VieNeu và giảm nhiễu tùy chọn bằng FRCRN_SE_16K. Bản WAV sau lọc có thể nghe/tải; video xuất MP4 với hình gốc. Dữ liệu train mới đi qua tách giọng trước ASR/LoRA.
- Thử tích hợp trên bản trộn hai mẫu giọng 6,4 giây: tương quan Pearson giữa giọng mục tiêu gốc với audio hỗn hợp là **0,918**, với WAV mục tiêu sau xử lý là **0,946**; tương quan giữa giọng mục tiêu và WAV phần dư là **−0,057**. Đây là **một bản trộn thử**, không phải benchmark tổng quát về độ tách, chất lượng nghe hoặc độ giống người nói.
- Kiểm tra một nguồn video dài 194 giây: với mẫu tham chiếu hiện có, lần chạy đầu chấp nhận 6/18 chunk. Chấm lại hai stem đã lưu bằng ngưỡng có xét tương đồng file hỗn hợp cho **13/18 chunk**. Số chunk chỉ là chỉ báo chọn giọng, không chứng minh từng âm tiết sạch; video vẫn ngắn hơn mức tối thiểu 5 phút lời nói sạch cho quy trình train mới.
- Thư viện ban đầu có 18 adapter fine-tune và 4 mục giọng upload còn file nguồn; 3 mục upload trỏ đến cùng nội dung video, mục thứ tư là nguồn khác. Không có dữ liệu một người nói mới để train lại tất cả giọng. Hai adapter ứng viên được train lại từ **chính dữ liệu cũ** để so recipe/cấu hình; chúng không thắng benchmark 19 câu nên **không thay adapter đang dùng và không được đăng ký vào thư viện**. Chế độ TTS `Tự nhiên` giờ giữ nguyên PCM model khi không chỉnh cao độ/nhịp/âm lượng.
- Worker mới yêu cầu ít nhất 5 phút và 50 clip sau lọc. Precheck giữ ít nhất 5 câu test riêng; nếu WER AI kém bản gốc quá 20 điểm phần trăm, dừng trước khi train cuối/thay adapter. WER là proxy từ Whisper, không đo độ tự nhiên. Dữ liệu bổ sung có thể tải thêm cho từng giọng đã lưu; cần bản thu mới, đúng người, ít nhiễu và kiểm tra transcript.

## Speech to speech và vòng lặp thử nghiệm 22/10/2026

- Trang **Speech to speech** dùng VAD trong trình duyệt, 350 ms tiền đệm, chốt câu sau khoảng 680 ms im lặng, gửi WAV tới FastAPI, nhận dạng bằng faster-whisper rồi tạo lại bằng giọng VieNeu đã chọn. Whisper và một VieNeu worker được giữ nóng; khi đổi sang một model fine-tune khác, worker nạp lại model tương ứng. API trả riêng thời gian decode/STT/TTS/tổng.
- Trên 19 câu VIVOSSPK15 test: `small` beam 1 đạt WER **13,58%**, inference **2,24 giây** cho cả 19 câu; `turbo` beam 3 đạt **4,32%**, **3,74 giây**; `turbo` beam 5 vẫn **4,32%**, **3,93 giây**. Mặc định vì vậy là `turbo` beam 3. Tải model từ cache mất khoảng 1,3–1,7 giây; VieNeu fine-tune nạp khoảng 4,8–7,0 giây. Khi giữ cả hai model, GPU dùng khoảng 3,2 GiB trên RTX 5060 Ti 16 GiB trong phép kiểm tra.
- Với adapter upload hiện tại `abc · Fine-tune toàn file`, 19 lượt HTTP sau warmup đạt WER đầu vào **4,94%**, WER của WAV đầu ra khi Whisper đọc lại **9,26%**, cosine so với mẫu giọng **0,5274** và latency xử lý trung bình **550 ms/lượt**. Auto-VAD thêm khoảng 680 ms sau âm cuối trước khi gửi. Đây là số đo pipeline cục bộ trên câu ngắn, không phải cam kết thời gian cho máy khác.
- Ba cấu hình sampling VieNeu được thử trên cùng 8 câu. `natural` đạt WER đầu ra **17,74%**, cosine **0,4911**; `clear` đạt **12,90%**, **0,4764**; `stable` đạt **9,68%**, **0,5042**. Pipeline tương tác mặc định `stable` (`temperature=0.35`, `top_k=15`, `top_p=0.85`) vì thắng trên bộ này. Kết quả sinh có tính ngẫu nhiên; cần bài nghe người thật trước khi kết luận tự nhiên hơn.
- Hậu xử lý 19 WAV với bốn mức cho thấy bản gốc WER **9,26%**, cosine **0,5274**, noise-floor proxy **−59,24 dBFS**; lọc nhẹ đạt **8,64%**, **0,5303**, **−61,39 dBFS**; lọc mạnh đạt **8,64%**, **0,5379**, **−62,29 dBFS**. Giao diện mặc định lọc nhẹ để giảm nguy cơ tạo artifact; người dùng có thể chọn mạnh hoặc tắt. Noise floor là proxy trên frame năng lượng thấp, chưa phải đánh giá cảm nhận.
- Hai adapter ứng viên mới dùng cùng dữ liệu upload cũ khoảng 2,75 phút: `upload_098_latest_r16_s120` (rank 16, target all, 120 bước, LR `1e-4`) và `upload_098_latest_r8_s180` (rank 8, target backbone, 180 bước, LR `5e-5`). Cả hai merge/reload thành công. Trên 19 câu ngoài dữ liệu upload, model đang dùng đạt WER/cosine **9,26% / 0,5274**; ứng viên rank 16 đạt **10,49% / 0,4767**; ứng viên rank 8 đạt **10,49% / 0,5077**. Vì model cũ thắng cả hai chỉ số trong lượt mở rộng, không ứng viên nào được promote. Kết quả này cũng cho thấy tăng bước train không thay thế được dữ liệu sạch mới.

## Benchmark phỏng vấn tiếng Anh 22/10/2026

Benchmark hai người nói LibriTTS CC BY 4.0 dài 11 phút 10 giây đã được tạo ở ba mức nhiễu. Sau khi sửa ngưỡng gộp cụm, cả ba mức đều tìm đúng hai cụm và cụm chính khớp speaker `8297`; ở bản đường phố SNR 10 dB, cosine với khách mời là **0,8376**, với người phỏng vấn là **0,5860**. Bản khó nhất được tách toàn bộ 61/61 cửa sổ.

Hai adapter mới được train 100 bước trên 8,87 phút audio đã tách và trên bản trộn 13,02 phút có thêm dữ liệu sạch. So công bằng trên cùng 22 câu/380 từ chưa train: baseline hiện tại đạt WER **3,16%**, ứng viên chỉ phỏng vấn **4,47%**, ứng viên trộn **4,21%**. Ứng viên trộn có cosine AI với WAV gốc **0,8603**, nhỉnh hơn baseline **0,8553**, nhưng có 16 lỗi từ thay vì 12. Vì mục tiêu phát âm rõ và ít vấp, baseline được giữ; hai ứng viên không được đăng ký. Xem [báo cáo đầy đủ](docs/ENGLISH_INTERVIEW_BENCHMARK_2026-10-22.md).

## Kết quả từng adapter

| Run | Corpus | Train/val/test | Phút train | Updates | WER gốc → AI | Cosine gốc → AI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [emov_en_jenie_amused_v1](https://github.com/minhnhatdepzai/voice-deeplearning/releases/download/v0.1.0-models/emov_en_jenie_amused_v1.tar.gz) | emov · jenie_amused | 182/21/19 | 14.7 | 90 | 4.3% → 6.5% | 0.690 → 0.655 |
| [libritts_en_1462_v1](https://github.com/minhnhatdepzai/voice-deeplearning/releases/download/v0.1.0-models/libritts_en_1462_v1.tar.gz) | libritts · 1462 | 227/26/30 | 14.7 | 90 | 4.1% → 5.9% | 0.749 → 0.795 |
| [libritts_en_3752_v1](https://github.com/minhnhatdepzai/voice-deeplearning/releases/download/v0.1.0-models/libritts_en_3752_v1.tar.gz) | libritts · 3752 | 197/34/23 | 13.8 | 90 | 8.9% → 10.5% | 0.804 → 0.872 |
| [libritts_en_6313_v1](https://github.com/minhnhatdepzai/voice-deeplearning/releases/download/v0.1.0-models/libritts_en_6313_v1.tar.gz) | libritts · 6313 | 218/25/26 | 13.7 | 90 | 5.7% → 7.9% | 0.818 → 0.802 |
| [libritts_en_8297_v1](https://github.com/minhnhatdepzai/voice-deeplearning/releases/download/v0.1.0-models/libritts_en_8297_v1.tar.gz) | libritts · 8297 | 193/27/32 | 18.3 | 90 | 2.5% → 3.4% | 0.861 → 0.875 |
| [ljspeech_en_lj_extended_b14](https://github.com/minhnhatdepzai/voice-deeplearning/releases/download/v0.1.0-models/ljspeech_en_lj_extended_b14.tar.gz) | ljspeech · LJ | 4005/516/473 | 434.6 | 600 | 2.4% → 3.5% | 0.879 → 0.886 |
| [vivos_VIVOSSPK15_20261014062505_142b](https://github.com/minhnhatdepzai/voice-deeplearning/releases/download/v0.1.0-models/vivos_VIVOSSPK15_20261014062505_142b.tar.gz) | vivos · VIVOSSPK15 | 251/29/19 | 14.1 | 90 | 13.6% → 11.7% | 0.735 → 0.741 |
| [vivos_vi_spk13_v2](https://github.com/minhnhatdepzai/voice-deeplearning/releases/download/v0.1.0-models/vivos_vi_spk13_v2.tar.gz) | vivos · VIVOSSPK13 | 242/29/29 | 15.8 | 30 | 27.8% → 32.3% | 0.681 → 0.621 |
| [vivos_vi_spk14_v1](https://github.com/minhnhatdepzai/voice-deeplearning/releases/download/v0.1.0-models/vivos_vi_spk14_v1.tar.gz) | vivos · VIVOSSPK14 | 238/30/32 | 13.8 | 90 | 23.5% → 24.6% | 0.686 → 0.683 |
| [vivos_vi_spk16_v1](https://github.com/minhnhatdepzai/voice-deeplearning/releases/download/v0.1.0-models/vivos_vi_spk16_v1.tar.gz) | vivos · VIVOSSPK16 | 245/23/32 | 10.9 | 90 | 21.1% → 28.2% | 0.736 → 0.734 |
| [vivos_vi_spk17_extended_b14](https://github.com/minhnhatdepzai/voice-deeplearning/releases/download/v0.1.0-models/vivos_vi_spk17_extended_b14.tar.gz) | vivos · VIVOSSPK17 | 243/32/25 | 10.1 | 250 | 30.5% → 45.8% | 0.726 → 0.667 |
| [vivos_vi_spk17_v1](https://github.com/minhnhatdepzai/voice-deeplearning/releases/download/v0.1.0-models/vivos_vi_spk17_v1.tar.gz) | vivos · VIVOSSPK17 | 243/32/25 | 10.1 | 90 | 28.7% → 31.2% | 0.720 → 0.654 |
| [vivos_vi_spk18_extended_b14](https://github.com/minhnhatdepzai/voice-deeplearning/releases/download/v0.1.0-models/vivos_vi_spk18_extended_b14.tar.gz) | vivos · VIVOSSPK18 | 244/36/20 | 9.9 | 250 | 20.5% → 27.9% | 0.704 → 0.696 |
| [vivos_vi_spk18_v1](https://github.com/minhnhatdepzai/voice-deeplearning/releases/download/v0.1.0-models/vivos_vi_spk18_v1.tar.gz) | vivos · VIVOSSPK18 | 244/36/20 | 9.9 | 90 | 25.8% → 33.0% | 0.698 → 0.693 |
| [vivos_vi_spk19_v1](https://github.com/minhnhatdepzai/voice-deeplearning/releases/download/v0.1.0-models/vivos_vi_spk19_v1.tar.gz) | vivos · VIVOSSPK19 | 246/30/24 | 13.2 | 90 | 29.0% → 46.4% | 0.680 → 0.686 |

Mỗi run trong bảng có `status=evaluated`, adapter SHA-256 khớp và phép nạp lại/merge adapter trong tiến trình mới đã được ghi nhận. [Catalog JSON](models/catalog.json) chứa cấu hình batch, rank, learning rate, số test item và checksum asset. Các WAV `preview.wav` trong Release là **audio do model sinh**, không phải bản gốc của dataset.

## Cách đo và giới hạn

- **WER/CER:** faster-whisper `small` đọc lại WAV gốc và WAV AI của cùng câu test, rồi so với transcript. Giá trị bao gồm cả lỗi STT. Các run có tập câu/độ khó khác nhau, vì vậy không so WER giữa speaker như một leaderboard.
- **Cosine speaker:** encoder speaker của VieNeu tính độ gần embedding của WAV gốc/WAV AI so với mẫu enroll. Đây là proxy âm sắc; cosine cao không chứng minh danh tính hay cảm xúc. Mẫu phát hành chỉ để nghe thử.
- **Tập train/test:** corpus công khai dùng clip test riêng không đưa vào Parquet train. Recipe upstream có thể lấy một phần nhỏ từ nhóm train để theo dõi loss; validation riêng trong manifest không tự động trở thành phép chọn checkpoint độc lập.
- **Giọng file người dùng:** model cuối dùng toàn bộ clip lời nói được nhận diện, không có held-out test. Những WER từ run precheck tách câu không phải metric của model cuối. File upload và adapter của file upload được giữ cục bộ, không có trong Release công khai.
- **Thử nghe:** chưa có MOS/ABX bởi hội đồng người nghe. Không tuyên bố mức giống giọng hoặc chất lượng MC đã được chứng minh bằng các chỉ số trên.

## Benchmark STT tiếng Việt độc lập

[FPT Open Speech](https://data.mendeley.com/datasets/k9sxg2twv4/4) được dùng để đo faster-whisper `small`, không dùng train TTS do mirror thiếu speaker ID đáng tin. Chọn cố định 30 câu từ Set001 và 30 câu từ Set002 theo hash tên file; WER Set001 **29,5%**, Set002 **40,1%**, tổng **35,3%**. Whisper chưa được fine-tune trong dự án. Kết quả là baseline của STT hiện tại trên mẫu 60 câu, không phải accuracy tổng quát tiếng Việt.

## Nguồn và giấy phép

- [VIVOS trên Zenodo](https://zenodo.org/records/7068130): CC BY-NC-SA 4.0, nghiên cứu phi thương mại; [mirror Kaggle](https://www.kaggle.com/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr) có metadata khác, áp điều kiện bản gốc.
- [LibriTTS](https://www.openslr.org/60/): CC BY 4.0, ghi công.
- [LJ Speech](https://keithito.com/LJ-Speech-Dataset/): public domain tại Mỹ; xem quy định nơi sử dụng.
- [EmoV-DB](https://github.com/numediart/EmoV-DB/blob/master/LICENSE.md): chỉ nghiên cứu, giảng dạy và thử nghiệm cá nhân phi thương mại. Adapter Jenie Amused thuộc phạm vi này.

Dùng adapter và tiếng nói mô phỏng theo quyền của dữ liệu/model/người nói. Repo công khai không bao gồm dữ liệu âm thanh nguồn hoặc giọng người dùng tải lên.
