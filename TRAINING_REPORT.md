# Báo cáo huấn luyện và đánh giá công khai

Ngày ghi nhận: 14/10/2026. Model nền: VieNeu-TTS v3 Turbo; recipe upstream tại commit `bd0f88eb8de7d560753a6172ba9c868f309a101c`. Mỗi run dùng một speaker. Các con số bên dưới lấy từ kết quả cục bộ đã lưu và được tóm tắt, không công bố audio nguồn hay transcript cá nhân.

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
