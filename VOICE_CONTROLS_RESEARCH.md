# Khảo sát công cụ giọng đọc và lồng tiếng (14/10/2026)

| Dịch vụ / tài liệu chính thức | Cách họ điều khiển | Áp dụng cho Vocalab |
| --- | --- | --- |
| [ElevenLabs Text to Speech](https://elevenlabs.io/docs/eleven-creative/playground/text-to-speech) và [voice settings API](https://elevenlabs.io/docs/api-reference/voices/settings/update) | Stability, similarity, style exaggeration và tốc độ tùy dòng model; không phải mọi model đều có đủ slider. | Có preset lời thoại và tốc độ từng đoạn; không giả lập slider stability/similarity của model khác. |
| [ElevenLabs Dubbing](https://elevenlabs.io/docs/eleven-creative/products/dubbing) / [Studio video voiceover](https://elevenlabs.io/docs/help-center/product/studio/studio/how-can-i-create-a-voiceover-for-my-video-using-studio) | Thay audio video, sửa lời thoại và tái tạo từng đoạn; timeline chỉnh vị trí/độ dài. | Vocalab xuất WAV + MP4 với lời thoại mới; chưa có timeline căn khớp lời nói theo người trong video hay dịch tự động. |
| [Azure AI Speech SSML](https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/speech-synthesis-markup-voice) | `express-as` cho giọng hỗ trợ style; `prosody` điều khiển pitch, contour, rate, volume. | Lấy ý tưởng điều khiển theo đoạn; xử lý cao độ/nhịp/âm lượng bằng FFmpeg Rubber Band. Không nhận SSML của Azure. |
| [Google Cloud TTS SSML](https://docs.cloud.google.com/text-to-speech/docs/ssml) | Thẻ `prosody` điều chỉnh pitch, speaking rate, volume; bao từ trong câu có thể tạo khoảng nghỉ ngoài ý muốn. | UI chia theo câu/dòng để người dùng quyết định điểm nghỉ. |
| [EmoV-DB](https://github.com/numediart/EmoV-DB) trên [OpenSLR 115](https://openslr.org/115/) | Giọng có nhãn cảm xúc thật: neutral, amused, angry, sleepy, disgusted. | Fine-tune một adapter tiếng Anh của cùng speaker Jenie, style Amused; không trộn speaker hoặc gắn nhãn sai thành MC. [Giấy phép](https://github.com/numediart/EmoV-DB/blob/master/LICENSE.md) chỉ cho nghiên cứu phi thương mại. |

## Giới hạn kỹ thuật

VieNeu-TTS v3 Turbo hiện đánh dấu tham số `style` là **deprecated/ignored** trong `src/vieneu/v3turbo.py`; phong cách chủ yếu đến từ mẫu giọng và dữ liệu speaker. Các preset MC, thuyết trình, kể chuyện của Vocalab là quy tắc về cao độ, tốc độ, âm lượng, ngắt nghỉ trên audio đã tổng hợp, không phải bộ điều khiển biểu cảm được huấn luyện riêng. `Lên giọng`/`xuống giọng` thay đổi cao độ phần cuối đoạn bằng Rubber Band; mức biểu cảm tự nhiên cần đánh giá bằng bài nghe người thật. Giọng EmoV-DB Amused bổ sung một màu giọng biểu cảm tiếng Anh, nhưng không làm các giọng khác tự động biết diễn cảm theo nhãn.

Dữ liệu ESD của [HLT Singapore](https://github.com/HLTSingapore/Emotional-Speech-Data) đòi gửi thỏa thuận cấp phép và chỉ cho nghiên cứu; chưa đưa vào pipeline. Dữ liệu VIVOS hiện có transcript và speaker ID nhưng không có nhãn MC/người kể chuyện; thêm speaker giúp đa dạng giọng tiếng Việt chứ không chứng minh học được vai diễn này.
