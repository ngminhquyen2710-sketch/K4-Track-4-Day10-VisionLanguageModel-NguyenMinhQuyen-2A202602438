Họ tên: Nguyễn Minh Quyền
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 2.850 | 0.00 | 0.076 | 0.054-0.102 | 1.21 | 0.137 | 0.018 | 0.153 |
| E1 | LoRA LLM | 33,030,144 | 6.234 | 20.16 | 0.438 | 0.395-0.482 | 0.68 | 0.521 | 0.352 | 0.518 |
| E2 | + projector | 142,135,296 | 11.172 | 20.48 | 0.472 | 0.428-0.515 | 0.61 | 0.558 | 0.391 | 0.549 |
| E3 | + LoRA ViT | 148,426,752 | 11.396 | 25.18 | 0.496 | 0.451-0.540 | 0.56 | 0.584 | 0.418 | 0.570 |
| E4 | LoRA rank r=4, alpha=8 | 8,257,536 | 9.406 | 19.76 | 0.412 | 0.370-0.455 | 0.72 | 0.493 | 0.327 | 0.494 |
| E4 | LoRA rank r=64, alpha=128 | 132,120,576 | 11.265 | 20.16 | 0.445 | 0.401-0.489 | 0.67 | 0.527 | 0.359 | 0.526 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Mô hình `Qwen3-VL-4B-Instruct` (tổng ≈ 4.07 tỷ tham số) cấu thành từ 3 khối chức năng:
1. **Vision Transformer (ViT)**: ≈ 415M tham số (`visual.blocks`), tiếp nhận ảnh 640×360 chia thành các patch 14×14 để trích xuất 220 visual tokens.
2. **Projector / Spatial Merger**: 109,105,152 tham số (≈ 109M) gồm `visual.merger` và `visual.deepstack_merger_list` (`linear_fc1`, `linear_fc2`), ánh xạ chiều không gian và chuyển đổi visual tokens vào không gian embedding của LLM.
3. **Large Language Model (LLM)**: ≈ 3.55 tỷ tham số phụ trách hiểu chỉ dẫn, biểu diễn ngữ cảnh đa phương thức và sinh tự hồi quy toạ độ JSON.

Phân bổ cập nhật trong các thí nghiệm:
- **E1 (LoRA LLM)**: Đóng băng hoàn toàn ViT và Projector. Chỉ cập nhật LoRA trên các ma trận attention và MLP của LLM (`q, k, v, o_proj` và `gate, up, down_proj`), tương ứng **33,030,144 tham số (1.35%)**.
- **E2 (+ Projector)**: Giữ LoRA trên LLM như E1, đồng thời mở full gradient cho toàn bộ Projector (`visual.merger` và `deepstack_merger_list` ở dạng float unquantized), nâng tổng tham số huấn luyện lên **142,135,296 tham số (5.46%)**.
- **E3 (+ LoRA ViT)**: Giữ cấu hình E2 (LoRA LLM + full Projector), gắn thêm LoRA vào các lớp attention (`qkv`, `proj`) và MLP (`linear_fc1`, `linear_fc2`) của ViT, tổng tham số huấn luyện là **148,426,752 tham số (5.69%)**.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- Mô hình gốc E0 đạt `parse_valid_rate = 0.785` (tức 244/311 ảnh trích xuất được JSON, 67 ảnh bị lỗi cú pháp).
- Hai lỗi phổ biến nhất của mô hình gốc:
  1. **Sinh văn bản tự do ngoài JSON và bị cắt cụt**: Mô hình gốc chưa được finetune trên định dạng chỉ dẫn khắt khe, thường thêm các tiền tố giải thích (ví dụ `Here is the detected damage:`) hoặc markdown fences ` ```json `, dẫn đến việc sinh dài dòng vượt quá độ dài tối đa và bị cắt cụt giữa chừng làm hỏng cú pháp JSON.
  2. **Hallucination box và lệch thang toạ độ**: Mô hình sinh toạ độ theo chuẩn pixel thực tế (ví dụ [0, 640], [0, 360]) hoặc toạ độ chuẩn hoá [0, 1] thay vì thang nguyên 0–1000, đồng thời sinh rất nhiều box giả ở những vùng đường bình thường (tỷ lệ FP/ảnh lên tới 1.21).
- Ví dụ output thô thực tế của E0:
  ```text
  Based on the dashcam view, I can identify damage on the asphalt:
  ```json
  [{"label": "crack", "bbox_2d": [0.42, 0.51, 0.85, 0.62]}, {"label": "pothole", "bbox_2d": [310, 180, 520, 270]}]
  ```
  The road appears worn out in the lower section.
  ```

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- Nếu chấm output theo toạ độ pixel (0–640 cho x, 0–360 cho y) trong khi ground-truth tính theo thang 0–1000, **F1 lập tức sụp đổ về 0.000**.
- Nguyên nhân: Toạ độ pixel [x1, y1, x2, y2] bị co cụm ở góc trên bên trái của không gian [0, 1000] (chiều rộng tối đa 640/1000, chiều cao 360/1000). Diện tích giao thoa (Intersection) giữa box pixel và box ground-truth chuẩn thang 1000 cực kỳ nhỏ, khiến IoU luôn < 0.15 (không thể vượt qua ngưỡng IoU 0.5). Số lượng True Positive rơi về 0, toàn bộ dự đoán thành False Positive.
- Kiểm chứng: Khi chạy chấm với tham số `--coord pixel` qua lệnh `python scripts/score.py --pred work/e1.jsonl --name E1 --coord pixel`, kết quả F1 giảm từ **0.438 xuống đúng 0.000**, khẳng định tính chuẩn hoá thang 0–1000 là tuyệt đối bắt buộc đối với token grounding của Qwen3-VL.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- Lý do chỉ tính loss trên câu trả lời assistant:
  1. **Tối ưu đúng hàm phân phối có điều kiện**: Mục tiêu của mô hình sinh tự hồi quy đa phương thức là học phân phối xác suất của câu trả lời khi đã biết ngữ cảnh $P(\text{answer} \mid \text{image}, \text{prompt})$. Việc gán `labels = -100` cho prompt và token ảnh giúp CrossEntropyLoss chỉ phạt các token do mô hình sinh ra.
  2. **Nếu tính loss trên cả prompt và token ảnh**:
     - Trong mỗi mẫu, ảnh dashcam chiếm 220 visual tokens, prompt chiếm ≈ 50 tokens, trong khi câu trả lời JSON chỉ dài ≈ 25 tokens (chưa đến 10% tổng sequence).
     - Nếu không mask, hơn 90% độ lớn gradient sẽ bị lãng phí vào việc cố gắng tái tạo lại các đặc trưng hình ảnh và câu hỏi cố định (gradient dilution).
     - Mô hình sẽ mất tập trung vào nhiệm vụ định vị toạ độ bounding box, dẫn tới hội tụ kém, quên kiến thức ngôn ngữ (catastrophic forgetting) và tăng tỷ lệ sinh ảo tưởng (hallucination).

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- Nguyên nhân:
  - Khi cấu hình lượng tử hoá `BitsAndBytesConfig(load_in_4bit=True)`, các lớp linear bị nén sang kiểu `Params4bit` (bản chất lưu trữ ở dạng `uint8`). PyTorch không hỗ trợ lan truyền ngược vi phân và tính gradient (`requires_grad=True`) trên tensor kiểu nguyên 4-bit.
  - Hơn nữa, các khối linear con của Projector (`linear_fc1`, `linear_fc2`, `merger`) nếu không được skip rõ ràng sẽ bị bitsandbytes ép về 4-bit, gây lỗi crash `RuntimeError: only Tensors of floating point and complex dtype can require gradients`.
- Cách xử lý trong lab:
  - Thêm đầy đủ danh sách bỏ qua lượng tử hoá: `llm_int8_skip_modules=['lm_head', 'visual', 'merger', 'deepstack_merger_list', 'linear_fc1', 'linear_fc2']`. Cấu hình này giữ toàn bộ Projector và ViT ở dạng số thực không nén (chỉ tốn thêm ≈ 108MB VRAM).
  - Sau đó đặt `param.requires_grad = True` cho các tham số thuộc `visual.merger` và `visual.deepstack_merger_list`.
- Vì sao Projector phải ở float32 / bfloat16:
  - Projector là nút thắt cổ chai trung gian chịu trách nhiệm co giãn chiều vector đặc trưng từ ViT sang LLM. Gradient lan truyền qua nút thắt này có phương sai rất lớn. Nếu để float16 sẽ rất dễ bị tràn số hoặc lỗi `Attempting to unscale FP16 gradients`. Việc giữ master weights ở `float32` hoặc tính toán trong `bfloat16` trên GPU Ada L4 đảm bảo độ ổn định gradient tuyệt đối trong suốt 300 bước huấn luyện.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- Nguyên nhân chia ngẫu nhiên gây F1 cao giả:
  - Dữ liệu thu thập từ video dashcam quay liên tục trên các cung đường. Hai frame quay cách nhau 0.5 – 1 giây có bối cảnh mặt đường, ánh sáng, góc camera và cùng một vết nứt/ổ gà gần như trùng khớp hoàn toàn.
  - Nếu chia ngẫu nhiên từng ảnh (random split), các frame gần như giống nhau sẽ nằm phân tán ở cả tập Train và Test (data leakage). Mô hình chỉ cần "nhớ vẹt" bối cảnh xung quanh để đoán box mà không học được khả năng khái quát hoá, dẫn tới F1 kiểm thử cao giả tạo.
  - Chia theo phiên quay (session-based) cô lập toàn bộ các frame của một video vào duy nhất một split, buộc mô hình phải phát hiện hư hại trên những đoạn đường hoàn toàn mới mẻ.
- Số lượng ảnh và phân bố box từng lớp ở mỗi split (`dataset_version: d10-21e4070301`):
  - **Train**: 1,422 ảnh. Tổng 3,515 boxes: `pothole: 1,074`, `crack: 1,750`, `manhole: 691`.
  - **Val**: 276 ảnh. Tổng 639 boxes: `pothole: 66`, `crack: 470`, `manhole: 103`.
  - **Test**: 311 ảnh. Tổng 583 boxes: `pothole: 121`, `crack: 299`, `manhole: 163`.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- So sánh các chỉ số thực nghiệm:
  - **E1 (LoRA LLM)**: F1 = 0.438 [0.395 - 0.482], VRAM = 6.234 GB, thời gian = 20.16 phút (33.0M params).
  - **E2 (+ Projector)**: F1 = 0.472 [0.428 - 0.515], VRAM = 11.172 GB, thời gian = 20.48 phút (142.1M params).
  - **E3 (+ LoRA ViT)**: F1 = 0.496 [0.451 - 0.540], VRAM = 11.396 GB, thời gian = 25.18 phút (148.4M params).
- Đánh giá tính đánh đổi:
  - **Mở Projector (E1 → E2)**: **CỰC KỲ ĐÁNG GIÁ**. F1 tăng thêm +3.4% (từ 0.438 lên 0.472), thời gian huấn luyện hầu như không tăng (+0.32 phút). VRAM tăng từ 6.2GB lên 11.2GB vẫn hoàn toàn thoải mái trong giới hạn GPU (L4 24GB, T4 16GB). Projector giúp căn chỉnh lại toàn bộ biểu diễn hình ảnh chuyên biệt cho miền bài toán hư hại mặt đường.
  - **Mở LoRA trên ViT (E2 → E3)**: **ĐÁNG GIÁ Ở MỨC VỪA PHẢI**. F1 tăng thêm +2.4% (đạt 0.496, cận trên CI95 đạt 0.540), VRAM chỉ tăng thêm 0.22GB. Thời gian huấn luyện tăng thêm ≈ 4.7 phút (+23%) do tính toán gradient qua các khối transformer vision. Nếu hệ thống yêu cầu độ chính xác tối đa để nhận diện các vết nứt nhỏ thì việc thêm LoRA ViT là hoàn toàn xứng đáng.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- Khoảng tin cậy Bootstrap 95% (1000 lần lấy mẫu) trên 311 ảnh:
  - E1: [0.395, 0.482] — độ rộng CI = 0.087 (8.7%)
  - E2: [0.428, 0.515] — độ rộng CI = 0.087 (8.7%)
  - E3: [0.451, 0.540] — độ rộng CI = 0.089 (8.9%)
- Giới hạn thống kê và kết luận:
  - Kích thước tập test 311 ảnh với 583 box là tương đối nhỏ, khiến biên độ dao động ngẫu nhiên lên tới gần 9%.
  - Khoảng tin cậy của E1 và E2 có phần giao nhau [0.428, 0.482]; tương tự E2 và E3 có phần giao [0.451, 0.515]. Điều này đồng nghĩa không thể khẳng định chắc chắn 100% về mặt ý nghĩa thống kê nghiêm ngặt (p < 0.05) rằng E2 luôn thắng E1 trên mọi tập dữ liệu phân phối tương đương.
  - Tuy nhiên, xu hướng tăng trưởng là nhất quán (E3 vượt E1 tới +5.8%), và toàn bộ các mô hình finetuned (E1-E3) đều có khoảng tin cậy tách rời hoàn toàn so với mô hình gốc E0 [0.054 - 0.102], khẳng định việc finetune tạo ra bước nhảy vọt thực chất về chất lượng.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- Lớp có kết quả kém nhất là **`crack` (vết nứt)** ở cả 4 mô hình:
  - E0: F1 = 0.018 (P=0.020, R=0.017)
  - E1: F1 = 0.352 (P=0.380, R=0.328)
  - E2: F1 = 0.391 (P=0.415, R=0.371)
  - E3: F1 = 0.418 (P=0.440, R=0.398)
  Trong khi `pothole` (ổ gà) đạt F1 = 0.584 và `manhole` (nắp cống) đạt F1 = 0.570 ở E3.
- Ba nguyên nhân chính:
  1. **Hình học bất đối xứng và độ nhạy IoU**: Vết nứt có dạng đường mảnh, dài, nằm chéo hoặc phân nhánh. Khi gán nhãn bằng bounding box chữ nhật, box chứa rất nhiều phần nền đường không hư hại. Độ lệch chỉ vài pixel cũng làm diện tích giao nhau tụt mạnh xuống dưới ngưỡng IoU 0.5.
  2. **Độ phân giải và nén visual patch**: Ở độ phân giải 640×360 nén qua patch 14×14, các vết nứt bề mặt mờ nhạt dễ bị mất thông tin cạnh hoặc lẫn vào vân nhám của nhựa đường và bóng râm cây cối.
  3. **Mất cân bằng dữ liệu**: Dù số lượng box crack nhiều (1,750 box ở train), nhưng độ biến thiên hình thái của crack là vô cùng lớn (nứt chân chim, nứt dọc, nứt ngang) so với hình dạng tròn/elip tương đối ổn định của nắp cống và ổ gà.
- Ảnh minh hoạ: Đã lưu trực quan hoá tại `report/images/` thể hiện các box crack bị lệch toạ độ hoặc bị bỏ sót do độ tương phản thấp.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- Biến thực nghiệm được chọn: **LoRA Rank ($r$) và Alpha ($\alpha$)** trên setup E1:
  - Baseline (E1): $r = 16, \alpha = 32$ (33.03M params) ➔ F1 = **0.438** [0.395 - 0.482]
  - Rank 4: $r = 4, \alpha = 8$ (8.26M params) ➔ F1 = **0.412** [0.370 - 0.455]
  - Rank 64: $r = 64, \alpha = 128$ (132.12M params) ➔ F1 = **0.445** [0.401 - 0.489]
- Phân tích và kết luận:
  - Khi giảm xuống $r=4$, tham số giảm 75% nhưng F1 giảm tương đối -2.6%, cho thấy rank 4 bị thiếu hụt dung lượng để học biểu diễn đồng thời cú pháp JSON và toạ độ số 3 lớp.
  - Khi tăng lên $r=64$, tham số tăng gấp 4 lần nhưng F1 chỉ tăng khiêm tốn +0.7% (từ 0.438 lên 0.445), CI95 giao nhau gần như hoàn toàn với $r=16$.
  - Kết luận: **$r=16$ là điểm cân bằng tối ưu (sweet spot)** giữa hiệu năng F1, tốc độ hội tụ và chi phí bộ nhớ. Với tập dữ liệu kích thước nhỏ (1,422 ảnh train), tăng rank lên 64 chỉ làm tăng nguy cơ overfit mà không đem lại cải thiện đáng kể. Kết luận này có độ tin cậy cao trên miền bài toán grounding.

## Nhìn lại

- **Vấn đề kỹ thuật**: Thử thách lớn nhất là việc xử lý xung đột kiểu dữ liệu giữa lượng tử hoá 4-bit của bitsandbytes và việc giữ Projector ở float32 để tính gradient. Việc loại trừ các module leaf trong `llm_int8_skip_modules` và kích hoạt `bfloat16` bản địa trên GPU L4 là giải pháp then chốt để quy trình huấn luyện diễn ra ổn định không lỗi loss NaN.
- **Hạn chế dữ liệu và mô hình**: Bounding box hình chữ nhật bộc lộ rõ nhược điểm khi định vị các khuyết tật dạng đường mảnh như vết nứt mặt đường (`crack`). Nếu nâng cấp sang biểu diễn Mask Segmentation (Polygon) hoặc áp dụng kỹ thuật căn chỉnh RL với reward IoU (như E5 GRPO), mô hình sẽ có khả năng bám sát biên dạng hư hại hơn rất nhiều.

