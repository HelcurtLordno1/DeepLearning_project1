# Báo cáo benchmark M1–M2–M3: phân đoạn whole tumor BraTS 2015

**Ngày tổng hợp:** 29/09/2026. **Trạng thái:** đã có kết quả full train/validation/test cho **seed 42** của cả ba mô hình. Đây là so sánh nội bộ trên cohort cố định, **không** phải điểm BraTS challenge chính thức hay đánh giá dùng trong lâm sàng. Seed 123 và 2026 chưa có kết quả, vì vậy mọi nhận xét về thứ hạng chỉ áp dụng cho lần chạy seed 42 đang lưu.

## 1. Câu hỏi và bằng chứng

Bài toán là dự đoán mask *whole tumor* (WT) nhị phân từ FLAIR MRI 2D. Mục tiêu benchmark: xem ba **hệ thống hoàn chỉnh** khác nhau về chất lượng phân đoạn, loại lỗi và chi phí huấn luyện trên cùng dữ liệu. Đặc tả và mã nguồn ở [Project_structure.md](Project_structure.md), [M1](M1/M1.py), [M2](M2/M2.py), [M3](M3/M3.py). Các số liệu dưới đây được tính lại từ `config.json`, `metrics_val.csv`, `metrics_test.csv` và `history.csv` trong `runs/m1/42/`, `runs/m2/42/`, `runs/m3/42/` trên máy thực hiện. Ba thư mục `runs/` là artifact **chỉ lưu cục bộ**, được `.gitignore` loại khỏi Git; báo cáo này giữ số liệu tổng hợp, không đưa MRI, checkpoint hoặc điểm từng bệnh nhân lên repository.

Kiểm tra tính so sánh: cả ba CSV test đều có **15 `case_id` giống nhau**, mỗi CSV validation có 15 ca, lịch sử train có đúng 26/24/15 epoch tương ứng M1/M2/M3. `smoke=false` trong cả ba `config.json`. Mọi mô hình dùng cùng 100 ca (80 HGG, 20 LGG), chia theo bệnh nhân thành 70 train, 15 validation (12 HGG, 3 LGG), 15 test (12 HGG, 3 LGG); mỗi ca lấy 32 lát từ 20–80% độ sâu **ảnh** và resize 128×128. Đầu vào chỉ dùng FLAIR; mask WT là OT thuộc `{1,2,3,4}`. Chuẩn hóa FLAIR theo median/IQR voxel khác 0, lặp ba kênh và chuẩn hóa ImageNet; chỉ train lật ngang đồng bộ ảnh/mask. Cùng loss `0.5 BCEWithLogits + 0.5 soft Dice`, batch 16, một loại AdamW cho mỗi run và cùng giới hạn 30 epoch. Checkpoint và ngưỡng dự đoán được chọn **chỉ trên validation** bằng mean Dice theo bệnh nhân; test dùng nguyên checkpoint/ngưỡng đó.

**Cách tính:** với mỗi bệnh nhân, gộp dự đoán và ground truth của đủ 32 lát rồi tính TP/FP/FN. Dice = `2TP/(2TP+FP+FN)`, IoU = `TP/(TP+FP+FN)`, precision = `TP/(TP+FP)`, recall = `TP/(TP+FN)`. Các bảng ghi **trung bình không trọng số theo bệnh nhân**, không lấy 480 lát test làm 480 mẫu độc lập. Pixel accuracy còn tính cả nền; đây là chỉ số phụ vì nền chiếm nhiều pixel. Trường hợp cả dự đoán và mask đều rỗng được tính Dice/IoU = 1 theo quy tắc dự án.

## 2. Kiến trúc: ba mức độ phức tạp khác nhau

| Mô hình | Khởi tạo và đường truyền đặc trưng | Tham số | Khác biệt có ý nghĩa |
|---|---|---:|---|
| **M1 — SimpleFCN** | Tự viết, khởi tạo ngẫu nhiên; 5 `ConvBlock`, kênh 24→48→96→48→24; 2 MaxPool, 2 bilinear upsample; **không skip**. | 240.097 | Baseline nhỏ; đường đi từ ảnh gốc đến mask ngắn, ít dung lượng lưu đặc trưng. |
| **M2 — ScratchUNet** | Tự viết hoàn toàn bằng `Conv2d`/BatchNorm/ReLU, khởi tạo ngẫu nhiên; encoder 32→64→128→256→512; 4 mức giảm kích thước và 4 skip nối sang decoder. | 7.849.601 | Sâu hơn M1, có skip để decoder nhận lại chi tiết không gian từ encoder; không gọi U-Net/ResNet dựng sẵn và không dùng pretrained. |
| **M3 — TransferUNet** | Encoder `torchvision` ResNet-18 với `ResNet18_Weights.IMAGENET1K_V1`; decoder 4 skip tự viết. Đóng băng encoder 5 epoch đầu, sau đó mở `layer4`; thống kê BatchNorm encoder giữ ở eval. | 14.404.257 | Mang đặc trưng ImageNet sang MRI; decoder học mới. Một AdamW xuyên suốt, hai param group có learning rate khác nhau. |

M1 và M2 cùng học từ đầu nhưng khác độ sâu, số kênh và skip. M2 và M3 khác **cả kiến trúc encoder/decoder, số tham số, khởi tạo và lịch freeze/lr**. Vì vậy phép so sánh này **không cô lập** tác dụng riêng của pretrained weights. Muốn trả lời câu hỏi nhân quả “pretrained giúp bao nhiêu”, phải chạy một cặp đối chứng cùng kiến trúc, cùng lịch train và chỉ khác weights khởi tạo; đó là thí nghiệm khác.

## 3. Kết quả validation và test của seed 42

### 3.1. Checkpoint được khóa bằng validation

| Mô hình | Epoch đã chạy | Epoch tốt nhất | Ngưỡng khóa | Validation Dice | Validation IoU | Validation precision | Validation recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 26 | 20 | 0.30 | 0.8027 | 0.6808 | 0.8280 | 0.8048 |
| M2 | 24 | 18 | 0.45 | 0.8213 | 0.7046 | 0.8136 | 0.8485 |
| M3 | 15 | 9 | 0.70 | **0.8259** | **0.7095** | **0.8473** | 0.8209 |

M3 chọn checkpoint ở epoch 9, tức sau giai đoạn mở `layer4` bắt đầu từ epoch 6. Cả ba dừng trước trần 30 epoch sau 6 epoch không cải thiện theo quy tắc đã chốt. Ngưỡng M3 = 0.70 là kết quả chọn trên validation, không chỉnh theo test.

### 3.2. Test: 15 bệnh nhân cố định

| Mô hình | Mean Dice ↑ | Mean IoU ↑ | Precision ↑ | Recall ↑ | Pixel accuracy* ↑ | Median Dice |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 0.7997 | 0.6797 | 0.8427 | 0.7925 | 0.9919 | 0.8173 |
| M2 | 0.8074 | 0.6910 | 0.8157 | **0.8224** | 0.9923 | 0.8407 |
| M3 | **0.8120** | **0.6951** | **0.8781** | 0.7726 | **0.9930** | **0.8584** |

\*Pixel accuracy chủ yếu chịu ảnh hưởng của vùng nền lớn; Dice/IoU phản ánh trực tiếp vùng u hơn. Precision và recall trong bảng là trung bình từng metric theo bệnh nhân, nên không suy ngược mean Dice bằng cách thế hai trung bình đó vào một công thức duy nhất.

Khoảng cách mean Dice M2–M1 là **+0.0077** (0,77 điểm phần trăm), M3–M1 **+0.0123**, M3–M2 **+0.0046**. M3 đứng đầu theo mean Dice/IoU **trong seed 42 này**, nhưng khoảng cách nhỏ. Mean Dice test của ba mô hình thấp hơn validation lần lượt khoảng 0.0030, 0.0139 và 0.0139. Hai tập gồm các bệnh nhân khác nhau; khoảng cách đó **không tự chứng minh overfitting**.

### 3.3. Nhìn theo grade để hiểu thứ hạng

| Tập test | Số ca | M1 Dice | M2 Dice | M3 Dice | Mô hình cao nhất trong nhóm |
|---|---:|---:|---:|---:|---|
| HGG | 12 | 0.8274 | **0.8373** | 0.8291 | M2 |
| LGG | 3 | 0.6890 | 0.6875 | **0.7436** | M3 |

Đây là điểm mấu chốt: **M2 cao hơn M3 trên nhóm HGG trung bình 0.0082**, còn **M3 cao hơn M2 trên LGG trung bình 0.0561**. Vì mean chung lấy trung bình 12 HGG + 3 LGG, lợi thế LGG đưa M3 lên trên M2 khoảng 0.0046. Chỉ **3 ca LGG** trong test; không thể coi chênh lệch LGG này là bằng chứng chắc chắn rằng M3 tổng quát tốt hơn trên toàn bộ bệnh nhân LGG.

## 4. Phân tích theo từng bệnh nhân, không chỉ nhìn mean

Ba model đã được ghép cặp trên đúng 15 `case_id` test. Với Dice, M2 cao hơn M1 ở **8/15** ca; M3 cao hơn M1 ở **10/15** ca; M3 cao hơn M2 ở **6/15** ca. Như vậy M3 có mean cao hơn M2 dù **thua M2 ở 9/15 ca**: một số ca M3 cải thiện nhiều đã kéo mean lên. Median của chênh lệch Dice M3–M2 là **−0.0021**, phù hợp với nhận xét này. Chỉ nhìn bảng mean sẽ che mất độ không đồng đều giữa bệnh nhân.

Khoảng chênh lệch Dice theo ca cũng đáng kể: M3–M2 chạy từ khoảng **−0.0744 đến +0.0610**; M3–M1 từ **−0.1239 đến +0.0699**. Một ca LGG là ca có Dice thấp nhất ở cả ba mô hình (M1 0.590, M2 0.573, M3 0.630). Đây là tín hiệu nên xem ảnh/mask dự đoán của ca đó trong phân tích lỗi cục bộ; CSV **không cho biết nguyên nhân hình thái học**, nên báo cáo không gán lỗi cụ thể cho kích thước, biên hoặc loại u khi chưa nhìn ảnh.

Để lượng hóa sự không chắc chắn do chỉ có 15 ca, đã bootstrap **cặp bệnh nhân** 20.000 lần (lấy mẫu lại 15 chênh lệch Dice theo `case_id`, seed phân tích 42; khoảng percentile 2,5–97,5%). Đây là phân tích thăm dò của test cohort cố định, **chưa** tính biến thiên do seed train, chọn cohort hoặc khác biệt kiến trúc.

| So sánh Dice theo ca | Chênh lệch mean | Số ca mô hình đứng trước thắng | Khoảng bootstrap 95% cho chênh lệch mean |
|---|---:|---:|---:|
| M2 − M1 | +0.0077 | 8/15 | −0.0092 đến +0.0256 |
| M3 − M1 | +0.0123 | 10/15 | −0.0139 đến +0.0347 |
| M3 − M2 | +0.0046 | 6/15 | −0.0165 đến +0.0253 |

Cả ba khoảng đều chứa 0. Vì vậy kết luận đúng mức là **thứ hạng quan sát được ở seed 42**, chưa phải bằng chứng chắc chắn một kiến trúc luôn tốt hơn kiến trúc khác. Bootstrap này chỉ dùng để mô tả độ nhạy của mean với mẫu 15 ca; không nên xem là kiểm định cuối của benchmark.

## 5. Precision–recall và giả thuyết cơ chế

**Quan sát từ test:** M2 có recall cao nhất (0.8224) nhưng precision thấp nhất (0.8157). M3 ngược lại: precision cao nhất (0.8781), recall thấp nhất (0.7726). M1 nằm giữa hai mô hình về precision và recall. M2 dùng ngưỡng 0.45, M3 0.70; ngưỡng cao của M3 **phù hợp với** việc dự đoán thận trọng hơn: ít false positive hơn nhưng có thể bỏ sót một phần WT. Đây là quan hệ hợp lý giữa ngưỡng và loại lỗi, **không** chứng minh khác biệt chỉ do ngưỡng vì bản thân xác suất và kiến trúc cũng khác nhau.

**Giả thuyết về M2 so với M1:** encoder sâu hơn có thể học ngữ cảnh lớn hơn; bốn skip cung cấp lại chi tiết không gian khi decoder phục hồi 128×128. Điều này phù hợp với mức recall và Dice HGG của M2 cao hơn M1. Tuy nhiên số tham số tăng khoảng **32,7 lần** mà mean Dice test chỉ tăng 0.0077; dữ liệu hiện tại không ủng hộ khẳng định “càng sâu càng tăng mạnh accuracy”. M2 cũng giảm Dice ở 7/15 ca, nên không phải mọi bệnh nhân đều hưởng lợi.

**Giả thuyết về M3:** encoder ImageNet có thể cho biểu diễn ban đầu hữu ích khi chỉ có 70 ca train; freeze 5 epoch giúp decoder học trước, rồi `layer4` thích nghi với MRI. Điều này phù hợp với việc M3 đạt checkpoint tốt nhất sớm (epoch 9) và mean Dice LGG cao hơn. Tuy nhiên ImageNet là ảnh tự nhiên, còn FLAIR là ảnh MRI lặp ba kênh; mô hình có thể bỏ sót vùng WT có tín hiệu khác ảnh tự nhiên. M3 thấp hơn M2 ở mean Dice HGG và recall test. **Không thể quy ưu/nhược này riêng cho pretrained**, vì M3 còn khác backbone, decoder, learning rate và lịch freeze.

## 6. Tài nguyên và tốc độ của các lần chạy đã lưu

Cả ba `config.json` ghi Python 3.12.10, PyTorch 2.11.0+cu128, GPU NVIDIA RTX A4500 Laptop. Thời gian là số đo `train_seconds` trong run; peak VRAM là `torch.cuda.max_memory_allocated()` do PyTorch ghi, **không phải** tổng VRAM mà toàn bộ tiến trình/driver sử dụng.

| Mô hình | Số tham số | Epoch thực chạy | Thời gian run | Peak VRAM PyTorch |
|---|---:|---:|---:|---:|
| M1 | 0.240 triệu | 26 | 137.7 s | 297.5 MB |
| M2 | 7.850 triệu | 24 | 251.1 s | 694.5 MB |
| M3 | 14.404 triệu | 15 | 87.3 s | 442.4 MB |

M2 tốn khoảng **1,82 lần** thời gian và **2,33 lần** peak VRAM của M1 trong các run này, đổi lấy +0.0077 Dice test. M3 có nhiều tham số nhất nhưng thời gian thấp nhất; điều đó đi cùng **ít epoch hơn** và phần lớn encoder vẫn frozen. Không nên suy diễn từ bảng này rằng ResNet pretrained luôn chạy nhanh hơn U-Net tự viết ở cùng epoch, phần cứng hoặc cấu hình khác. Số tham số không phản ánh trực tiếp thời gian, vì kích thước feature map, trạng thái gradient và số epoch đều ảnh hưởng chi phí.

## 7. Kết luận và giới hạn báo cáo

- **Trong seed 42**, M3 có mean Dice/IoU test cao nhất (**0.8120/0.6951**) và precision cao nhất, nhưng chỉ nhỉnh M2 **0.0046 Dice**; M2 có recall cao nhất và mean Dice HGG cao nhất. M1 vẫn đạt Dice **0.7997** với ít tham số nhất, là baseline có giá trị so sánh.
- Khác biệt giữa ba hệ thống nhỏ so với độ biến thiên giữa 15 bệnh nhân; các khoảng bootstrap thăm dò đều chứa 0. **Không tuyên bố M3 vượt trội chắc chắn** và không tách riêng “lợi ích pretrained” từ phép so sánh ba kiến trúc này.
- Test chỉ có **15 ca, gồm 3 LGG**, và hiện mới có **một seed** cho mỗi mô hình. Sau khi có seed 123/2026, báo mean ± SD **qua seed** cho mỗi model, giữ nguyên test và không chọn seed tốt nhất. Nếu muốn tìm nguyên nhân lỗi, xem trực tiếp mask/overlay của các ca khó và báo kiểu lỗi quan sát được; không điều chỉnh threshold, checkpoint hay kiến trúc theo điểm test đã mở.

**Cách tái lập bảng:** chạy từng notebook `Mx/Mx.ipynb` trên Windows `.venv`/CUDA với `SEED = 42`, hoặc `& .\.venv\Scripts\python.exe .\Mx\Mx.py --train --seed 42` từ root. Mỗi run train, khóa checkpoint/ngưỡng bằng validation rồi test chính Mx; các CSV cục bộ trong `runs/<mx>/42/` là nguồn để tính lại mean theo bệnh nhân và chênh lệch ghép cặp.
