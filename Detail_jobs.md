# Phân công và nghiệm thu — mỗi thành viên một Mx

Mỗi thành viên đọc và trình bày được **toàn bộ pipeline trong notebook của mình**: xác minh dữ liệu, chuẩn hóa, model, loss, train, validation, test và hình. Notebook là bản chính; `.py` tự chứa cùng logic để chạy PowerShell. Không import code Mx khác. [Project_structure.md](Project_structure.md) là đặc tả kiến trúc và benchmark; [AGENTS.md](AGENTS.md) là quy tắc repository.

## Luồng công việc chung

```mermaid
flowchart LR
    G0["G0: raw/checksum + Windows .venv + CUDA"] --> G1["G1: code và docs khớp hợp đồng"]
    G1 --> G2["G2: M1 Run All; M2/M3 smoke cho code mới"]
    G2 --> G3["G3: full train + test độc lập từng seed/Mx"]
    G3 --> G4["G4: tổng hợp CSV và báo cáo"]
```

| Gate | Điều kiện nghiệm thu |
|---|---|
| G0 | 201 SHA-256 đúng; kernel và PowerShell dùng `.venv\Scripts\python.exe`; CUDA sẵn sàng. |
| G1 | Cả sáu file cùng split, 32 lát/ca, 128×128, FLAIR/WT, chuẩn hóa, augmentation, loss và patient-level metrics. M2 hoàn toàn tự viết và khởi tạo ngẫu nhiên; M3 ghi pretrained weights. |
| G2 | Cả ba notebook đã lưu output Run All seed 42; `Mx.py --smoke` kiểm nhanh riêng và smoke không phải điểm benchmark. |
| G3 | `--train --seed 42`, `123`, `2026` cho từng Mx. Mỗi run dùng một AdamW, checkpoint/ngưỡng chọn trên validation. Sau khi khóa, Mx đó tự đánh giá 15 ca test và lưu CSV, **không chờ Mx khác**. |
| G4 | Tổng hợp CSV test độc lập của ba Mx theo bệnh nhân, mean/std và báo cáo. Không chọn lại model/hyperparameter theo test. |

Ba seed sau khi khóa cấu hình nhằm báo độ ổn định, không phải ba lựa chọn optimizer. Mốc tối đa 30 epoch; dừng sớm sau 6 epoch không cải thiện và tối thiểu 8 epoch là quy tắc chốt trước khi train. Nếu sửa quy tắc dữ liệu/loss/metric, cập nhật cả sáu file, README và đặc tả trước khi chạy lại. MRI/cache/checkpoint/log lớn không commit.

## Thành viên 1 — M1, CNN nhỏ tự xây

| ID | Việc và bằng chứng |
|---|---|
| M1-01 | Trong notebook, in split/shape và xem FLAIR–WT mẫu; giải thích `case_id`, OT→WT, 32 lát theo độ sâu ảnh và cache chung. Bước xác minh 201 checksum của dữ liệu gốc nằm trong quy trình chuẩn bị dữ liệu của dự án, không lặp ở mỗi Run All. |
| M1-02 | Giải thích `ConvBlock` gồm `Conv2d → BatchNorm2d → ReLU` hai lần; `SimpleFCN` 24→48→96→48→24, hai MaxPool, hai bilinear upsample, head 1×1; **không pretrained/skip**. Xác minh input `(B,3,128,128)` → logits `(B,1,128,128)`. |
| M1-03 | Giải thích BCE + soft Dice, AdamW, scheduler, dừng sớm, threshold/checkpoint theo patient-level val. Notebook Run All full và in log/điểm/hình; PowerShell `--smoke` kiểm nhanh trên Windows `.venv`/CUDA. |
| M1-04 | Sau khi code ổn định, chạy full 3 seed; ghi thời gian, peak VRAM, loss/val curve, ngưỡng, preview thực vào `runs/m1/<seed>/` và README. |
| M1-05 | M1 Run All test ngay sau khi train/khóa checkpoint của cùng seed; kiểm `metrics_test.csv` có 15 ca, bàn giao để tổng hợp. |

Đọc [M1/README.md](M1/README.md) để xem sơ đồ và giải thích từng nhóm hàm. M1 là baseline có dung lượng vừa phải, không hứa điểm cao nhất trước khi chạy.

## Thành viên 2 — M2, U-Net nhiều tầng viết từ đầu

| ID | Việc và bằng chứng |
|---|---|
| M2-01 | Viết `DoubleConv`, `DownBlock`, `UpBlock`, `ScratchUNet` trực tiếp trong `M2.ipynb` và `M2.py`; chỉ dùng lớp PyTorch cơ bản. **Không gọi built-in U-Net/ResNet hoặc bất kỳ pretrained weights nào**, kể cả built-in `weights=None`. |
| M2-02 | Encoder 32→64→128→256→512; decoder dùng bốn skip, concat và DoubleConv; head 1×1. In và kiểm shape mỗi mức, tổng số tham số, gradient của encoder từ epoch đầu. |
| M2-03 | Giữ data/loss/metric chung; một AdamW và scheduler trong một run, dừng sớm theo validation; không có vòng thử optimizer. Chạy smoke cả notebook/PowerShell từ clean kernel. |
| M2-04 | Chạy full ba seed, lưu artifact và số đo thật; giải thích khác M1 ở độ sâu/skip và khác M3 ở pretrained **lẫn kiến trúc**. |
| M2-05 | Sau khi M2 khóa checkpoint của mình, test theo ca và bàn giao CSV; không diễn giải riêng tác dụng pretrained từ so sánh M2/M3. |

Các slide CNN do người dùng cung cấp trong `slides/` có thể dùng để học cách tổ chức `nn.Module`, `nn.Conv2d`, pooling và optimizer; giữ file slide nguyên vẹn. Sơ đồ chi tiết trong [Project_structure.md](Project_structure.md#2-kiến-trúc-và-phạm-vi-so-sánh).

## Thành viên 3 — M3, transfer learning và tổng hợp

| ID | Việc và bằng chứng |
|---|---|
| M3-01 | Kiểm môi trường Windows `.venv`, CUDA và nguồn pretrained weights; lưu phiên bản Python/torch/torchvision/GPU. |
| M3-02 | Dùng encoder pretrained (thiết kế hiện hành: ResNet-18 ImageNet) và decoder phân đoạn tự viết. Freeze encoder + BN eval 5 epoch, sau đó mở `layer4`. Dùng **cùng một instance AdamW** với param groups suốt run khi cập nhật code. |
| M3-03 | Giữ data/loss/metric chung, dừng sớm validation; smoke notebook/PowerShell sau khi hoàn tất code mới. Không dùng các run smoke cũ để xác nhận thiết kế mới. |
| M3-04 | Full ba seed và test độc lập sau khi M3 khóa từng checkpoint; lưu checkpoint, per-case CSV, preview, thời gian/VRAM. Nêu rõ transfer từ ImageNet sang MRI là giả thuyết cần kiểm chứng. |
| M3-05 | Tổng hợp `reports/benchmark.csv` và báo cáo từ CSV thật: mean/std Dice/IoU theo bệnh nhân, số tham số, thời gian, peak VRAM, giới hạn cohort 15 ca test. Không dùng test để tinh chỉnh. |

## Trạng thái hiện tại

Dữ liệu raw đã kiểm 201/201 và cache 100 ca. Seed 42 đã có artifact full cho cả ba Mx: M1 26 epoch, checkpoint epoch 20, test mean Dice **0.7997**; M2 24 epoch, checkpoint epoch 18, **0.8074**; M3 15 epoch, checkpoint epoch 9, **0.8120**. Mỗi test có 15 ca và cả ba notebook đã lưu output. Seed 123/2026 chưa có kết quả. Xem [Benchmark_evaluate.md](Benchmark_evaluate.md) để đọc bảng so sánh seed 42 và giới hạn suy luận.
