# M1 — CNN nông

**Owner:** thành viên 1. [M1.ipynb](M1.ipynb) là bản đọc/trình bày/chạy chính; [M1.py](M1.py) là cách chạy PowerShell. Cả hai tự chứa kiểm dữ liệu, 32 lát FLAIR/WT, chuẩn hóa/cache, CNN 3 Conv, loss/metric patient-level, train/validation/test và ảnh preview. Không import mã project khác.

```powershell
& .\.venv\Scripts\python.exe .\M1\M1.py --smoke
& .\.venv\Scripts\python.exe .\M1\M1.py --train --seed 42
```

Chạy từ **root trong Windows PowerShell** với `.venv` chung; notebook dùng kernel `BraTS 2015 (.venv)`, cell cuối mặc định smoke. Lặp `--train` cho seed 123 và 2026; `--test` chỉ sau khi cả M1/M2/M3 có checkpoint full. Artifacts: `runs/smoke/m1/42/` hoặc `runs/m1/<seed>/`. Đọc [phân công và nghiệm thu](../Detail_jobs.md#3-thành-viên-1--m1-baseline-đơn-giản) và [FCN paper](https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html).
