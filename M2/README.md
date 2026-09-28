# M2 — U-Net/ResNet-18 từ đầu

**Owner:** thành viên 2. [M2.ipynb](M2.ipynb) là bản đọc/trình bày/chạy chính; [M2.py](M2.py) là cách chạy PowerShell. Cả hai tự chứa toàn bộ pipeline và class `ResidualUNet` với ResNet-18 encoder random, bốn up-block/skip, head phân đoạn. Class/decoder phải khớp M3; không import mã M3.

```powershell
& .\.venv\Scripts\python.exe .\M2\M2.py --smoke
& .\.venv\Scripts\python.exe .\M2\M2.py --train --seed 42
```

Chạy từ **root trong Windows PowerShell**; notebook dùng kernel Windows `.venv`, cell cuối mặc định smoke. Lặp full với seed 123/2026; `--test` sau khi cả ba Mx khóa checkpoint. Artifacts trong `runs/smoke/m2/42/` hoặc `runs/m2/<seed>/`. Đọc [phân công](../Detail_jobs.md#4-thành-viên-2--m2-u-netresnet-18-từ-đầu), [U-Net](https://arxiv.org/abs/1505.04597), [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).
