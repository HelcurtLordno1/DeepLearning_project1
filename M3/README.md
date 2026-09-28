# M3 — U-Net/ResNet-18 transfer learning

**Owner:** thành viên 3. [M3.ipynb](M3.ipynb) là bản đọc/trình bày/chạy chính; [M3.py](M3.py) là cách chạy PowerShell. Cả hai tự chứa toàn bộ pipeline và **cùng class `ResidualUNet` với M2**. M3 dùng `ResNet18_Weights.IMAGENET1K_V1`, freeze encoder 5 epoch rồi mở `layer4` tối đa 25 epoch; không import mã M2.

```powershell
& .\.venv\Scripts\python.exe .\M3\M3.py --smoke
& .\.venv\Scripts\python.exe .\M3\M3.py --train --seed 42
```

Chạy từ **root trong Windows PowerShell**; lần đầu tải ImageNet weights qua `torchvision`. Notebook dùng kernel Windows `.venv`, cell cuối mặc định smoke. Lặp full với seed 123/2026; `--test` sau khi cả ba Mx khóa checkpoint. Artifacts trong `runs/smoke/m3/42/` hoặc `runs/m3/<seed>/`. Đọc [phân công](../Detail_jobs.md#5-thành-viên-3--m3-transfer-learning-và-tích-hợp), [PyTorch transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), [ResNet weights](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18).
