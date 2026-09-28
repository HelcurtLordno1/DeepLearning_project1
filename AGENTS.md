# Quy tắc bắt buộc cho mọi agent và thành viên — Project_midterm

Phạm vi: toàn bộ repository. Đọc `README.md`, `Project_structure.md`, `Detail_jobs.md` và README trong Mx liên quan trước khi sửa. Yêu cầu mới nhất của người dùng trong hội thoại được ưu tiên. Trạng thái và kết quả phải phản ánh những gì đã chạy thật.

## 1. Vị trí mã và hai cách chạy

- **Mọi mã Python thực hiện bài toán nằm trong `M1/`, `M2/`, `M3/`.** Mỗi thư mục có đúng một notebook chính `Mx.ipynb` và một file `Mx.py`. Notebook là bản trình bày và chạy **chính**, phải chứa trực tiếp đầy đủ mã đọc dữ liệu, tiền xử lý, model, loss, train, validation, test và hình minh họa, kèm Markdown giải thích. File `.py` tự chứa cùng quy trình để chạy bằng PowerShell. Được import thư viện chuẩn và package cài qua pip; **không import, `%run`, `exec`, `%load` hay gọi file Python của project hoặc notebook khác**. Không tạo lại `scripts/`, `src/`, notebook chung, package nội bộ hay file Python ở root/data/reports.
- Notebook và `.py` của cùng Mx phải có cùng công thức, tham số và đường dẫn artifact. Khi sửa một bên, sửa bên còn lại trong cùng thay đổi; kiểm cả **Restart Kernel + Run All** và lệnh PowerShell. Việc lặp mã giữa Mx là lựa chọn có chủ ý để từng thành viên đọc trọn quy trình; mọi thay đổi hợp đồng benchmark phải được đồng bộ ở cả sáu file.
- Lệnh PowerShell từ root: `& .\.venv\Scripts\python.exe .\M1\M1.py --smoke` (tương tự M2/M3). `--train --seed 42|123|2026` chạy một seed đầy đủ; `--test --seed ...` chỉ sau khi cả ba model đã có checkpoint cho seed đó. Trong notebook, cell cuối chọn `ACTION = "smoke"|"prepare"|"train"|"test"` và `SEED`; mặc định smoke. Không mô tả smoke như kết quả benchmark.
- `data/` chỉ chứa dataset, split, checksum và cache; `runs/` chứa checkpoint/kết quả; `reports/` chứa báo cáo/hình. README và `Detail_jobs.md` là tài liệu. `requirements.txt` là danh sách thư viện. `slides/` do người dùng cung cấp, không xóa hoặc sửa khi không được yêu cầu.

## 2. Windows và môi trường chung

- Dùng đúng `.venv` **Windows** ở root, tạo bằng `py -3.12 -m venv .venv` nếu thiếu. Không tạo `.venv` bằng WSL/Linux, không dùng kernel WSL hoặc đường dẫn `/mnt/...`/`/home/...` trong code, config, notebook và hướng dẫn chạy.
- Chạy notebook trên Windows với kernel có `sys.executable` trỏ tới `Project_midterm\.venv\Scripts\python.exe`; file `.py` dùng chính interpreter đó từ PowerShell. Kiểm `torch.cuda.is_available()`; benchmark cuối dùng GPU CUDA. Nếu CUDA thiếu, ghi lỗi rõ và sửa môi trường trước khi train cuối.
- Dùng `pathlib.Path` tìm root từ current working directory hoặc `__file__`, rồi đường dẫn tương đối tới `data/`, `runs/`, `reports/`. Không hard-code ổ đĩa/người dùng trong mã. Lệnh README có thể dùng đường dẫn D: cụ thể của máy chủ để giúp mở thư mục.
- Không commit `.venv/`, MRI gốc, cache, checkpoint, dữ liệu bệnh nhân, log lớn hoặc output notebook. Không thêm bí mật vào notebook/`.env`.

## 3. Hợp đồng benchmark không được đổi âm thầm

- BraTS 2015, đúng 100 `case_id = grade/patient_id` trong `data/splits_v1.csv`: 80 HGG/20 LGG, 70 train/15 val/15 test. Mọi Mx đọc đúng 32 lát/ca từ 20–80% độ sâu ảnh, resize 128×128, FLAIR đơn modality, WT = OT thuộc `{1,2,3,4}`. Không dùng mask để chọn lát.
- Cùng chuẩn hóa FLAIR theo median/IQR voxel khác 0, clip `[-5,5]`, lặp 3 kênh và ImageNet mean/std; cùng augmentation train, batch, loss `0.5 BCEWithLogits + 0.5 soft Dice`, seed `42,123,2026`, ngưỡng validation và metric patient-level. `data/file_sha256.csv` xác nhận file raw; cache chung ở `data/processed/flair_wt_v1/`.
- M1: CNN nông từ đầu. M2 và M3: **cùng class U-Net/ResNet-18 và decoder**, mã hiện trực tiếp trong cả hai notebook và `.py`; M2 dùng encoder random, M3 dùng ImageNet weights, freeze encoder 5 epoch rồi mở `layer4` tối đa 25 epoch. Mọi khác biệt khác phải ghi thành thí nghiệm riêng.
- Chọn threshold/checkpoint/hyperparameter bằng validation. Chỉ mở test sau khi đủ ba model có checkpoint đã khóa; tính Dice/IoU theo **bệnh nhân** từ 32 lát, không xem lát như mẫu độc lập. Không ghi số đo chưa chạy.

## 4. Cách sửa và bàn giao

- Mỗi thành viên sở hữu một Mx theo `Detail_jobs.md`. Không thêm API ẩn hay module dùng chung. Nếu thay quy tắc dữ liệu/metric, cập nhật cả ba notebook, ba `.py`, `README.md`, `Project_structure.md`, `Detail_jobs.md` và báo nhóm trước khi chạy lại kết quả.
- Mỗi Mx phải chạy smoke trên **cả notebook và PowerShell**, sau đó full train/validation cho 3 seed. Ghi phiên bản package, thời gian, peak VRAM, checkpoint, CSV theo ca và hình thật. Test một lần sau khi khóa cấu hình.
- Khi kiểm tra cài đặt, dùng tài liệu [PyTorch Windows](https://docs.pytorch.org/get-started/locally/), [Python venv](https://docs.python.org/3.12/library/venv.html), [Jupyter kernel](https://ipython.readthedocs.io/en/stable/install/index.html). Chọn wheel CUDA phù hợp máy từ nguồn chính thức; không tự tạo môi trường Linux.
