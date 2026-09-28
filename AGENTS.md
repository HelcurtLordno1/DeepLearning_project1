# Quy tắc bắt buộc cho mọi agent và thành viên — Project_midterm

Phạm vi: toàn bộ repository. Đọc `README.md`, `Project_structure.md`, `Detail_jobs.md` và README trong Mx liên quan trước khi sửa. Yêu cầu mới nhất của người dùng trong hội thoại được ưu tiên. Trạng thái và kết quả phải phản ánh những gì đã chạy thật.

## 1. Vị trí mã và hai cách chạy

- **Mọi mã Python thực hiện bài toán nằm trong `M1/`, `M2/`, `M3/`.** Mỗi thư mục có đúng một notebook chính `Mx.ipynb` và một file `Mx.py`. Notebook là bản trình bày và chạy **chính**, phải chứa trực tiếp đầy đủ mã đọc dữ liệu, tiền xử lý, model, loss, train, validation, test và hình minh họa, kèm Markdown giải thích. File `.py` tự chứa cùng quy trình để chạy bằng PowerShell. Được import thư viện chuẩn và package cài qua pip; **không import, `%run`, `exec`, `%load` hay gọi file Python của project hoặc notebook khác**. Không tạo lại `scripts/`, `src/`, notebook chung, package nội bộ hay file Python ở root/data/reports.
- Notebook và `.py` của cùng Mx phải có cùng công thức, tham số và đường dẫn artifact. Khi sửa một bên, sửa bên còn lại trong cùng thay đổi; kiểm cả **Restart Kernel + Run All** và lệnh PowerShell. Việc lặp mã giữa Mx là lựa chọn có chủ ý để từng thành viên đọc trọn quy trình; mọi thay đổi hợp đồng benchmark phải được đồng bộ ở cả sáu file.
- Lệnh PowerShell từ root: `& .\.venv\Scripts\python.exe .\M1\M1.py --smoke` (tương tự M2/M3). Cả ba notebook chỉ cần `SEED` ở ô đầu: **Run All train đầy đủ, hiện log/biểu đồ/metric validation rồi test của chính Mx**. `Mx.py --train --seed 42|123|2026` làm cùng quy trình; `--smoke` chỉ kiểm nhanh. Mỗi Mx chỉ cần checkpoint của chính mình để test; benchmark tổng hợp CSV sau này. Không mô tả smoke như kết quả benchmark.
- `data/` chỉ chứa dataset, split, checksum và cache; `runs/` chứa checkpoint/kết quả; `reports/` chứa báo cáo/hình. README và `Detail_jobs.md` là tài liệu. `requirements.txt` là danh sách thư viện. `slides/` do người dùng cung cấp, không xóa hoặc sửa khi không được yêu cầu.

## 2. Windows và môi trường chung

- Dùng đúng `.venv` **Windows** ở root, tạo bằng `py -3.12 -m venv .venv` nếu thiếu. Không tạo `.venv` bằng WSL/Linux, không dùng kernel WSL hoặc đường dẫn `/mnt/...`/`/home/...` trong code, config, notebook và hướng dẫn chạy.
- Chạy notebook trên Windows với kernel có `sys.executable` trỏ tới `Project_midterm\.venv\Scripts\python.exe`; file `.py` dùng chính interpreter đó từ PowerShell. Kiểm `torch.cuda.is_available()`; benchmark cuối dùng GPU CUDA. Nếu CUDA thiếu, ghi lỗi rõ và sửa môi trường trước khi train cuối.
- Dùng `pathlib.Path` tìm root từ current working directory hoặc `__file__`, rồi đường dẫn tương đối tới `data/`, `runs/`, `reports/`. Không hard-code ổ đĩa/người dùng trong mã. Lệnh README có thể dùng đường dẫn D: cụ thể của máy chủ để giúp mở thư mục.
- Không commit `.venv/`, MRI gốc, cache, checkpoint, dữ liệu bệnh nhân hoặc log lớn. Output minh họa trong notebook chỉ commit khi người dùng yêu cầu rõ, như lần push ngày 29/09/2026; không thêm bí mật vào notebook/`.env`.

## 3. Hợp đồng benchmark không được đổi âm thầm

- BraTS 2015, đúng 100 `case_id = grade/patient_id` trong `data/splits_v1.csv`: 80 HGG/20 LGG, 70 train/15 val/15 test. Mọi Mx đọc đúng 32 lát/ca từ 20–80% độ sâu ảnh, resize 128×128, FLAIR đơn modality, WT = OT thuộc `{1,2,3,4}`. Không dùng mask để chọn lát.
- Cùng chuẩn hóa FLAIR theo median/IQR voxel khác 0, clip `[-5,5]`, lặp 3 kênh và ImageNet mean/std; cùng augmentation train, batch, loss `0.5 BCEWithLogits + 0.5 soft Dice`, seed `42,123,2026`, ngưỡng validation và metric patient-level. `data/file_sha256.csv` xác nhận file raw; cache chung ở `data/processed/flair_wt_v1/`.
- M1: CNN encoder–decoder nhỏ, tự xây từ các lớp `nn.Conv2d`/`nn.BatchNorm2d`, học từ đầu. M2: U-Net nhiều tầng với encoder và decoder **đều tự viết bằng các lớp cơ bản**, khởi tạo ngẫu nhiên; không gọi model dựng sẵn hoặc pretrained. M3: transfer learning với encoder pretrained và decoder phân đoạn tự viết. M2/M3 được phép khác kiến trúc; báo cáo phải nêu rõ vì vậy không quy chênh lệch điểm riêng cho pretrained weights.
- Mỗi Mx chốt **một loại optimizer** cho một lần train, không tự chạy vòng tìm optimizer/kiến trúc; lịch learning rate và dừng sớm dựa trên validation được phép. Ba seed 42/123/2026 là ba lần lặp để báo độ ổn định sau khi khóa cấu hình, không phải quá trình chọn optimizer.
- Chọn threshold/checkpoint/hyperparameter bằng validation. Sau khi **từng Mx** khóa checkpoint và ngưỡng của chính mình, Mx đó được đánh giá test độc lập; không chờ Mx khác và không chọn lại model bằng test. Tính Dice/IoU theo **bệnh nhân** từ 32 lát, không xem lát như mẫu độc lập. Điểm test từng Mx được tổng hợp về sau từ CSV; không ghi số đo chưa chạy.

## 4. Cách sửa và bàn giao

- Mỗi thành viên sở hữu một Mx theo `Detail_jobs.md`. Không thêm API ẩn hay module dùng chung. Nếu thay quy tắc dữ liệu/metric, cập nhật cả ba notebook, ba `.py`, `README.md`, `Project_structure.md`, `Detail_jobs.md` và báo nhóm trước khi chạy lại kết quả.
- Cả ba notebook Run All chạy full train/validation/test của một seed và hiển thị output; `Mx.py --smoke` là kiểm nhanh riêng. Hiện đã có artifact full seed 42 và output notebook cho cả M1/M2/M3. Khi code từng Mx ổn định, chạy full cho 3 seed. Ghi phiên bản package, thời gian, peak VRAM, checkpoint, CSV theo ca và hình thật. Test từng Mx sau khi khóa checkpoint/ngưỡng của chính nó.
- Khi kiểm tra cài đặt, dùng tài liệu [PyTorch Windows](https://docs.pytorch.org/get-started/locally/), [Python venv](https://docs.python.org/3.12/library/venv.html), [Jupyter kernel](https://ipython.readthedocs.io/en/stable/install/index.html). Chọn wheel CUDA phù hợp máy từ nguồn chính thức; không tự tạo môi trường Linux.
