# Notebook dùng chung

- `00_prepare.ipynb`: chuẩn bị dữ liệu từ `data/raw/BRATS2015/`, kiểm kê FLAIR/OT, kiểm tra `data/splits_v1.csv`, tạo cache và EDA. Thành viên 1 phụ trách.
- `90_compare.ipynb`: đọc checkpoint/kết quả đã khóa của M1/M2/M3, vẽ bảng và hình so sánh. Thành viên 3 phụ trách, thành viên 2 review metric.

Notebook gọi `brats_benchmark.pipeline` từ Windows `.venv`, không chép logic mô hình hoặc metric vào cell. Các file `.ipynb` sẽ được tạo trong các đầu việc ở `Detail_jobs.md`.
