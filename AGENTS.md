# Quy tắc làm việc cho agent và thành viên dự án

Phạm vi: toàn bộ repository `Project_midterm/`. Đọc `README.md`, `Project_structure.md` và `Detail_jobs.md` trước khi sửa code. Yêu cầu của người dùng trong hội thoại hiện tại được ưu tiên nếu khác tài liệu này.

## 1. Hai cách chạy bắt buộc

1. **Notebook `.ipynb` là luồng chạy chính và cách trình bày kết quả.** Mỗi mức có đúng một notebook chính: `M1/M1.ipynb`, `M2/M2.ipynb`, `M3/M3.ipynb`. Notebook chung `notebooks/00_prepare.ipynb` chuẩn bị dữ liệu; `notebooks/90_compare.ipynb` tổng hợp benchmark. Mỗi notebook phải chạy từ đầu tới cuối bằng **Restart Kernel + Run All** trong Windows với kernel của `.venv`.
2. **PowerShell + file Python là luồng chạy thứ hai.** Các entry point `scripts/prepare_data.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/compare.py`, `scripts/profile.py` phải chạy bằng `& .\.venv\Scripts\python.exe .\scripts\<file>.py ...` từ thư mục gốc dự án. Không dùng cú pháp shell Linux trong tài liệu hướng dẫn chạy.
3. Hai luồng trên phải gọi **cùng hàm trong `src/brats_benchmark/`**, dùng cùng file config, split và phép tính metric. Notebook giữ code điều phối, Markdown, biểu đồ, phân tích; không sao chép nguyên vòng train hoặc logic tiền xử lý từ Python source vào notebook.

## 2. Môi trường Windows

- Tạo virtual environment bằng **Python Windows** ở `Project_midterm/.venv/` (`py -3.12 -m venv .venv`). Không tạo hoặc dùng `.venv` bằng WSL/Linux. Không dùng đường dẫn `/mnt/...` hay `/home/...` trong code, config và lệnh chạy của dự án.
- Chọn interpreter/kernel trỏ tới `Project_midterm\.venv\Scripts\python.exe`. Kiểm tra `sys.executable` trong notebook và `torch.cuda.is_available()` ở cả notebook lẫn PowerShell. Nếu CUDA chưa sẵn sàng thì ghi rõ nguyên nhân, không ngầm chạy benchmark cuối trên CPU.
- Dùng `pathlib.Path` và đường dẫn tương đối từ project root. Không hard-code tên người dùng, ổ đĩa hoặc đường dẫn dữ liệu của một thành viên.
- `.venv/`, dữ liệu MRI, cache, checkpoint lớn và output notebook quá lớn phải được bỏ qua trong Git. Không lưu thông tin cá nhân/bí mật vào `.env` hoặc notebook.
- Chỉ ghi lệnh cài PyTorch CUDA theo [bộ chọn chính thức cho Windows](https://docs.pytorch.org/get-started/locally/) sau khi xác nhận phiên bản phù hợp; tránh chép một URL wheel có thể lỗi thời vào tài liệu.

## 3. Hợp đồng benchmark chung

- Chỉ dùng BraTS 2015 đã kiểm kê. Tập con cố định 100 bệnh nhân có FLAIR và OT: 80 HGG, 20 LGG; split theo bệnh nhân 70/15/15. Mọi mô hình đọc cùng `data/splits_v1.csv` và cùng 32 lát/ca, kích thước 128×128.
- Mask WT nhị phân: các giá trị OT `{1,2,3,4}` thành 1; 0 thành 0. Cùng chuẩn hóa ảnh, augmentation train, loss, batch hiệu dụng, định nghĩa metric và danh sách seed cho cả M1/M2/M3.
- M1 là CNN nông; M2 và M3 dùng **cùng factory U-Net/ResNet-18**. M2 khởi tạo encoder ngẫu nhiên; M3 dùng ImageNet weights rồi fine-tune. Không thay decoder hoặc input của riêng M3 mà không ghi thành thí nghiệm khác.
- Chọn checkpoint, hyperparameter và ngưỡng trên validation. Chỉ chạy test sau khi ba mô hình đã khóa cấu hình. Dice/IoU tính theo **bệnh nhân**, không gộp mọi lát thành một mẫu độc lập.
- Pilot/smoke test ghi vào `runs/smoke/`; không trộn với kết quả benchmark. Không bịa số Dice, thời gian hoặc ảnh demo.

## 4. Chia việc và tích hợp

- Một thành viên chịu trách nhiệm một mức: M1, M2 hoặc M3. Phân bổ chi tiết và mốc bàn giao ở `Detail_jobs.md`; các module chung có một người sở hữu rõ ràng và được hai người còn lại review.
- Trước khi code Mx, thống nhất chữ ký hàm cho `prepare_data`, `train_model`, `evaluate_model`, config và định dạng `runs/<model>/<seed>/`. Notebook và script phải dùng chính các hàm này.
- Mọi thay đổi về split, nhãn, metric hoặc preprocessing phải cập nhật `configs/benchmark.yaml`, `README.md`, `Project_structure.md` và thông báo cả nhóm trước khi chạy lại benchmark.
- Khi bàn giao: chạy kiểm tra split/metric, smoke notebook từ clean kernel, smoke PowerShell script tương ứng; ghi lệnh đã chạy, kết quả và hạn chế. Không đánh dấu hoàn thành nếu chỉ một trong hai cách chạy hoạt động.

## 5. Tài liệu tham chiếu

- [Python `venv` trên Windows](https://docs.python.org/3.12/library/venv.html), [PyTorch cài trên Windows](https://docs.pytorch.org/get-started/locally/), [IPython kernel](https://ipython.readthedocs.io/en/stable/install/index.html), [VS Code chọn kernel Jupyter](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management).
