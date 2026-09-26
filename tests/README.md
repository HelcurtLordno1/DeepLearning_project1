# Các kiểm tra phải có khi triển khai

- Split theo `case_id = grade/patient_id` không rò rỉ train/val/test; không lấy nhầm tập testing không nhãn.
- Ghép đúng FLAIR–OT, mask chỉ còn 0/1, ảnh và mask cùng shape sau resize.
- Dice/IoU theo bệnh nhân đúng trên ví dụ tay và mask rỗng.
- Notebook từ clean kernel và file Python từ PowerShell đọc cùng config/split, nạp cùng checkpoint và cho dự đoán tương ứng.

Không viết test chỉ lặp lại cấu trúc hàm. Mỗi test kiểm tra một lỗi có thể làm sai benchmark.
