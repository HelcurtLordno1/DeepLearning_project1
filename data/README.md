# Dữ liệu BraTS 2015 cho cả M1/M2/M3

`raw/BRATS2015/` có 200 MRI FLAIR/OT `.mha` và một file giấy phép. PowerShell Windows đã đối chiếu **201/201 file với SHA-256** trong `file_sha256.csv` ngày 28/09/2026. Dữ liệu gốc nằm trên máy và bị `.gitignore` loại khỏi Git; repo chỉ giữ manifest/checksum để tái tạo cùng cohort.

```text
data/
├── source_manifest.json          # 100 case đã khóa, đường dẫn, byte, SHA-1 nguồn
├── splits_v1.csv                # case_id, grade, train/val/test, đường dẫn FLAIR/OT
├── file_sha256.csv              # SHA-256 200 MRI + 1 license
├── raw/BRATS2015/
│   ├── License_CC_BY_NC_SA_3.0.txt
│   └── training/{HGG,LGG}/<patient_id>/*.mha
└── processed/flair_wt_v1/      # cache 32 lát/ca, tạo bởi từng Mx khi chạy
```

Mỗi `Mx.ipynb` và `Mx.py` tự đọc các manifest, tự kiểm checksum và dùng cùng cache. `patient_id` có thể trùng giữa HGG/LGG, nên khóa bệnh nhân là `case_id = grade/patient_id`. Không chỉnh split hay cache một mình; thay preprocessing phải đồng bộ **cả sáu file Mx** và đổi cache version. Cách tải lại bằng PowerShell trên máy khác nằm ở [README chính](../README.md#tải-lại-trên-máy-windows-khác-nếu-dataraw-trống).
