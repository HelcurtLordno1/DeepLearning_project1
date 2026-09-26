# Bố trí dữ liệu

Script `scripts/download_dataset.py` đối chiếu metadata torrent BraTS 2015 với inventory web seed Archive.org, chọn 100 cặp FLAIR–OT hiện có, rồi khóa danh sách trong `source_manifest.json`. Bản sao HTTPS chỉ có 100 HGG và 54 LGG hoàn chỉnh; script lấy cố định 80 HGG + 20 LGG.

```text
data/
├── README.md
├── source_manifest.json      # Kế hoạch chọn 100 ca và provenance; commit được
├── splits_v1.csv             # 70/15/15 theo case_id; commit được
├── metadata/brats2015.torrent  # Cache tải về; bỏ qua Git
├── raw/BRATS2015/
│   ├── License_CC_BY_NC_SA_3.0.txt
│   └── training/{HGG,LGG}/<patient_id>/{FLAIR,OT}.mha
└── processed/               # Cache 32 lát × 128×128; bỏ qua Git
```

Một `patient_id` có thể xuất hiện trong cả HGG và LGG của torrent, vì vậy khóa ca là **`case_id = grade/patient_id`**. `source_manifest.json` và `splits_v1.csv` chỉ ghi metadata/đường dẫn tương đối; ảnh MRI, file torrent cache và trạng thái download không đưa lên Git.
