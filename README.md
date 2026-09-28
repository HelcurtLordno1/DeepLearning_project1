# BraTS 2015 Light Benchmark — M1 / M2 / M3

**Bài toán:** phân đoạn *whole tumor* (WT) nhị phân từ lát MRI FLAIR 2D. M1 là CNN encoder–decoder nhỏ tự xây; M2 được đặc tả là U-Net sâu với **toàn bộ encoder/decoder tự viết từ các lớp PyTorch cơ bản và học từ đầu**; M3 dùng encoder pretrained để transfer learning. Đây là benchmark nội bộ trên **cùng 100 ca BraTS 2015**, không phải điểm chính thức của challenge. Vì M2/M3 được phép khác kiến trúc, không quy riêng chênh lệch điểm cho pretrained weights.

**Trạng thái kiểm tra:** 201/201 file raw đúng SHA-256; cache đã tạo đủ 100 ca. Cả ba Mx đã có full train/validation/test seed 42 trên Windows `.venv`/CUDA: test mean Dice theo 15 bệnh nhân lần lượt **M1 0.7997**, **M2 0.8074**, **M3 0.8120**. Cả ba notebook đã lưu output Run All seed 42 để trình bày; artifact chi tiết nằm ở `runs/m1/42/`, `runs/m2/42/`, `runs/m3/42/`. Seed 123/2026 chưa có kết quả. Xem [báo cáo benchmark](Benchmark_evaluate.md) để đọc phân tích và giới hạn so sánh.

## Cấu trúc đã tạo

```text
Project_midterm/
├── AGENTS.md, README.md, Project_structure.md, Detail_jobs.md, Benchmark_evaluate.md
├── requirements.txt              # thư viện bên ngoài; không có package mã nội bộ
├── .venv/                       # Python Windows; không commit
├── M1/
│   ├── M1.ipynb                 # code và Markdown đầy đủ, luồng chính
│   ├── M1.py                    # cùng quy trình, chạy qua PowerShell
│   └── README.md
├── M2/
│   ├── M2.ipynb
│   ├── M2.py
│   └── README.md
├── M3/
│   ├── M3.ipynb
│   ├── M3.py
│   └── README.md
├── data/
│   ├── source_manifest.json     # khóa 100 ca và SHA-1 nguồn
│   ├── splits_v1.csv           # split cố định 70/15/15 theo case_id
│   ├── file_sha256.csv         # checksum 201 file
│   ├── raw/BRATS2015/          # 200 MRI + giấy phép; không commit
│   └── processed/             # cache chung; không commit
├── runs/                       # checkpoint, CSV, hình của mỗi Mx; không commit
├── reports/                    # báo cáo nhóm
└── slides/                     # tài liệu người dùng đang có
```

**Không có `scripts/`, `src/`, notebook chung hoặc package nội bộ.** Mỗi notebook chứa trực tiếp toàn bộ mã của mức đó: kiểm dữ liệu → tiền xử lý → model → loss/metric → train/validation → checkpoint → test → hình. `Mx.py` tự chứa cùng quy trình; không import mã từ notebook hay folder khác. Các thư viện ngoài (`torch`, `torchvision`, `SimpleITK`, `numpy`, `matplotlib`, `torchinfo` để in bảng kiến trúc cả ba model) được cài trong `.venv` chung. Xem [AGENTS.md](AGENTS.md) và [chia việc](Detail_jobs.md).

## Dữ liệu và kiểm tra đã thực hiện

`data/raw/BRATS2015/training/{HGG,LGG}/<patient_id>/...` đã có **200 file `.mha` FLAIR/OT và 1 file giấy phép**. Ngày 28/09/2026, PowerShell Windows kiểm đủ **201/201 file**, đúng kích thước và **SHA-256 khớp `data/file_sha256.csv`**; vì vậy máy này **không cần tải lại**. `data/splits_v1.csv` có 80 HGG + 20 LGG, chia 70 train / 15 validation / 15 test theo `case_id = grade/patient_id`.

Nguồn là [Academic Torrents BraTS 2015](https://academictorrents.com/details/c4f39a0a8e46e8d2174b8a8a81b9887150f44d50), tải các file đã chọn từ [web seed Archive.org](https://archive.org/metadata/BRATS2015). Bản sao HTTPS có 100 HGG và 54 LGG hoàn chỉnh; cohort 100 ca đã được khóa trong `data/source_manifest.json`. Không đưa MRI gốc lên Git.

### Tải lại trên máy Windows khác nếu `data/raw/` trống

Chạy **PowerShell Windows** từ root. Dòng `Set-Location` dưới đây là đường dẫn trên máy hiện tại; khi clone ở vị trí khác, thay bằng đường dẫn root của bản clone. Khối lệnh dùng đúng danh sách file và SHA-1 đã khóa; file hợp lệ được bỏ qua, file thiếu được tải bằng HTTPS vào đúng đường dẫn. `Invoke-WebRequest` là lệnh có sẵn trong PowerShell 5.1. Tôi đã thử tải lại một file giấy phép bị xóa tạm thời, sau đó xác minh 201/201 SHA-256 bằng chính khối lệnh này.

```powershell
Set-Location 'D:\Desktop_informations\SGK năm 4\SGK kì 1 năm 4\DeepLearning - Ngân\Project\Project_midterm'
$source = Get-Content .\data\source_manifest.json -Raw | ConvertFrom-Json
$items = @($source.license_file)
foreach ($case in $source.cases) { $items += $case.flair; $items += $case.mask }
foreach ($item in $items) {
    $target = Join-Path .\data\raw\BRATS2015 ($item.path.Replace('/', [IO.Path]::DirectorySeparatorChar))
    if ((Test-Path -LiteralPath $target) -and
        (Get-Item -LiteralPath $target).Length -eq [int64]$item.bytes -and
        (Get-FileHash -LiteralPath $target -Algorithm SHA1).Hash.ToLowerInvariant() -eq $item.sha1) { continue }
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    $url = 'https://archive.org/download/BRATS2015/BRATS2015/' + $item.path
    for ($attempt = 1; $attempt -le 4; $attempt++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $target
            if ((Get-Item -LiteralPath $target).Length -ne [int64]$item.bytes -or
                (Get-FileHash -LiteralPath $target -Algorithm SHA1).Hash.ToLowerInvariant() -ne $item.sha1) {
                throw "Sai kích thước hoặc SHA-1: $target"
            }
            break
        } catch {
            Remove-Item -LiteralPath $target -ErrorAction SilentlyContinue
            if ($attempt -eq 4) { throw }
            Start-Sleep -Seconds (2 * $attempt)
        }
    }
}
$checks = Import-Csv .\data\file_sha256.csv
$bad = @($checks | Where-Object {
    $path = Join-Path .\data\raw\BRATS2015 ($_.path.Replace('/', [IO.Path]::DirectorySeparatorChar))
    -not (Test-Path -LiteralPath $path) -or
    (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $_.sha256
})
if ($bad.Count -ne 0) { throw "$($bad.Count) file sai checksum" }
"Đã kiểm tra $($checks.Count)/$($checks.Count) file."
```

## Cài môi trường chung trên Windows

Máy hiện có `.venv` tạo bằng **Python Windows 3.12**. Nếu clone sang máy khác, chạy trong PowerShell tại root:

```powershell
py -3.12 -m venv .venv
$ProjectPython = '.\.venv\Scripts\python.exe'
& $ProjectPython -m pip install --upgrade pip
# Cài torch + torchvision bản CUDA phù hợp theo https://pytorch.org/get-started/locally/
& $ProjectPython -m pip install -r requirements.txt
& $ProjectPython -m ipykernel install --user --name brats2015-midterm --display-name 'BraTS 2015 (.venv)'
& $ProjectPython -c "import torch; print(torch.cuda.is_available())"
```

Lấy lệnh cài PyTorch CUDA hiện hành từ [PyTorch Start Locally](https://pytorch.org/get-started/locally/); chọn **Windows / Pip / CUDA** trước bước `requirements.txt`. Trong VS Code/Jupyter, chọn kernel **BraTS 2015 (.venv)** và kiểm `sys.executable` là `.venv\Scripts\python.exe`. Không tạo hoặc dùng môi trường WSL.

**Môi trường đã thử trên máy này:** Python 3.12.10, torch 2.11.0+cu128, torchvision 0.26.0+cu128, SimpleITK 2.5.6, matplotlib 3.11.2, torchinfo 1.8.0, ipykernel 7.3.0; `torch.cuda.is_available()` trả `True`. Trọng số ImageNet của M3 đã tải được qua `torchvision`.

## Chạy mỗi Mx theo hai cách

**Cách chính — notebook:** mở `Mx/Mx.ipynb` trên Windows, chọn kernel `.venv`, rồi **Restart Kernel + Run All**. Ô đầu chỉ chọn `SEED = 42` (sau đó 123, 2026). Notebook dùng đủ 70 ca train/15 ca validation, in log từng epoch, điểm Dice/IoU/precision/recall/pixel accuracy, biểu đồ và ảnh; sau đó **đánh giá 15 ca test của chính Mx trong cùng lượt chạy**. Checkpoint/ngưỡng luôn chọn bằng validation, không chọn lại theo test. Cả ba notebook trên Git đã lưu output seed 42; Run All sẽ tạo lại khi đổi seed hoặc chạy trên máy khác. Mỗi Mx chạy độc lập, không chờ checkpoint của Mx khác.

**Cách hai — file `.py` trong chính folder Mx:** từ PowerShell Windows ở root:

```powershell
$ProjectPython = '.\.venv\Scripts\python.exe'
& $ProjectPython .\M1\M1.py --smoke
& $ProjectPython .\M2\M2.py --smoke
& $ProjectPython .\M3\M3.py --smoke
# Huấn luyện mỗi Mx (tự đánh giá test của chính model sau khi train):
& $ProjectPython .\M1\M1.py --train --seed 42
& $ProjectPython .\M2\M2.py --train --seed 42
& $ProjectPython .\M3\M3.py --train --seed 42
# Có thể đánh giá lại checkpoint riêng của M1 khi cần:
& $ProjectPython .\M1\M1.py --test --seed 42
```

`--prepare` tạo cache cho 100 ca trước khi train; các Mx không lặp kiểm SHA-256 trong mỗi lần Run All vì dữ liệu raw đã được xác minh trong bước chuẩn bị ở trên. Artifacts nằm ở `runs/smoke/<mx>/<seed>/` hoặc `runs/<mx>/<seed>/`: `best.pt`, `history.csv`, `metrics_val.csv`, `preview.png`, `config.json`; Cả ba có `learning_curve.png`, và train đầy đủ tạo `metrics_test.csv`. Các Mx dùng cùng `data/splits_v1.csv` và cache `data/processed/flair_wt_v1/`.

## Kiến trúc và benchmark

```mermaid
flowchart LR
    A["BraTS 2015: FLAIR + OT"] --> B["100 case_id; split 70/15/15"]
    B --> C["32 lát/ca; 128×128; WT nhị phân"]
    C --> D["M1: CNN nhỏ tự xây, random init"]
    C --> E["M2: U-Net sâu tự xây, random init"]
    C --> F["M3: pretrained encoder + decoder"]
    D --> G["Validation: checkpoint + threshold"]
    E --> G
    F --> G
    G --> H["Mỗi Mx test độc lập sau khi khóa validation; tổng hợp CSV về sau"]
```

M1 dùng năm `ConvBlock` tự viết, hai MaxPool và hai bilinear upsample, không skip. M2 dùng `DoubleConv`/`DownBlock`/`UpBlock` tự viết với bốn skip; **không gọi built-in U-Net/ResNet, kể cả `weights=None`**. M3 được dùng pretrained encoder. Mỗi Mx huấn luyện một seed bằng **một loại AdamW**, có thể giảm learning rate trong cùng run và dừng sớm trên validation; không chạy vòng thử nhiều optimizer. Ba seed cố định để báo độ ổn định sau khi chốt cấu hình. Cả ba dùng chung loss `0.5 BCEWithLogits + 0.5 soft Dice`, ImageNet normalization, batch 16 và trần 30 epoch. Xem [thiết kế chi tiết](Project_structure.md) và [M1/README.md](M1/README.md).

**Phần cứng:** i7-12800H, RTX A4500 Laptop 16 GiB VRAM, khoảng 23 GiB RAM. M1/M2/M3 seed 42 chạy lần lượt 26/24/15 epoch trong khoảng **138/251/87 giây**, peak GPU memory PyTorch **297.548.288/694.520.320/442.448.384 byte**. Điểm smoke không dùng làm kết quả; các seed còn lại vẫn cần chạy để đánh giá độ ổn định.
