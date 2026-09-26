# M1 — CNN nông

**Chủ sở hữu:** Thành viên 1. Notebook chính sẽ là `M1.ipynb` trong thư mục này; hiện chưa có code huấn luyện.

Đầu vào FLAIR `3×128×128` → Conv 3×3 `3→16` → Conv 3×3 `16→16` → Conv 3×3 `16→1` → logits WT `1×128×128`. Huấn luyện từ đầu trên split chung. Mã mô hình thuộc `src/brats_benchmark/models/`; notebook gọi API chung, vẽ đường học và phân tích mask.

Checklist, file bàn giao và tiêu chí nghiệm thu: [Detail_jobs.md](../Detail_jobs.md#thành-viên-1--m1-và-dữ-liệu-chung). Nguồn đọc: [Fully Convolutional Networks for Semantic Segmentation](https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html).
