# M2 — U-Net/ResNet-18 từ đầu

**Chủ sở hữu:** Thành viên 2. Notebook chính sẽ là `M2.ipynb` trong thư mục này; hiện chưa có code huấn luyện.

U-Net 2D dùng encoder ResNet-18 khởi tạo ngẫu nhiên, decoder có skip connection. M2 và M3 phải gọi **cùng factory**; chỉ giá trị `encoder_weights` và lịch train khác. Đầu vào FLAIR `3×128×128`, đầu ra logits WT `1×128×128`.

Checklist, file bàn giao và tiêu chí nghiệm thu: [Detail_jobs.md](../Detail_jobs.md#thành-viên-2--m2-và-huấn-luyện-chung). Nguồn đọc: [U-Net](https://arxiv.org/abs/1505.04597), [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).
