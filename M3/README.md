# M3 — U-Net/ResNet-18 transfer learning

**Chủ sở hữu:** Thành viên 3. Notebook chính sẽ là `M3.ipynb` trong thư mục này; hiện chưa có code huấn luyện.

Cùng kiến trúc U-Net/ResNet-18 với M2. Encoder dùng trọng số ImageNet; học decoder/head khi encoder đóng băng 5 epoch, sau đó mở `layer4` để fine-tune tối đa 25 epoch. M3 không dùng thêm modality hoặc ca bệnh khác.

Checklist, file bàn giao và tiêu chí nghiệm thu: [Detail_jobs.md](../Detail_jobs.md#thành-viên-3--m3-và-tích-hợp-benchmark). Nguồn đọc: [PyTorch Transfer Learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), [Segmentation Models PyTorch U-Net](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/docs/quickstart.rst).
