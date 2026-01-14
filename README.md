# Few-Shot Medical Image Segmentation

Dự án nghiên cứu và đánh giá các phương pháp Few-Shot Segmentation cho ảnh y tế, tập trung vào bài toán phân đoạn nang buồng trứng (Ovarian Cysts).

## 📋 Tổng quan

Dự án này so sánh hiệu suất của 3 kiến trúc Few-Shot Segmentation khác nhau trên dataset y tế:

1. **SENet** - Squeeze-and-Excitation Network based Few-Shot Segmentation
2. **PANet** - Prototype Alignment Network  
3. **SSL-ALPNet** - Self-Supervised Adaptive Local Prototype Network

## 🗂️ Cấu trúc dự án

```
few_shot_model/
├── few-shot-segmentation/          # SENet implementation
├── PANet/                          # PANet implementation
├── Self-supervised-Fewshot-Medical-Image-Segmentation/  # SSL-ALPNet implementation
└── README.md                       # File này
```

---

## 📁 1. few-shot-segmentation/

**Mô tả:** Triển khai SENet-based Few-Shot Segmentation với các biến thể Squeeze-and-Excitation modules.

### Cấu trúc chính:

```
few-shot-segmentation/
├── few_shot_segmentor.py          # Model chính SENet
├── solver.py                       # Training loop và logic huấn luyện
├── run.py                          # Script chạy training/testing
├── settings.py                     # Cấu hình model và dataset
├── settings.ini                    # File config
│
├── datasets/                       # Danh sách train/test volumes
│   ├── train_volumes.txt
│   ├── test_volumes.txt
│   ├── eval_support.txt
│   └── eval_query.txt
│
├── utils/                          # Tiện ích
│   ├── data_utils.py              # Load và xử lý dữ liệu
│   ├── evaluator.py               # Đánh giá metrics
│   ├── evaluator_kshot.py         # K-shot evaluation
│   ├── common_utils.py            # Hàm tiện ích chung
│   └── preprocessor.py            # Tiền xử lý ảnh
│
├── other_experiments/              # Các thí nghiệm khác
│   ├── channel_sne_all/           # SNE ở tất cả layers (channel-wise)
│   ├── spatial_sne_all_*/         # SNE spatial attention
│   ├── channel_and_spatial_sne_all/  # Kết hợp cả hai
│   ├── shaban/                    # Baseline từ Shaban et al.
│   └── rakelly/                   # Baseline từ Rakelly et al.
│
├── saved_models/                   # Checkpoints
│
└── *.ipynb                         # Notebooks phân tích
    ├── Finetuning.ipynb
    ├── SEnet_OTU2D_inference.ipynb
    ├── universeg_OTU2d.ipynb
    └── universeg_analization_Ovatus_02-01.ipynb
```

### Chức năng:

- **Training:** `python run.py --mode train`
- **Testing:** `python run.py --mode test`
- **Inference:** Sử dụng notebooks như `SEnet_OTU2D_inference.ipynb`

### Đặc điểm:

- Hỗ trợ nhiều vị trí đặt SNE module (encoder, decoder, bottleneck, all)
- Hỗ trợ cả channel attention và spatial attention
- Fine-tuning trên domain-specific data

---

## 📁 2. PANet/

**Mô tả:** Triển khai PANet (Prototype Alignment Network) - một trong những baseline mạnh nhất cho Few-Shot Segmentation.

### Cấu trúc chính:

```
PANet/
├── train.py                        # Training script
├── test.py                         # Testing script
├── config.py                       # Cấu hình model
│
├── models/                         # Kiến trúc model
│   ├── fewshot.py                 # PANet model
│   └── vgg.py                     # VGG backbone
│
├── dataloaders/                    # Data loading
│   ├── customized.py              # Custom medical dataset loader
│   ├── pascal.py                  # PASCAL VOC loader
│   ├── coco.py                    # MS COCO loader
│   └── transforms.py              # Data augmentation
│
├── util/                           # Utilities
│   ├── utils.py
│   └── metric.py                  # Evaluation metrics
│
├── experiments/                    # Các file script chạy thí nghiệm
│   └── *.sh                       # Bash scripts cho các cấu hình khác nhau
│
├── pretrained_model/               # Pre-trained weights
│   └── vgg16-397923af.pth
│
├── test_panet_*.ipynb             # Notebooks test trên các dataset
└── panet_*_results.csv            # Kết quả đánh giá
```

### Chức năng:

**Training:**
```bash
python train.py --config config.py
```

**Testing:**
```bash
python test.py --config config.py --load <checkpoint_path>
```

**Jupyter Notebooks:**
- `test_panet_otu2d.ipynb` - Test trên OTU2D dataset
- `test_panet_ovatus.ipynb` - Test trên Ovatus dataset

### Đặc điểm:

- Sử dụng VGG16 backbone
- Prototype alignment mechanism
- Hỗ trợ 1-shot và 5-shot learning
- Pre-trained trên PASCAL VOC và MS COCO

---

## 📁 3. Self-supervised-Fewshot-Medical-Image-Segmentation/

**Mô tả:** Triển khai SSL-ALPNet - Few-Shot Segmentation với self-supervised learning và adaptive local prototypes.

### Cấu trúc chính:

```
Self-supervised-Fewshot-Medical-Image-Segmentation/
├── training.py                     # Training script
├── validation.py                   # Validation script
├── config_ssl_upload.py           # Configuration
│
├── models/                         # Model architecture
│   ├── grid_proto_fewshot.py      # Main SSL-ALPNet model
│   ├── alpmodule.py               # Adaptive Local Prototype module
│   └── backbone/
│       └── torchvision_backbones.py  # ResNet, DeepLabV3 backbones
│
├── dataloaders/                    # Data loading
│   ├── GenericSuperDatasetv2.py   # Generic dataset loader
│   ├── ManualAnnoDatasetv2.py     # Manual annotation loader
│   ├── dataset_utils.py
│   ├── image_transforms.py
│   └── niftiio.py                 # NIfTI file I/O
│
├── util/                           # Utilities
│   ├── utils.py
│   └── metric.py
│
├── data/                           # Dataset preparation
│   ├── CHAOST2/                   # Abdominal MRI dataset
│   └── SABS/                      # Abdominal CT dataset
│
├── examples/                       # Example scripts
│   ├── train_ssl_abdominal_*.sh
│   └── test_ssl_abdominal_*.sh
│
├── test_ssl_alpnet_*.ipynb        # Test notebooks
└── ssl_alpnet_*_results.csv       # Evaluation results
```

### Chức năng:

**Training:**
```bash
python training.py --config config_ssl_upload.py
```

**Testing:**
```bash
python validation.py --config config_ssl_upload.py --load <checkpoint>
```

**Jupyter Notebooks:**
- `test_ssl_alpnet_otu2d.ipynb` - Test trên OTU2D dataset
- `test_ssl_alpnet_ovatus.ipynb` - Test trên Ovatus dataset (hiện tại đang mở)

### Đặc điểm:

- Sử dụng ResNet101 + DeepLabV3 backbone
- Adaptive Local Prototype (ALP) module
- Grid-based prototype aggregation
- Self-supervised pre-training
- Hỗ trợ N-shot learning (N = 1, 2, 4, 8, 16, 32)

---

## 🏥 Dataset: Ovatus (Ovarian Cysts)

### Classes (6 loại nang buồng trứng):

| ID | Tên class            | Mô tả                    |
|----|----------------------|--------------------------|
| 0  | nang_da_thuy        | Nang đa buồng chứa dịch |
| 1  | nang_don_thuy       | Nang đơn buồng chứa dịch |
| 2  | nang_da_thuy_dac    | Nang đa buồng hỗn hợp   |
| 3  | nang_don_thuy_dac   | Nang đơn buồng hỗn hợp  |
| 4  | u_bi                | U lành tính             |
| 5  | u_dac               | U đặc                   |

### Cấu trúc dữ liệu:

```
DATA_ROOT/
├── patient_001/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── patient_002/
│   └── ...
└── mapping_normalized4.jsonl      # Annotations
```

### Format annotation (JSONL):

```json
{
  "patient_name": "patient_001",
  "images": [
    {
      "image_name": "image1.jpg",
      "imageWidth": 1920,
      "imageHeight": 1080,
      "labels": ["nang_da_thuy", "u_bi"],
      "points": [
        [[x1, y1], [x2, y2], ...],  // Polygon cho nang_da_thuy
        [[x1, y1], [x2, y2], ...]   // Polygon cho u_bi
      ]
    }
  ]
}
```

---

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt môi trường

```bash
# Tạo virtual environment
conda create -n fewshot python=3.8
conda activate fewshot

# Cài đặt dependencies cho từng project
cd few-shot-segmentation && pip install -r requirements.txt
cd ../PANet && pip install -r requirements.txt  
cd ../Self-supervised-Fewshot-Medical-Image-Segmentation && pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Đảm bảo có:
- Thư mục chứa ảnh (`DATA_ROOT`)
- File annotation JSONL (`ANNOT_PATH`)

### 3. Testing với Jupyter Notebooks

**Ví dụ: Test SSL-ALPNet trên Ovatus**

```bash
cd Self-supervised-Fewshot-Medical-Image-Segmentation
jupyter notebook test_ssl_alpnet_ovatus.ipynb
```

Notebook sẽ:
1. Load dataset và chia train/test theo patient
2. Load model (pre-trained hoặc random init)
3. Chạy evaluation với N-shot khác nhau (1, 2, 4, 8, 16, 32)
4. Tính metrics: Dice, IoU, Precision, Recall
5. Tạo visualization và lưu kết quả

### 4. So sánh kết quả

Sau khi chạy notebooks, các file CSV kết quả sẽ được tạo:

```
PANet/panet_ovatus_results.csv
Self-supervised-Fewshot-Medical-Image-Segmentation/ssl_alpnet_ovatus_by_N.csv
Self-supervised-Fewshot-Medical-Image-Segmentation/ssl_alpnet_ovatus_per_class_N8.csv
```

---

## 📊 Metrics đánh giá

Các metrics được sử dụng:

- **Dice Score:** $\text{Dice} = \frac{2|P \cap G|}{|P| + |G|}$
- **IoU (Intersection over Union):** $\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$
- **Precision:** $\text{Precision} = \frac{TP}{TP + FP}$
- **Recall:** $\text{Recall} = \frac{TP}{TP + FN}$

Trong đó:
- $P$: Predicted mask
- $G$: Ground truth mask
- $TP$: True Positive, $FP$: False Positive, $FN$: False Negative

---

## 🔬 Các thí nghiệm

### Few-Shot Learning scenarios:

1. **1-shot:** Model học từ 1 ảnh support duy nhất
2. **5-shot:** Model học từ 5 ảnh support
3. **N-shot:** Đánh giá với N = 1, 2, 4, 8, 16, 32 support images

### Evaluation protocols:

- **Per N-shot:** So sánh hiệu suất với số lượng support khác nhau
- **Per class:** Đánh giá riêng cho từng loại nang
- **Cross-patient:** Support và query từ các bệnh nhân khác nhau

---

## 📝 Files quan trọng

### Configuration files:
- `few-shot-segmentation/settings.ini` - SENet config
- `PANet/config.py` - PANet config
- `Self-supervised-Fewshot-Medical-Image-Segmentation/config_ssl_upload.py` - SSL-ALPNet config

### Model files:
- `few-shot-segmentation/few_shot_segmentor.py` - SENet model
- `PANet/models/fewshot.py` - PANet model
- `Self-supervised-Fewshot-Medical-Image-Segmentation/models/grid_proto_fewshot.py` - SSL-ALPNet model

### Evaluation notebooks:
- `test_ssl_alpnet_ovatus.ipynb` - Đánh giá SSL-ALPNet trên Ovatus (★ đang sử dụng)
- `test_panet_ovatus.ipynb` - Đánh giá PANet trên Ovatus
- `SEnet_OTU2D_inference.ipynb` - Đánh giá SENet

---

## 📚 Tài liệu tham khảo

### Papers:

1. **PANet:** Wang et al., "PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment", ICCV 2019
2. **SSL-ALPNet:** Hansen et al., "Self-supervised Pre-training for Few-shot Medical Image Segmentation", arXiv 2021
3. **SENet:** Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018

### Repositories:

- [PANet original](https://github.com/kaixin96/PANet)
- [SSL-ALPNet original](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation)

---

## 🛠️ Troubleshooting

### Lỗi thường gặp:

1. **CUDA out of memory:**
   - Giảm batch size
   - Giảm kích thước ảnh (RESIZE_TO)

2. **Module not found:**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **Dataset not found:**
   - Kiểm tra đường dẫn `DATA_ROOT` và `ANNOT_PATH` trong config

---

## 👥 Contributors

- Research team: Few-Shot Medical Image Segmentation
- Dataset: Ovatus - Ovarian Cysts Ultrasound Images

---

## 📄 License

Mỗi subproject có license riêng, xem file LICENSE trong từng thư mục.

---

## 📧 Contact

For questions or issues, please create an issue in the respective project folder.

---

**Last updated:** January 2026
