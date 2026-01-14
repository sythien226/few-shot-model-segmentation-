"""
Test PANet với OTU 2D Dataset
Tham khảo cách xử lý dữ liệu từ UniverSeg notebook
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# Import PANet model
from models.fewshot import FewShotSeg

# ================== CONFIG ==================
DATA_ROOT = "/thiends/hdd2t/UniverSeg/OTU_2D"
TRAIN_IMAGES = os.path.join(DATA_ROOT, "train1/Image/")
TRAIN_LABELS = os.path.join(DATA_ROOT, "train1/Label/")
VAL_IMAGES = os.path.join(DATA_ROOT, "validation1/Image/")
VAL_LABELS = os.path.join(DATA_ROOT, "validation1/Label/")
TRAIN_TXT = os.path.join(DATA_ROOT, "train.txt")
VAL_TXT = os.path.join(DATA_ROOT, "val.txt")
TRAIN_CLS = os.path.join(DATA_ROOT, "train_cls.txt")
VAL_CLS = os.path.join(DATA_ROOT, "val_cls.txt")

# PANet config
RESIZE_TO = (417, 417)  # PANet sử dụng 417x417
NUM_CLASSES = 8
LABEL_NAMES = [f"Class {i}" for i in range(NUM_CLASSES)]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Evaluation config
N_SHOTS_LIST = [1, 5]  # 1-shot và 5-shot
N_EVAL_SAMPLES = 100   # Số ảnh test mỗi class
SEED = 1234

# ================== UTILS ==================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_cls_labels(filepath):
    """Load class labels từ file txt"""
    labels = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                filename = parts[0].replace('.JPG', '')
                cls = int(parts[1])
                labels[filename] = cls
    return labels

def process_image(image_path, resize_to):
    """Load và preprocess ảnh cho PANet"""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(resize_to, Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        
        # Normalize theo ImageNet mean/std (PANet sử dụng VGG pretrained)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        return np.transpose(img, (2, 0, 1))  # HWC -> CHW
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def process_mask(mask_path, resize_to):
    """Load mask và binary hóa"""
    try:
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize(resize_to, Image.NEAREST)
        mask = np.array(mask, dtype=np.float32)
        # Binary hóa: pixel > 0 → 1.0
        mask = (mask > 0).astype(np.float32)
        return mask
    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        return None

# ================== DATASET ==================
class OTU2DDataset:
    """Dataset cho OTU 2D compatible với PANet"""
    
    def __init__(self, images_dir, labels_dir, ids_file, cls_labels, resize_to=RESIZE_TO):
        self.samples = []
        self.samples_by_class = defaultdict(list)
        self.cls_labels = cls_labels
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.resize_to = resize_to

        print(f"\nLoading OTU2DDataset from {images_dir}...")

        with open(ids_file, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]

        for id_ in ids:
            img_name = f"{id_}.JPG"
            mask_name = f"{id_}.PNG"
            img_path = os.path.join(images_dir, img_name)
            mask_path = os.path.join(labels_dir, mask_name)

            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                continue

            cls = self.cls_labels.get(id_, None)
            if cls is None:
                continue

            img = process_image(img_path, resize_to)
            if img is None:
                continue

            mask = process_mask(mask_path, resize_to)
            if mask is None:
                continue

            if np.sum(mask) < 10:  # Skip nếu mask quá nhỏ
                continue

            sample = {
                'image': img,
                'mask': mask,
                'class_id': cls,
                'img_path': img_path
            }
            self.samples.append(sample)
            self.samples_by_class[cls].append(len(self.samples) - 1)

        print(f"Loaded {len(self.samples)} valid samples")
        for c in sorted(self.samples_by_class.keys()):
            print(f"  Class {c}: {len(self.samples_by_class[c])} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def get_samples_by_class(self, class_id):
        """Lấy tất cả samples của một class"""
        indices = self.samples_by_class.get(class_id, [])
        return [self.samples[i] for i in indices]

# ================== PANET INFERENCE ==================
class PANetInference:
    """Wrapper để inference PANet với OTU 2D"""
    
    def __init__(self, model_path=None, align=True):
        self.device = DEVICE
        self.align = align
        
        # Load model
        cfg = {'align': align}
        self.model = FewShotSeg(
            pretrained_path='./pretrained_model/vgg16-397923af.pth',
            cfg=cfg
        )
        self.model = nn.DataParallel(self.model.cuda(), device_ids=[0])
        
        if model_path and os.path.exists(model_path):
            print(f"Loading pretrained weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            print("Using model without pretrained few-shot weights (only VGG backbone)")
        
        self.model.eval()
    
    def predict(self, support_images, support_fg_masks, support_bg_masks, query_image):
        """
        Inference PANet
        
        Args:
            support_images: list of [C, H, W] numpy arrays
            support_fg_masks: list of [H, W] numpy arrays (foreground masks)
            support_bg_masks: list of [H, W] numpy arrays (background masks)
            query_image: [C, H, W] numpy array
            
        Returns:
            prediction: [H, W] numpy array with class predictions
        """
        # Convert to tensors
        # Support: way x shot x [B x C x H x W]
        supp_imgs = [[torch.from_numpy(img).float().unsqueeze(0).to(self.device) 
                      for img in support_images]]
        
        # Masks: way x shot x [B x H x W]
        supp_fg = [[torch.from_numpy(mask).float().unsqueeze(0).to(self.device) 
                    for mask in support_fg_masks]]
        supp_bg = [[torch.from_numpy(mask).float().unsqueeze(0).to(self.device) 
                    for mask in support_bg_masks]]
        
        # Query: N x [B x C x H x W]
        qry_imgs = [torch.from_numpy(query_image).float().unsqueeze(0).to(self.device)]
        
        with torch.no_grad():
            output, _ = self.model(supp_imgs, supp_fg, supp_bg, qry_imgs)
            pred = output.argmax(dim=1)[0].cpu().numpy()
        
        return pred

# ================== METRICS ==================
def compute_metrics(pred, gt, threshold=0.5):
    """Tính Dice, IoU, Precision, Recall"""
    pred_bin = (pred > threshold).astype(np.float32)
    gt_bin = (gt > threshold).astype(np.float32)
    
    TP = np.sum(pred_bin * gt_bin)
    FP = np.sum(pred_bin * (1 - gt_bin))
    FN = np.sum((1 - pred_bin) * gt_bin)
    
    smooth = 1e-6
    dice = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
    iou = (TP + smooth) / (TP + FP + FN + smooth)
    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall
    }

# ================== EVALUATION ==================
def evaluate_panet_on_otu2d(model, support_pool, test_set, n_shots=1, n_eval=100):
    """
    Đánh giá PANet trên OTU 2D
    
    Args:
        model: PANetInference instance
        support_pool: OTU2DDataset cho support images
        test_set: OTU2DDataset cho test images  
        n_shots: số lượng support images
        n_eval: số ảnh test mỗi class
    """
    results_per_class = defaultdict(lambda: {'dice': [], 'iou': [], 'precision': [], 'recall': []})
    
    # Lấy các class có đủ dữ liệu
    active_classes = [c for c in range(NUM_CLASSES) 
                      if len(support_pool.samples_by_class[c]) >= n_shots 
                      and len(test_set.samples_by_class[c]) > 0]
    
    print(f"\nEvaluating with {n_shots}-shot on {len(active_classes)} classes...")
    
    for class_id in tqdm(active_classes, desc=f"{n_shots}-shot evaluation"):
        # Lấy support samples
        support_indices = support_pool.samples_by_class[class_id]
        
        # Lấy test samples
        test_indices = test_set.samples_by_class[class_id]
        n_test = min(n_eval, len(test_indices))
        
        for test_idx in test_indices[:n_test]:
            test_sample = test_set[test_idx]
            query_image = test_sample['image']
            query_gt = test_sample['mask']
            
            # Random chọn n_shots support images (không trùng với query)
            available_supports = [i for i in support_indices]
            random.shuffle(available_supports)
            selected_supports = available_supports[:n_shots]
            
            support_images = []
            support_fg_masks = []
            support_bg_masks = []
            
            for sup_idx in selected_supports:
                sup_sample = support_pool[sup_idx]
                support_images.append(sup_sample['image'])
                fg_mask = sup_sample['mask']
                bg_mask = 1.0 - fg_mask  # Background = inverse of foreground
                support_fg_masks.append(fg_mask)
                support_bg_masks.append(bg_mask)
            
            # Predict
            pred = model.predict(support_images, support_fg_masks, support_bg_masks, query_image)
            
            # pred có giá trị 0 (background) hoặc 1 (foreground)
            pred_binary = (pred == 1).astype(np.float32)
            
            # Compute metrics
            metrics = compute_metrics(pred_binary, query_gt)
            for key, value in metrics.items():
                results_per_class[class_id][key].append(value)
    
    return results_per_class

def print_results(results_per_class, n_shots):
    """In kết quả đánh giá"""
    print("\n" + "=" * 90)
    print(f"KẾT QUẢ ĐÁNH GIÁ PANET - {n_shots}-SHOT")
    print("=" * 90)
    
    print(f"\n{'Class':>6} | {'Name':>12} | {'#Samples':>8} | {'Dice':>12} | {'IoU':>12} | {'Precision':>12} | {'Recall':>12}")
    print("-" * 90)
    
    all_dice, all_iou, all_prec, all_rec = [], [], [], []
    
    for class_id in sorted(results_per_class.keys()):
        results = results_per_class[class_id]
        n_samples = len(results['dice'])
        
        if n_samples == 0:
            continue
        
        dice_mean = np.mean(results['dice'])
        dice_std = np.std(results['dice'])
        iou_mean = np.mean(results['iou'])
        iou_std = np.std(results['iou'])
        prec_mean = np.mean(results['precision'])
        prec_std = np.std(results['precision'])
        rec_mean = np.mean(results['recall'])
        rec_std = np.std(results['recall'])
        
        print(f"{class_id:>6} | {LABEL_NAMES[class_id]:>12} | {n_samples:>8} | "
              f"{dice_mean:.4f}±{dice_std:.3f} | {iou_mean:.4f}±{iou_std:.3f} | "
              f"{prec_mean:.4f}±{prec_std:.3f} | {rec_mean:.4f}±{rec_std:.3f}")
        
        all_dice.extend(results['dice'])
        all_iou.extend(results['iou'])
        all_prec.extend(results['precision'])
        all_rec.extend(results['recall'])
    
    print("-" * 90)
    print(f"{'MEAN':>6} | {'':>12} | {len(all_dice):>8} | "
          f"{np.mean(all_dice):.4f}±{np.std(all_dice):.3f} | {np.mean(all_iou):.4f}±{np.std(all_iou):.3f} | "
          f"{np.mean(all_prec):.4f}±{np.std(all_prec):.3f} | {np.mean(all_rec):.4f}±{np.std(all_rec):.3f}")
    
    return {
        'mean_dice': np.mean(all_dice),
        'mean_iou': np.mean(all_iou),
        'mean_precision': np.mean(all_prec),
        'mean_recall': np.mean(all_rec)
    }

def visualize_predictions(model, support_pool, test_set, n_shots=1, n_samples=3, save_path=None):
    """Visualize một số predictions"""
    fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4*n_samples))
    
    active_classes = [c for c in range(NUM_CLASSES) 
                      if len(support_pool.samples_by_class[c]) >= n_shots 
                      and len(test_set.samples_by_class[c]) > 0]
    
    for row in range(n_samples):
        # Random chọn một class
        class_id = random.choice(active_classes)
        
        # Lấy test sample
        test_idx = random.choice(test_set.samples_by_class[class_id])
        test_sample = test_set[test_idx]
        query_image = test_sample['image']
        query_gt = test_sample['mask']
        
        # Lấy support samples
        support_indices = support_pool.samples_by_class[class_id]
        selected_supports = random.sample(support_indices, min(n_shots, len(support_indices)))
        
        support_images = []
        support_fg_masks = []
        support_bg_masks = []
        
        for sup_idx in selected_supports:
            sup_sample = support_pool[sup_idx]
            support_images.append(sup_sample['image'])
            support_fg_masks.append(sup_sample['mask'])
            support_bg_masks.append(1.0 - sup_sample['mask'])
        
        # Predict
        pred = model.predict(support_images, support_fg_masks, support_bg_masks, query_image)
        pred_binary = (pred == 1).astype(np.float32)
        
        # Compute metrics
        metrics = compute_metrics(pred_binary, query_gt)
        
        # Denormalize image for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        query_display = query_image.transpose(1, 2, 0) * std + mean
        query_display = np.clip(query_display, 0, 1)
        
        support_display = support_images[0].transpose(1, 2, 0) * std + mean
        support_display = np.clip(support_display, 0, 1)
        
        # Plot
        axes[row, 0].imshow(support_display)
        axes[row, 0].set_title(f'Support Image\n(Class {class_id})', fontsize=10)
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(support_fg_masks[0], cmap='gray')
        axes[row, 1].set_title('Support Mask', fontsize=10)
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(query_display)
        axes[row, 2].set_title('Query Image', fontsize=10)
        axes[row, 2].axis('off')
        
        axes[row, 3].imshow(query_gt, cmap='gray')
        axes[row, 3].set_title('Ground Truth', fontsize=10)
        axes[row, 3].axis('off')
        
        axes[row, 4].imshow(pred_binary, cmap='gray')
        axes[row, 4].set_title(f'Prediction\nDice={metrics["dice"]:.3f}', fontsize=10)
        axes[row, 4].axis('off')
    
    plt.suptitle(f'PANet {n_shots}-shot Predictions on OTU 2D', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

# ================== MAIN ==================
def main():
    print("=" * 70)
    print("TEST PANET VỚI OTU 2D DATASET")
    print("=" * 70)
    
    set_seed(SEED)
    
    # Load class labels
    print("\n[1] Loading class labels...")
    train_cls_labels = load_cls_labels(TRAIN_CLS)
    val_cls_labels = load_cls_labels(VAL_CLS)
    
    # Load datasets
    print("\n[2] Loading datasets...")
    support_pool = OTU2DDataset(TRAIN_IMAGES, TRAIN_LABELS, TRAIN_TXT, train_cls_labels, RESIZE_TO)
    test_set = OTU2DDataset(VAL_IMAGES, VAL_LABELS, VAL_TXT, val_cls_labels, RESIZE_TO)
    
    # Load model
    print("\n[3] Loading PANet model...")
    # Bạn có thể thay đổi model_path nếu có pretrained weights
    # Ví dụ: model_path = './runs/PANet_VOC_sets_0_1way_1shot_[train]/1/snapshots/30000.pth'
    model_path = None  # Sử dụng model chưa train (chỉ có VGG backbone)
    
    panet = PANetInference(model_path=model_path, align=True)
    
    # Evaluate
    print("\n[4] Evaluating...")
    all_results = {}
    
    for n_shots in N_SHOTS_LIST:
        results = evaluate_panet_on_otu2d(
            panet, support_pool, test_set, 
            n_shots=n_shots, 
            n_eval=N_EVAL_SAMPLES
        )
        summary = print_results(results, n_shots)
        all_results[n_shots] = summary
    
    # Summary comparison
    print("\n" + "=" * 50)
    print("SO SÁNH TỔNG HỢP")
    print("=" * 50)
    print(f"\n{'N-shot':>8} | {'Dice':>10} | {'IoU':>10} | {'Precision':>10} | {'Recall':>10}")
    print("-" * 55)
    for n_shots, summary in all_results.items():
        print(f"{n_shots:>8} | {summary['mean_dice']:.4f}     | {summary['mean_iou']:.4f}     | "
              f"{summary['mean_precision']:.4f}     | {summary['mean_recall']:.4f}")
    
    # Visualize
    print("\n[5] Visualizing predictions...")
    for n_shots in N_SHOTS_LIST:
        visualize_predictions(
            panet, support_pool, test_set, 
            n_shots=n_shots, 
            n_samples=3,
            save_path=f'panet_otu2d_{n_shots}shot_vis.png'
        )
    
    print("\n✅ HOÀN TẤT!")

if __name__ == '__main__':
    main()
