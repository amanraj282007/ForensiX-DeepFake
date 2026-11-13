"""
calibrate_robustness.py - 15-Condition Anti-Degradation Robustness Benchmark Script
ForensiX-DeepFake System

Evaluates the 5-model deepfake ensemble across 15 synthetic perturbation conditions:
- Clean baseline
- Gaussian blur (sigma = 0.5, 1.0, 2.0)
- JPEG compression (Quality = 85, 60, 35)
- Additive Gaussian noise (sigma = 5, 10, 20 out of 255)
- Multi-scale downscaling (192px, 112px, 56px)
- Brightness shifts (factor = 0.85x, 0.65x, 0.45x)

HISTORICAL REPRODUCIBILITY & WEIGHTS DISCLOSURE:
- robustness_results.csv contains reference benchmark outputs evaluated during development
  on the Stage 1 isolated validation split (1,400 images, drawn from train_metadata.csv seed 42)
  using the original Stage 1 validation-split checkpoints with probability-averaging and no TTA
  (matching Cell 14 of training_notebook.ipynb).
- Shipped model_weights.pt contains the final Stage 2 ensemble retrained on 100% of the 7,000 images.
  Evaluating model_weights.pt on validation images evaluates trained data (yielding inflated metrics).
  This script serves as the executable test harness demonstrating how the 15 perturbations are constructed
  (defaulting to inferencescript.py's clamped-logit + TTA mode; pass --blend prob --no_tta to match
  the Stage 1 benchmark mode).
"""

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EXTS = {".jpg", ".jpeg", ".png"}


def build_effnet_b0(dropout: float = 0.4) -> nn.Module:
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(m.classifier[1].in_features, 1)
    )
    return m


def build_convnext() -> nn.Module:
    m = models.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
    return m


def build_vit() -> nn.Module:
    m = models.vit_b_16(weights=None)
    m.heads.head = nn.Linear(m.heads.head.in_features, 1)
    return m


def build_swin() -> nn.Module:
    m = models.swin_t(weights=None)
    m.head = nn.Linear(m.head.in_features, 1)
    return m


BUILDERS = {
    "b0_corrupt": build_effnet_b0,
    "b0_base": build_effnet_b0,
    "convnext": build_convnext,
    "vit": build_vit,
    "swin": build_swin,
}


def get_perturbation_transform(condition: str):
    base_norm = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    
    if condition == "Clean":
        return transforms.Compose([transforms.Resize((224, 224))] + base_norm)
    
    if condition.startswith("Blur"):
        sigma = float(condition.split()[1])
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(sigma))),
            *base_norm
        ])
    
    if condition.startswith("JPEG"):
        q = int(condition.split()[1])
        def apply_jpeg(img):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(apply_jpeg),
            *base_norm
        ])
    
    if condition.startswith("Noise"):
        level = float(condition.split()[1])
        def apply_noise(tensor):
            noise = torch.randn_like(tensor) * (level / 255.0)
            return (tensor + noise).clamp(0.0, 1.0)
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(apply_noise),
            transforms.Normalize(MEAN, STD)
        ])
    
    if condition.startswith("Resolution"):
        sz = int(condition.split()[1])
        return transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.Resize((224, 224)),
            *base_norm
        ])
    
    if condition.startswith("Brightness"):
        factor = float(condition.split()[1])
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: ImageEnhance.Brightness(img).enhance(factor)),
            *base_norm
        ])
    
    raise ValueError(f"Unknown condition: {condition}")


class LabeledImageDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def load_dataset_items(data_dir: Path, val_split: float = 0.20, seed: int = 42):
    meta_path = data_dir / "train_metadata.csv"
    if meta_path.exists():
        print(f"[INFO] Using metadata split from {meta_path}")
        meta = pd.read_csv(meta_path)
        meta["image_id"] = meta["image_id"].astype(str).str.strip()
        paths = {}
        for folder in ["real", "fake"]:
            for sub in (data_dir / "train" / folder, data_dir / folder):
                if sub.is_dir():
                    for p in sub.iterdir():
                        if p.suffix.lower() in EXTS:
                            paths[p.stem] = p
        meta["path"] = meta["image_id"].map(paths)
        meta = meta.dropna(subset=["path"])
        if val_split > 0:
            _, val_df = train_test_split(meta, test_size=val_split, random_state=seed, stratify=meta["label"])
            return [(Path(row["path"]), int(row["label"])) for _, row in val_df.iterrows()]
        return [(Path(row["path"]), int(row["label"])) for _, row in meta.iterrows()]

    items = []
    for p in sorted(data_dir.rglob("*")):
        if p.suffix.lower() in EXTS:
            parent = p.parent.name.lower()
            if parent in {"real", "fake"}:
                items.append((p, 1 if parent == "fake" else 0))
    if val_split > 0 and len(items) > 1000:
        labels = [y for _, y in items]
        _, val_items = train_test_split(items, test_size=val_split, random_state=seed, stratify=labels)
        return val_items
    return items


def evaluate_ensemble(models_dict, dataloader, device, threshold: float = 0.50, use_tta: bool = True, blend: str = "logit"):
    all_scores = []
    all_labels = []
    start = time.time()

    use_amp = (device.type == "cuda")
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            x_flip = torch.flip(x, dims=[3]) if use_tta else None

            batch_scores = []
            for name, m in models_dict.items():
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    z = m(x).float().squeeze(1)
                    if use_tta:
                        z_flip = m(x_flip).float().squeeze(1)
                if blend == "logit":
                    s = 0.5 * (z + z_flip) if use_tta else z
                    s = torch.clamp(s, -8.0, 8.0)
                else:
                    p = torch.sigmoid(z)
                    s = 0.5 * (p + torch.sigmoid(z_flip)) if use_tta else p
                batch_scores.append(s.cpu().numpy())

            fused = np.mean(batch_scores, axis=0)
            all_scores.extend(fused)
            all_labels.extend(y.numpy())

    elapsed = time.time() - start
    arr = np.array(all_scores)
    probs = 1.0 / (1.0 + np.exp(-arr)) if blend == "logit" else np.clip(arr, 0.0, 1.0)
    preds = (probs >= threshold).astype(int)
    labels = np.array(all_labels)

    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    return auc, acc, f1, elapsed


CONDITIONS = [
    "Clean",
    "Blur 0.5", "Blur 1.0", "Blur 2.0",
    "JPEG 85", "JPEG 60", "JPEG 35",
    "Noise 5", "Noise 10", "Noise 20",
    "Resolution 192", "Resolution 112", "Resolution 56",
    "Brightness 0.85", "Brightness 0.65", "Brightness 0.45"
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate 15-condition robustness benchmark.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing dataset images or train_metadata.csv.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to packed model_weights.pt.")
    parser.add_argument("--output", type=Path, default=Path("robustness_results.csv"), help="Output CSV path.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--val_split", type=float, default=0.20, help="Validation fraction (0.0 to evaluate all).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_tta", action="store_true", help="Disable horizontal-flip TTA")
    parser.add_argument("--blend", choices=["logit", "prob"], default="logit", help="Ensemble blend mode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    items = load_dataset_items(args.data_dir, args.val_split, args.seed)
    print(f"[INFO] Loaded {len(items)} evaluation images.")
    if not items:
        raise ValueError(f"No labeled images found in {args.data_dir}")

    checkpoint = torch.load(args.weights, map_location="cpu")
    loaded_models = {}
    for name, builder_fn in BUILDERS.items():
        if name in checkpoint:
            m = builder_fn()
            m.load_state_dict(checkpoint[name], strict=True)
            m.to(device).eval()
            loaded_models[name] = m
            print(f"[INFO] Loaded model: {name}")
        else:
            raise KeyError(f"Missing model key {name} in {args.weights}")

    results = []
    clean_auc = None

    for cond in CONDITIONS:
        # Seed once per condition for per-image variability and reproducible evaluation
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        print(f"\n[RUN] Evaluating condition: {cond} ...", flush=True)
        tf = get_perturbation_transform(cond)
        ds = LabeledImageDataset(items, tf)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))
        
        auc, acc, f1, elapsed = evaluate_ensemble(
            loaded_models, loader, device, args.threshold, use_tta=not args.no_tta, blend=args.blend
        )
        if clean_auc is None:
            clean_auc = auc
        auc_drop = clean_auc - auc
        
        print(f"[{cond}] AUC: {auc:.4f} | Acc: {acc*100:.2f}% | F1: {f1:.4f} | Time: {elapsed:.2f}s | Drop: {auc_drop:+.4f}")
        results.append({
            "Condition": cond,
            "AUC_ROC": auc,
            "Accuracy": acc,
            "F1_Score": f1,
            "Time_Seconds": elapsed,
            "AUC_Drop": auc_drop
        })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\n[DONE] Saved benchmark results to {args.output}")


if __name__ == "__main__":
    main()
