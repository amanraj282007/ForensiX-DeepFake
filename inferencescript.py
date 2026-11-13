"""
inferencescript.py - ForensiX-DeepFake Inference Pipeline

Ensemble Architecture:
    - EfficientNet-B0 (b0_corrupt, damage augmentation)
    - EfficientNet-B0 (b0_base, light augmentation)
    - ConvNeXt-Tiny (convnext, modern depthwise CNN)
    - ViT-B/16 (vit, global vision transformer)
    - Swin-T (swin, shifted-window hierarchical transformer)

Standard execution:
    python inferencescript.py --test_dir ./test/images/ --weights ./model_weights.pt --output predictions.csv

Features:
    - Logit-level (default) or probability-level ensembling
    - Test-Time Augmentation (TTA horizontal flip averaging enabled by default)
    - Automatic GPU Mixed Precision (AMP FP16 enabled by default on CUDA) or CPU execution
    - Fully self-contained (zero external network access or online weight downloads)
    - Output schema: image_id (string), label (0 or 1), confidence (float in [0.0, 1.0])
"""

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGE_SIZE = 224
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


def build_effnet_b3(dropout: float = 0.3) -> nn.Module:
    m = models.efficientnet_b3(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(m.classifier[1].in_features, 1)
    )
    return m


BUILDERS = {
    "b0_corrupt": build_effnet_b0,
    "b0_base": build_effnet_b0,
    "convnext": build_convnext,
    "convnext_tiny": build_convnext,
    "b3": build_effnet_b3,
    "efficientnet_b3": build_effnet_b3,
    "vit": build_vit,
    "vit_b_16": build_vit,
    "swin": build_swin,
    "swin_t": build_swin,
    "efficientnet_b0": build_effnet_b0,
}


class TestDataset(Dataset):
    """Dataset reader for flat folder of test face images."""

    def __init__(self, image_dir: str):
        folder = Path(image_dir)
        if not folder.is_dir():
            raise SystemExit(f"[ERROR] Directory does not exist: '{image_dir}'")

        self.paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in EXTS])
        if not self.paths:
            raise SystemExit(f"[ERROR] No valid images (.jpg, .jpeg, .png) found in '{image_dir}'")

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), path.stem


def load_models(weights_path: str, fallback_model: str, device: torch.device, exclude: List[str] = None) -> List[Tuple[str, nn.Module]]:
    """Loads ensemble weights dictionary into instantiated models."""
    checkpoint = torch.load(weights_path, map_location="cpu")
    exclude = [e.lower() for e in (exclude or [])]

    members = []
    if isinstance(checkpoint, dict) and "models" in checkpoint:
        for mem in checkpoint["models"]:
            members.append((mem["name"], mem["state_dict"]))
    elif isinstance(checkpoint, dict) and "name" in checkpoint and "state_dict" in checkpoint:
        members.append((checkpoint["name"], checkpoint["state_dict"]))
    elif isinstance(checkpoint, dict):
        has_known_keys = any(k in checkpoint for k in BUILDERS.keys())
        if has_known_keys:
            for k, v in checkpoint.items():
                if k in BUILDERS:
                    members.append((k, v))
        elif fallback_model and fallback_model in BUILDERS:
            members.append((fallback_model, checkpoint))
        else:
            raise SystemExit(f"[ERROR] Unknown checkpoint keys: {list(checkpoint.keys())[:5]}. "
                             f"Please specify --model with one of: {list(BUILDERS.keys())}")
    else:
        raise SystemExit(f"[ERROR] Unrecognized model weights format in {weights_path}")

    loaded_nets = []
    for name, state_dict in members:
        if name.lower() in exclude:
            print(f"[INFO] Skipping pruned model: '{name}'")
            continue
        builder = BUILDERS.get(name)
        if builder is None:
            raise SystemExit(f"[ERROR] No builder found for model member '{name}'.")
        net = builder()
        net.load_state_dict(state_dict, strict=True)
        net.to(device).eval()
        loaded_nets.append((name, net))

    names = [name for name, _ in loaded_nets]
    print(f"[INFO] Loaded {len(loaded_nets)} model(s): {names} on {device}")
    return loaded_nets


@torch.no_grad()
def run_inference(
    nets: List[Tuple[str, nn.Module]],
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
    use_tta: bool,
    blend: str = "logit"
) -> Tuple[List[str], np.ndarray]:
    """Runs batched inference with horizontal-flip TTA and logit/probability blending."""
    image_ids = []
    probs_list = []
    use_amp = use_fp16 and device.type == "cuda"

    for images, batch_ids in loader:
        images = images.to(device, non_blocking=True)
        if use_tta:
            images_flipped = torch.flip(images, dims=[3])

        total_batch_score = 0.0
        for _, net in nets:
            with torch.autocast(device_type=device.type, enabled=use_amp):
                z = net(images).float().squeeze(1)
                if use_tta:
                    z_flip = net(images_flipped).float().squeeze(1)

            if blend == "logit":
                score = (z + z_flip) / 2.0 if use_tta else z
                score = torch.clamp(score, min=-8.0, max=8.0)
            else:
                p_orig = torch.sigmoid(z)
                p_flip = torch.sigmoid(z_flip) if use_tta else p_orig
                score = (p_orig + p_flip) / 2.0

            total_batch_score = total_batch_score + score

        mean_batch_score = total_batch_score / len(nets)

        if blend == "logit":
            batch_probs = torch.sigmoid(mean_batch_score)
        else:
            batch_probs = mean_batch_score

        probs_list.append(batch_probs.cpu().numpy())
        image_ids.extend(list(batch_ids))

    all_probs = np.concatenate(probs_list)
    return image_ids, all_probs


def main():
    parser = argparse.ArgumentParser(description="ForensiX-DeepFake - Model Inference Pipeline")
    parser.add_argument("--test_dir", required=True, help="Path to directory containing test images")
    parser.add_argument("--weights", required=True, help="Path to packaged model_weights.pt")
    parser.add_argument("--output", default="predictions.csv", help="Path to output predictions CSV")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers")
    parser.add_argument("--threshold", type=float, default=0.50, help="Decision threshold for label 1 (default: 0.50)")
    parser.add_argument("--model", default=None, help="Architecture name if weights is a single raw state_dict")
    parser.add_argument("--prune_vit", action="store_true", help="Prune ViT-B/16 from ensemble (reduces 58M params & VRAM < 1.5 GB)")
    parser.add_argument("--exclude", nargs="+", default=[], help="Model names to exclude from ensemble")
    parser.add_argument("--no_fp16", action="store_true", help="Disable automatic FP16 mixed precision on GPU")
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation (flip)")
    parser.add_argument("--blend", choices=["logit", "prob"], default="logit",
                        help="Ensemble blending mode: 'logit' (default) or 'prob'")
    parser.add_argument("--labels_csv", default=None, help="Optional validation labels CSV with image_id,label")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    exclude_list = list(args.exclude)
    if args.prune_vit:
        exclude_list.extend(["vit", "vit_b_16"])

    nets = load_models(args.weights, args.model, device, exclude=exclude_list)
    dataset = TestDataset(args.test_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    use_tta = not args.no_tta
    use_fp16 = (not args.no_fp16) and (device.type == "cuda")

    print(f"[INFO] Evaluating {len(dataset)} images | batch size: {args.batch_size} | "
          f"precision: {'fp16 (automatic)' if use_fp16 else 'fp32'} | "
          f"TTA: {'enabled' if use_tta else 'disabled'} | blend: {args.blend}")

    t1 = time.time()
    ids, confidences = run_inference(nets, loader, device, use_fp16, use_tta, args.blend)
    confidences = np.clip(confidences, 0.0, 1.0)
    t2 = time.time()

    labels = (confidences >= args.threshold).astype(int)

    out_df = pd.DataFrame({
        "image_id": ids,
        "label": labels,
        "confidence": confidences
    })

    # Strict submission integrity verification
    assert len(out_df) == len(dataset), f"Row count mismatch: expected {len(dataset)}, got {len(out_df)}"
    assert out_df["image_id"].is_unique, "Duplicate image IDs found in predictions"
    assert np.isfinite(out_df["confidence"]).all(), "NaN or infinite values in confidence column"
    assert out_df["confidence"].between(0.0, 1.0).all(), "Confidence values outside [0.0, 1.0]"
    assert out_df["label"].isin([0, 1]).all(), "Labels contain non-binary values"

    out_df.to_csv(args.output, index=False)

    print(f"[SUCCESS] Exported predictions to: '{args.output}'")
    print(f"  Total Images Processed: {len(out_df)}")
    print(f"  Predicted FAKE (1):     {out_df['label'].sum()} ({out_df['label'].mean() * 100:.2f}%)")
    print(f"  Predicted REAL (0):     {(out_df['label'] == 0).sum()} ({(1 - out_df['label'].mean()) * 100:.2f}%)")
    print(f"  Mean Confidence:        {out_df['confidence'].mean():.4f}")
    print(f"  Confidence Range:       [{out_df['confidence'].min():.4f}, {out_df['confidence'].max():.4f}]")
    print(f"  Timing: Model Load {t1 - t0:.2f}s | Inference {t2 - t1:.2f}s ({(t2 - t1) / len(out_df) * 1000:.2f} ms/image)")

    if args.labels_csv:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        labels_df = pd.read_csv(args.labels_csv)[["image_id", "label"]].rename(columns={"label": "true_label"})
        labels_df["image_id"] = labels_df["image_id"].astype(str).str.strip()
        merged = out_df.merge(labels_df, on="image_id")
        print(f"[EVALUATION] Evaluated against {len(merged)} labelled images:")
        print(f"  Validation AUC-ROC:  {roc_auc_score(merged['true_label'], merged['confidence']):.4f}")
        print(f"  Validation Accuracy: {accuracy_score(merged['true_label'], merged['label']) * 100:.2f}%")
        print(f"  Validation F1-Score: {f1_score(merged['true_label'], merged['label']):.4f}")


if __name__ == "__main__":
    main()
