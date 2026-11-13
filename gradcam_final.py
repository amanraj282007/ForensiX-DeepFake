"""
gradcam_final.py - Standalone Multi-Architecture Grad-CAM Interpretability Engine
ForensiX-DeepFake System

Generates multi-architecture forensic attention heatmaps across all ensemble backbones:
- EffNet-B0 (light aug)
- EffNet-B0 (damage aug)
- ConvNeXt-Tiny
- ViT-B/16 (if active)
- Swin-T
- Consensus Average

Features:
- Standalone execution: Works on ANY test or evaluation directory (--images_dir / --test_dir).
- No hardcoded dependency on train_metadata.csv:
    * If --data_dir with metadata is provided, picks 3 true fakes, 3 true reals, and 3 boundary mistakes.
    * If --images_dir / --test_dir is provided (default), dynamically scores images and selects:
        1. Confident FAKE (top P >= 0.50)
        2. Confident REAL (bottom P < 0.50)
        3. Ambiguous / Boundary Uncertainty (closest to 0.50)
- Fully self-contained, robust error handling, Headless Matplotlib (Agg).

Usage:
    python gradcam_final.py --weights ./model_weights.pt --output gradcam_visualizations.png
    python gradcam_final.py --images_dir ./test/images/ --weights ./model_weights.pt
    python gradcam_final.py --data_dir /path/to/dataset --weights ./model_weights.pt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Ensure local directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gradcam import (
    DEVICE, MEAN, STD, ALL_MEMBERS,
    grad_cam, resolve_active_members
)


def load_nets(weights_path: str):
    """Loads active ensemble members into evaluation mode."""
    if not os.path.exists(weights_path):
        raise SystemExit(f"[ERROR] Weights file not found: {weights_path}")
    packed = torch.load(weights_path, map_location=DEVICE)
    active = resolve_active_members(packed)
    if not active:
        raise SystemExit(f"[ERROR] No recognized ensemble members found in {weights_path}")
    
    nets = []
    for name, (builder, get_layer, kind, title) in active.items():
        m = builder()
        m.load_state_dict(packed[name], strict=True)
        m.to(DEVICE).eval()
        nets.append((m, get_layer(m), kind, title))
    print(f"[INFO] Loaded {len(nets)} model(s) for Grad-CAM explainability: {[t for _, _, _, t in nets]}")
    return nets


def pick_images_from_metadata(data_dir: Path, nets, tf, threshold: float, seed: int, first_n: int, per_group: int):
    """Selects categorized images using training metadata if available."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    meta = pd.read_csv(data_dir / "train_metadata.csv")
    paths = {}
    for folder in ["real", "fake"]:
        for p in (data_dir / "train" / folder).glob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths[p.stem] = p
    meta["path"] = meta["image_id"].map(paths)
    meta = meta.dropna(subset=["path"])
    
    _, val_df = train_test_split(meta, test_size=0.2, stratify=meta["label"], random_state=seed)
    candidates = val_df.head(first_n)

    groups = {"fake_ok": [], "real_ok": [], "mistake": []}
    with torch.no_grad():
        for _, row in candidates.iterrows():
            x = tf(Image.open(row["path"]).convert("RGB")).unsqueeze(0).to(DEVICE)
            prob = float(np.mean([torch.sigmoid(m(x).squeeze()).item() for m, _, _, _ in nets]))
            pred = int(prob >= threshold)
            label = int(row["label"])
            if pred != label:
                key = "mistake"
            else:
                key = "fake_ok" if label == 1 else "real_ok"
            if len(groups[key]) < per_group:
                groups[key].append((row["path"], f"true={'fake' if label==1 else 'real'}"))
    return groups["fake_ok"] + groups["real_ok"] + groups["mistake"]


def pick_images_from_folder(image_dir: Path, nets, tf, threshold: float, per_group: int = 3):
    """Standalone selection from unlabelled test images based on predicted confidence."""
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])
    if not paths:
        return []

    scored = []
    with torch.no_grad():
        for p in paths[:200]:  # Evaluate up to first 200 images for selection
            try:
                x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
                prob = float(np.mean([torch.sigmoid(m(x).squeeze()).item() for m, _, _, _ in nets]))
                scored.append((p, prob))
            except Exception:
                continue

    if not scored:
        return []

    # Sort by probability
    scored.sort(key=lambda item: item[1])

    # 1. Confident Real (lowest probabilities)
    confident_real = [(p, f"P(fake)={prob:.2f} (Real)") for p, prob in scored[:per_group]]

    # 2. Confident Fake (highest probabilities)
    confident_fake = [(p, f"P(fake)={prob:.2f} (Fake)") for p, prob in scored[-per_group:]]

    # 3. Boundary / Ambiguous (closest to threshold)
    boundary_candidates = sorted(scored, key=lambda item: abs(item[1] - threshold))
    # Exclude those already chosen in real or fake
    chosen_paths = {p for p, _ in confident_real} | {p for p, _ in confident_fake}
    ambiguous = []
    for p, prob in boundary_candidates:
        if p not in chosen_paths:
            ambiguous.append((p, f"P(fake)={prob:.2f} (Boundary)"))
            if len(ambiguous) >= per_group:
                break

    return confident_fake + confident_real + ambiguous


def main():
    ap = argparse.ArgumentParser(description="Standalone Grad-CAM figure for ensemble explainability")
    ap.add_argument("--weights", default="model_weights.pt", help="Path to packed model_weights.pt")
    ap.add_argument("--output", default="gradcam_visualizations.png", help="Output PNG file path")
    ap.add_argument("--data_dir", default=None, help="Optional folder with train_metadata.csv and train/ folders")
    ap.add_argument("--images_dir", default=None, help="Folder of unlabelled test face images")
    ap.add_argument("--test_dir", default=None, help="Alias for --images_dir")
    ap.add_argument("--threshold", type=float, default=0.50, help="Decision threshold (default: 0.50)")
    ap.add_argument("--per_group", type=int, default=3, help="Images per evaluation tier (default: 3)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Resolve target image directory
    img_dir_arg = args.images_dir or args.test_dir
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    nets = load_nets(args.weights)
    chosen = []

    # Strategy 1: If data_dir has train_metadata.csv
    if args.data_dir and (Path(args.data_dir) / "train_metadata.csv").is_file():
        print(f"[INFO] Using metadata ground truth from: {args.data_dir}")
        chosen = pick_images_from_metadata(
            Path(args.data_dir), nets, tf, args.threshold, args.seed, first_n=64, per_group=args.per_group
        )

    # Strategy 2: If image_dir or test_dir is provided or exists
    if not chosen:
        candidate_dirs = [img_dir_arg, "./test/images", "./test", "./dataset/test/images", "."]
        valid_dir = None
        for d in candidate_dirs:
            if d and Path(d).is_dir():
                found_imgs = [p for p in Path(d).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
                if found_imgs:
                    valid_dir = Path(d)
                    break
        if valid_dir:
            print(f"[INFO] Standalone evaluation: selecting diverse images from '{valid_dir}'")
            chosen = pick_images_from_folder(valid_dir, nets, tf, args.threshold, per_group=args.per_group)

    # Strategy 3: Synthetic fallback if no image files exist on disk
    if not chosen:
        print("[WARN] No external images found; generating synthetic probe samples for standalone visualization.")
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp())
        for i in range(9):
            img_arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
            p = tmp_dir / f"probe_sample_{i:02d}.jpg"
            Image.fromarray(img_arr).save(p)
        chosen = pick_images_from_folder(tmp_dir, nets, tf, args.threshold, per_group=args.per_group)

    if not chosen:
        raise SystemExit("[ERROR] Failed to select images for Grad-CAM generation.")

    print(f"[INFO] Rendering Grad-CAM heatmaps for {len(chosen)} images across {len(nets)} backbones...")
    mean_t = torch.tensor(MEAN).view(3, 1, 1)
    std_t = torch.tensor(STD).view(3, 1, 1)
    cols = len(nets) + 2

    fig, axes = plt.subplots(len(chosen), cols, figsize=(2.1 * cols, 2.2 * len(chosen)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for r, (path, tag) in enumerate(chosen):
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
        results = [grad_cam(m, layer, kind, x) for m, layer, kind, _ in nets]
        img = (x[0].cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()

        # Column 0: Original input
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"Input\n{tag}", fontsize=7)

        # Columns 1..N: Individual backbones
        for j, ((cam, prob), (_, _, _, title)) in enumerate(zip(results, nets)):
            axes[r, j + 1].imshow(img)
            axes[r, j + 1].imshow(cam, cmap="jet", alpha=0.45)
            axes[r, j + 1].set_title(f"{title}\nP(fake)={prob:.2f}", fontsize=7)

        # Column N+1: Ensemble consensus average
        avg_cam = np.mean([c for c, _ in results], axis=0)
        avg_prob = float(np.mean([p for _, p in results]))
        axes[r, cols - 1].imshow(img)
        axes[r, cols - 1].imshow(avg_cam, cmap="jet", alpha=0.45)
        axes[r, cols - 1].set_title(f"Consensus Avg\nP(fake)={avg_prob:.2f}", fontsize=7)

    fig.suptitle(f"Grad-CAM: Multi-Architecture Forensic Attention Across Ensemble Members", fontsize=11, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"[SUCCESS] Certified standalone Grad-CAM visualization saved to: '{args.output}'")


if __name__ == "__main__":
    main()
