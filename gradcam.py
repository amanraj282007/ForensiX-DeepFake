"""
Grad-CAM visualisation for the five-model deepfake ensemble.
For each chosen image it shows the original, the heat map and P(fake) of every member,
and the consensus average map across the ensemble.

Usage:
    python gradcam.py --images_dir ./test/images --weights ./model_weights.pt --output gradcam_test_examples.png
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny, efficientnet_b0, swin_t, vit_b_16

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_effnet_b0():
    m = efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(m.classifier[1].in_features, 1))
    return m


def build_convnext():
    m = convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
    return m


def build_vit():
    m = vit_b_16(weights=None)
    m.heads.head = nn.Linear(m.heads.head.in_features, 1)
    return m


def build_swin():
    m = swin_t(weights=None)
    m.head = nn.Linear(m.head.in_features, 1)
    return m


# 5 Ensemble backbones
ALL_MEMBERS = {
    "b0_base": (build_effnet_b0, lambda m: m.features[-1], "cnn", "EffNet-B0 (light aug)"),
    "b0_corrupt": (build_effnet_b0, lambda m: m.features[-1], "cnn", "EffNet-B0 (damage aug)"),
    "convnext": (build_convnext, lambda m: m.features[-1], "cnn", "ConvNeXt-Tiny"),
    "vit": (build_vit, lambda m: m.encoder.layers[-1].ln_1, "vit", "ViT-B/16"),
    "swin": (build_swin, lambda m: m.features[-1], "swin", "Swin-T"),
}

# Backwards compatible alias for gradcam_final.py
MEMBERS = ALL_MEMBERS


def resolve_active_members(packed_dict):
    active = {}
    for k, v in ALL_MEMBERS.items():
        if k in packed_dict:
            active[k] = v
    if not active:
        # Fallback if names are prefixed or formatted differently
        for k in packed_dict.keys():
            if k in ALL_MEMBERS:
                active[k] = ALL_MEMBERS[k]
    return active


def grad_cam(model, layer, kind, x):
    """Return (heat map 224x224 in [0, 1], P(fake)) for one image tensor of shape (1, 3, 224, 224)."""
    store = {}

    def hook(module, inputs, output):
        store["a"] = output
        output.retain_grad()

    handle = layer.register_forward_hook(hook)
    model.zero_grad()
    logit = model(x).squeeze()
    logit.backward()
    handle.remove()
    acts, grads = store["a"].detach(), store["a"].grad.detach()
    if kind == "swin":      # (1, H, W, C) -> (1, C, H, W)
        acts, grads = acts.permute(0, 3, 1, 2), grads.permute(0, 3, 1, 2)
    elif kind == "vit":     # drop class token, 196 patch tokens -> 14 x 14 grid
        acts = acts[:, 1:].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
        grads = grads[:, 1:].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy(), torch.sigmoid(logit).item()


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for the five-model deepfake ensemble")
    parser.add_argument("--images_dir", required=True, help="Folder of face images")
    parser.add_argument("--weights", required=True, help="Packed model_weights.pt")
    parser.add_argument("--output", default="gradcam_examples.png", help="Output image path")
    parser.add_argument("--n", type=int, default=6, help="Number of random images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    packed = torch.load(args.weights, map_location=DEVICE)
    active_members = resolve_active_members(packed)
    if not active_members:
        raise SystemExit(f"[ERROR] Could not match any members from weights keys: {list(packed.keys())}")

    nets = []
    for name, (builder, get_layer, kind, title) in active_members.items():
        m = builder()
        m.load_state_dict(packed[name])
        m.to(DEVICE).eval()
        nets.append((m, get_layer(m), kind, title))

    folder = Path(args.images_dir)
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise SystemExit(f"No images found in {folder}")
    chosen = random.Random(args.seed).sample(paths, min(args.n, len(paths)))

    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    mean_t = torch.tensor(MEAN).view(3, 1, 1)
    std_t = torch.tensor(STD).view(3, 1, 1)

    cols = len(nets) + 2
    fig, axes = plt.subplots(len(chosen), cols, figsize=(2.0 * cols, 2.2 * len(chosen)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for r, path in enumerate(chosen):
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
        results = [grad_cam(m, layer, kind, x) for m, layer, kind, _ in nets]
        img = (x[0].cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(path.name[:18], fontsize=7)
        for j, ((cam, prob), (_, _, _, title)) in enumerate(zip(results, nets)):
            axes[r, j + 1].imshow(img)
            axes[r, j + 1].imshow(cam, cmap="jet", alpha=0.4)
            axes[r, j + 1].set_title(f"{title}\nP(fake)={prob:.2f}", fontsize=7)
        avg_cam = np.mean([c for c, _ in results], axis=0)
        avg_prob = float(np.mean([p for _, p in results]))
        axes[r, cols - 1].imshow(img)
        axes[r, cols - 1].imshow(avg_cam, cmap="jet", alpha=0.4)
        axes[r, cols - 1].set_title(f"Average of {len(nets)}\nP(fake)={avg_prob:.2f}", fontsize=7)
    fig.suptitle(f"Grad-CAM across {len(nets)} ensemble members: randomly chosen images", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"[DONE] Saved {len(chosen)} Grad-CAM examples to {args.output}")


if __name__ == "__main__":
    main()
