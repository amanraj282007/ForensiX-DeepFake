"""
Retrain the five-model deepfake ensemble on ALL labelled images (no validation split).
Same recipe as the 80/20 runs; each model is saved at the epoch that was best there.
"""
import argparse
import io
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (convnext_tiny, efficientnet_b0, swin_t, vit_b_16)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
W = "IMAGENET1K_V1"


def build_effnet_b0():
    m = efficientnet_b0(weights=W)
    m.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(m.classifier[1].in_features, 1))
    return m


def build_convnext():
    m = convnext_tiny(weights=W)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
    return m


def build_vit():
    m = vit_b_16(weights=W)
    m.heads.head = nn.Linear(m.heads.head.in_features, 1)
    return m


def build_swin():
    m = swin_t(weights=W)
    m.head = nn.Linear(m.head.in_features, 1)
    return m


def build_effnet_b3():
    from torchvision.models import efficientnet_b3
    m = efficientnet_b3(weights=W)
    m.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(m.classifier[1].in_features, 1))
    return m


CFG = {
    "b0_corrupt": dict(b=build_effnet_b0, aug="damage", epochs=5, save=5, lr=3e-4, wd=1e-4, bs=64, ls=0.05, seed=42),
    "b0_base":    dict(b=build_effnet_b0, aug="light",  epochs=5, save=4, lr=3e-4, wd=1e-4, bs=64, ls=0.05, seed=123),
    "convnext":   dict(b=build_convnext,  aug="light",  epochs=5, save=2, lr=1e-4, wd=0.05, bs=64, ls=0.10, seed=7),
    "b3":         dict(b=build_effnet_b3, aug="damage", epochs=8, save=8, lr=3e-4, wd=1e-4, bs=24, ls=0.05, seed=42),
    "vit":        dict(b=build_vit,       aug="light",  epochs=3, save=2, lr=3e-5, wd=0.05, bs=32, ls=0.10, seed=42),
    "swin":       dict(b=build_swin,      aug="light",  epochs=5, save=3, lr=1e-4, wd=0.05, bs=32, ls=0.10, seed=42),
}


class RandomJPEG:
    def __init__(self, p, q):
        self.p, self.q = p, q

    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(*self.q))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomBlur:
    def __init__(self, p, sigma):
        self.p, self.sigma = p, sigma

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(random.uniform(*self.sigma)))


def damage(x):
    _, H, Wd = x.shape
    if random.random() < 0.15:
        for _ in range(random.randint(1, 4)):
            s = random.randint(10, 40)
            i, j = random.randint(0, H - s), random.randint(0, Wd - s)
            x[:, i:i + s, j:j + s] = torch.rand(3, 1, 1)
    if random.random() < 0.15:
        h, w = random.randint(15, 50), random.randint(15, 50)
        i, j = random.randint(0, H - h), random.randint(0, Wd - w)
        x[:, i:i + h, j:j + w] = random.choice([0.0, 0.5])
    if random.random() < 0.15:
        for _ in range(random.randint(20, 150)):
            r = random.randint(1, 3)
            i, j = random.randint(0, H - r), random.randint(0, Wd - r)
            x[:, i:i + r, j:j + r] = 0.0
    if random.random() < 0.2:
        x = (x + torch.randn_like(x) * random.uniform(0.04, 0.12)).clamp(0, 1)
    if random.random() < 0.03:
        x = 1 - x
    return x


def make_transform(kind):
    if kind == "damage":
        pil = [transforms.RandomHorizontalFlip(), RandomBlur(0.1, (0.1, 1.0)), RandomJPEG(0.3, (50, 95)),
               transforms.ColorJitter(0.15, 0.15, 0.15)]
        mid = [transforms.ToTensor(), transforms.Lambda(damage)]
    else:
        pil = [transforms.RandomHorizontalFlip(), RandomBlur(0.1, (0.1, 0.8)), RandomJPEG(0.2, (60, 95)),
               transforms.ColorJitter(0.1, 0.1, 0.1)]
        mid = [transforms.ToTensor()]
    return transforms.Compose([transforms.Resize((224, 224))] + pil + mid + [transforms.Normalize(MEAN, STD)])


def find_images(root):
    items = []
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            parent = p.parent.name.lower()
            if parent in {"real", "fake"}:
                items.append((str(p), 1 if parent == "fake" else 0))
    if not items:
        raise SystemExit("No images found in folders named 'real'/'fake'. Edit find_images().")
    return items


class DS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), torch.tensor(float(y))


def train_one(name, cfg, items, out_dir, workers):
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    model = cfg["b"]().to(DEVICE)
    loader = DataLoader(DS(items, make_transform(cfg["aug"])), batch_size=cfg["bs"], shuffle=True,
                        num_workers=workers, drop_last=False, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_s = y * (1 - cfg["ls"]) + 0.5 * cfg["ls"]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                loss = lossf(model(x).squeeze(1), y_s)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        print(f"[{name}] epoch {ep}/{cfg['epochs']} loss {tot / n:.4f}", flush=True)
        if ep == cfg["save"]:
            sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(sd, Path(out_dir) / f"{name}.pt")
            print(f"[{name}] saved epoch {ep}", flush=True)
            break
    del model, opt
    torch.cuda.empty_cache()


def pack(out_dir):
    packed = {n: torch.load(Path(out_dir) / f"{n}.pt", map_location="cpu") for n in CFG}
    torch.save(packed, Path(out_dir) / "model_weights.pt")
    print("[DONE] wrote", Path(out_dir) / "model_weights.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models", default=",".join(CFG))
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pack", action="store_true")
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    if a.pack:
        return pack(a.out_dir)
    items = find_images(a.data_dir)
    print(f"[INFO] {len(items)} images, {sum(y for _, y in items)} fake")
    for name in a.models.split(","):
        if (Path(a.out_dir) / f"{name}.pt").exists():
            print(f"[SKIP] {name} already trained"); continue
        train_one(name, CFG[name], items, a.out_dir, a.num_workers)
    if all((Path(a.out_dir) / f"{n}.pt").exists() for n in CFG):
        pack(a.out_dir)


if __name__ == "__main__":
    main()
