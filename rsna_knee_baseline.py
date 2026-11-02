"""Kaggle-ready 2.5D baseline for RSNA Knee Abnormality Detection.

Run with --help for options. The only expected input layout is the official
competition layout: train.csv, test.csv, train_series.csv, test_series.csv,
train_series/<study>/<series>/*.dcm and test_series/<study>/<series>/*.dcm.
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]

# These are deliberately conservative. Negation is checked first, and an
# unmentioned finding stays unknown (0.5) rather than becoming a false negative.
TERMS = {
    "ACL": r"\b(?:acl|anterior cruciate)\b",
    "MCL": r"\b(?:mcl|medial collateral)\b",
    "Medial Meniscus": r"\bmedial menisc(?:us|al)\b",
    "Lateral Meniscus": r"\blateral menisc(?:us|al)\b",
    "Medial OA": r"\bmedial (?:compartment )?(?:osteoarthritis|arthrosis|oa)\b",
    "Lateral OA": r"\blateral (?:compartment )?(?:osteoarthritis|arthrosis|oa)\b",
    "PF OA": r"\b(?:patellofemoral|pf) (?:osteoarthritis|arthrosis|oa)\b",
    "Effusion": r"\b(?:joint )?effusion\b",
    "Synovitis": r"\bsynovitis\b",
    "Baker's": r"\b(?:baker(?:'s)?|popliteal) cyst\b",
    "Contusion": r"\b(?:bone )?(?:contusion|bruise)\b",
    "Fracture": r"\bfracture\b",
}
NEG = r"(?:no|without|absent|negative for|free of|no evidence of)"


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def report_pseudolabel(report: str) -> tuple[np.ndarray, np.ndarray]:
    """Return soft targets and confidence weights from an English report."""
    text = " " + str(report).lower().replace("\n", " ") + " "
    y, w = np.full(12, .5, np.float32), np.full(12, .15, np.float32)
    for j, (_, term) in enumerate(TERMS.items()):
        hits = list(re.finditer(term, text))
        if not hits:
            continue
        # A small left context catches common report negations.
        negated = all(re.search(NEG + r".{0,45}$", text[max(0, h.start()-55):h.start()])
                      for h in hits)
        y[j] = 0.0 if negated else 1.0
        w[j] = .55
    return y, w


def read_volume(series_dir: Path, slices: int, image_size: int) -> np.ndarray:
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        return np.zeros((slices, image_size, image_size), np.float32)
    decoded = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f, force=True)
            decoded.append((float(getattr(dcm, "InstanceNumber", len(decoded))),
                            dcm.pixel_array.astype(np.float32)))
        except Exception:
            pass
    if not decoded:
        return np.zeros((slices, image_size, image_size), np.float32)
    decoded.sort(key=lambda x: x[0])
    ids = np.linspace(0, len(decoded) - 1, slices).round().astype(int)
    vol = []
    for i in ids:
        x = decoded[i][1]
        x -= np.percentile(x, 1); x /= max(np.percentile(x, 99), 1e-6)
        x = np.clip(x, 0, 1)
        x = F.interpolate(torch.from_numpy(x)[None, None], size=(image_size, image_size),
                          mode="bilinear", align_corners=False)[0, 0].numpy()
        vol.append(x)
    return np.stack(vol)


def select_series(series: pd.DataFrame, max_series: int) -> pd.DataFrame:
    """Keep complementary planes, preferring fluid-sensitive acquisitions."""
    if series.empty:
        return series
    x = series.copy()
    # `DataFrame.get(..., 0)` returns a scalar when a column is absent, which
    # has no ``fillna`` and confuses static type checkers. Keep defaults indexed
    # to the frame so both the runtime and type contract are unambiguous.
    fluid = x["Fluid_Sensitive"] if "Fluid_Sensitive" in x.columns else pd.Series(0.0, index=x.index)
    fat_suppression = x["Fat_Suppression"] if "Fat_Suppression" in x.columns else pd.Series(0.0, index=x.index)
    x["_rank"] = 2 * fluid.fillna(0) + fat_suppression.fillna(0)
    x = x.sort_values(["Anatomical_Plane", "_rank"], ascending=[True, False])
    first = x.groupby("Anatomical_Plane", dropna=False).head(1)
    rest = x.drop(first.index).sort_values("_rank", ascending=False)
    return pd.concat([first, rest]).head(max_series)


class KneeDataset(Dataset):
    def __init__(self, studies: pd.DataFrame, series: pd.DataFrame, root: Path,
                 slices: int, image_size: int, max_series: int, train: bool):
        self.studies, self.root, self.slices = studies.reset_index(drop=True), root, slices
        self.image_size, self.max_series, self.train = image_size, max_series, train
        self.by_study = {k: select_series(v, max_series) for k, v in series.groupby("StudyInstanceUID")}

    def __len__(self): return len(self.studies)

    def __getitem__(self, idx: int):
        row = self.studies.iloc[idx]; uid = row.StudyInstanceUID
        rows = self.by_study.get(uid, pd.DataFrame())
        vols, meta, mask = [], [], []
        for _, s in rows.iterrows():
            vol = read_volume(self.root / uid / str(s.SeriesInstanceUID), self.slices, self.image_size)
            if self.train and random.random() < .5: vol = vol[:, :, ::-1].copy()
            # [S,H,W] -> [S,3,H,W]: neighbouring slices make a 2.5D image.
            previous = np.concatenate([vol[:1], vol[:-1]])
            following = np.concatenate([vol[1:], vol[-1:]])
            vols.append(np.stack([previous, vol, following], 1))
            plane = str(s.get("Anatomical_Plane", ""))
            meta.append([plane == "Sagittal", plane == "Coronal", plane == "Axial",
                         float(s.get("Fluid_Sensitive", 0)), float(s.get("Fat_Suppression", 0))])
            mask.append(1.)
        while len(vols) < self.max_series:
            vols.append(np.zeros((self.slices, 3, self.image_size, self.image_size), np.float32))
            meta.append([0.] * 5); mask.append(0.)
        out = {"images": torch.tensor(np.stack(vols)), "meta": torch.tensor(meta),
               "mask": torch.tensor(mask)}
        if "target" in row:
            out["target"] = torch.tensor(row.target, dtype=torch.float32)
            out["weight"] = torch.tensor(row.weight, dtype=torch.float32)
        return out


class KneeNet(nn.Module):
    def __init__(self, encoder_weights: str | None = None):
        super().__init__()
        # Kaggle notebooks have Internet disabled.  A local ImageNet checkpoint
        # is therefore supplied explicitly instead of triggering a download.
        backbone = resnet18(weights=None)
        if encoder_weights:
            backbone.load_state_dict(torch.load(encoder_weights, map_location="cpu"), strict=False)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.meta = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 32))
        self.attn = nn.Sequential(nn.Linear(512 + 32, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Linear(512 + 32, 12)

    def forward(self, images: Tensor, meta: Tensor, mask: Tensor) -> Tensor:
        b, n, s, c, h, w = images.shape
        z = self.encoder(images.reshape(b*n*s, c, h, w)).flatten(1).reshape(b, n, s, -1).mean(2)
        z = torch.cat([z, self.meta(meta)], -1)
        a = self.attn(z).squeeze(-1).masked_fill(mask == 0, -1e4).softmax(-1)
        return self.head((z * a.unsqueeze(-1)).sum(1))


def weighted_bce(logits, targets, weights):
    # A fixed denominator preserves confidence weights even for one study.
    return (F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    ) * weights).mean()


def save_checkpoint(path, model, optimizer, epoch, step):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "step": step}, temporary)
    temporary.replace(path)


def run_epoch(model, loader, optimizer, device, checkpoint_path=None, epoch=0):
    model.train(); total = 0.
    for step, b in enumerate(loader, start=1):
        b = {k: v.to(device) for k, v in b.items()}
        logits = model(b["images"], b["meta"], b["mask"])
        loss = weighted_bce(logits, b["target"], b["weight"])
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        total += loss.item() * len(b["images"])
        if checkpoint_path and step % 200 == 0:
            save_checkpoint(checkpoint_path, model, optimizer, epoch, step)
    if checkpoint_path:
        save_checkpoint(checkpoint_path, model, optimizer, epoch, len(loader))
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); out = []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        out.append(model(b["images"], b["meta"], b["mask"]).sigmoid().cpu().numpy())
    return np.concatenate(out)


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """AUC robust to a small validation fold lacking a class."""
    scores = []
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) == 2:
            scores.append(roc_auc_score(y_true[:, j], y_pred[:, j]))
    return float(np.mean(scores)) if scores else float("nan")


def main(args):
    seed_everything(args.seed); root = Path(args.data_dir); device = "cuda" if torch.cuda.is_available() else "cpu"
    train, test = pd.read_csv(root / "train.csv"), pd.read_csv(root / "test.csv")
    tr_series, te_series = pd.read_csv(root / "train_series.csv"), pd.read_csv(root / "test_series.csv")
    gold = train[TARGETS].notna().all(axis=1)
    pseudo = np.stack([report_pseudolabel(x) for x in train.Report.fillna("")])
    train["target"] = list(np.where(gold.to_numpy()[:, None], train[TARGETS].fillna(.5).to_numpy(np.float32), pseudo[:, 0]))
    train["weight"] = list(np.where(gold.to_numpy()[:, None], 1., pseudo[:, 1]))
    # Hold out gold studies only; pseudo-labelled cases remain training-only.
    if args.epochs < 1 or not 0 <= args.fold < args.folds:
        raise ValueError("epochs must be positive and fold must be in [0, folds).")
    gold_idx = np.flatnonzero(gold)
    split = list(KFold(args.folds, shuffle=True, random_state=args.seed).split(gold_idx))[args.fold]
    valid_ids = set(train.iloc[gold_idx[split[1]]].StudyInstanceUID)
    valid = train[train.StudyInstanceUID.isin(valid_ids)].copy()
    fit = train[~train.StudyInstanceUID.isin(valid_ids)].copy()
    train_ds = KneeDataset(fit, tr_series, root / "train_series", args.slices, args.image_size, args.max_series, True)
    valid_ds = KneeDataset(valid, tr_series, root / "train_series", args.slices, args.image_size, args.max_series, False)
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_ld = DataLoader(valid_ds, args.batch_size, num_workers=args.workers, pin_memory=True)
    model = KneeNet(args.encoder_weights).to(device); optim = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    best = float("-inf")
    best_state: dict[str, Tensor] | None = None
    for epoch in range(args.epochs):
        loss = run_epoch(model, train_ld, optim, device,
                         Path(args.artifacts) / f"fold{args.fold}_latest.pt", epoch)
        p = predict(model, valid_ld, device)
        score = macro_auc(valid[TARGETS].to_numpy(np.float32), p)
        print(f"epoch={epoch + 1} loss={loss:.4f} gold_macro_auc={score:.4f}")
        # Preserve the first epoch even if a tiny gold fold cannot compute AUC
        # for any target, then retain only finite metric improvements.
        if best_state is None or (np.isfinite(score) and score > best):
            best = score if np.isfinite(score) else float("-inf")
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state.")
    model.load_state_dict(best_state)
    artifact_dir = Path(args.artifacts)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, artifact_dir / f"fold{args.fold}_best.pt")
    oof = valid[["StudyInstanceUID"]].copy()
    oof[TARGETS] = predict(model, valid_ld, device)
    oof.to_csv(artifact_dir / f"fold{args.fold}_oof.csv", index=False)
    test_ds = KneeDataset(test, te_series, root / "test_series", args.slices, args.image_size, args.max_series, False)
    test_ld = DataLoader(test_ds, args.batch_size, num_workers=args.workers, pin_memory=True)
    sub = test[["StudyInstanceUID"]].copy(); sub[TARGETS] = predict(model, test_ld, device)
    sub.to_csv(args.output, index=False); print(f"wrote {args.output}; best gold macro AUC={best:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data-dir", required=True); p.add_argument("--output", default="submission.csv")
    p.add_argument("--epochs", type=int, default=4); p.add_argument("--batch-size", type=int, default=6); p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--slices", type=int, default=24); p.add_argument("--image-size", type=int, default=224); p.add_argument("--max-series", type=int, default=5)
    p.add_argument("--folds", type=int, default=5); p.add_argument("--workers", type=int, default=2); p.add_argument("--seed", type=int, default=42); p.add_argument("--encoder-weights", help="Path to an offline ResNet-18 state dict.")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--artifacts", default="artifacts")
    main(p.parse_args())
