"""Fine-tune one classifier per task (gender, sleeve) via transfer learning.

For a given task we:
  1. load data/labeled.csv and keep rows that have that task's label,
  2. map the label strings to class indices (order fixed in config),
  3. make a stratified train/validation split,
  4. fine-tune a pretrained ResNet18 (small LR on the backbone, larger on the
     new head), tracking the best validation accuracy,
  5. save the best checkpoint and print accuracy + a confusion matrix.

Run:
    python -m src.train --task gender
    python -m src.train --task sleeve
    python -m src.train --task all        # train both, one after the other
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from . import config, model as model_lib

TASKS = {
    "gender": {"label_col": "gender_label", "classes": config.GENDER_CLASSES,
               "path": config.GENDER_MODEL_PATH},
    "sleeve": {"label_col": "sleeve_label", "classes": config.SLEEVE_CLASSES,
               "path": config.SLEEVE_MODEL_PATH},
}


def _set_seed(seed: int = config.RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FashionDataset(Dataset):
    """Loads images from local paths with an integer class label."""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["image_path"].tolist()
        self.labels = df["label_idx"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = config.BASE_DIR / self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Corrupt/missing file: substitute a blank image so a single bad
            # row doesn't crash a whole epoch.
            img = Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE))
        return self.transform(img), self.labels[idx]


def _prepare_dataframe(task: str) -> pd.DataFrame:
    cfg = TASKS[task]
    if not config.LABELED_CSV.exists():
        raise FileNotFoundError(
            f"{config.LABELED_CSV} not found. Run `python -m src.labeler` first."
        )
    df = pd.read_csv(config.LABELED_CSV)
    df = df[df[cfg["label_col"]].notna()].copy()

    # Keep only rows whose image actually exists on disk.
    df = df[df["image_path"].apply(lambda p: (config.BASE_DIR / str(p)).exists())]

    class_to_idx = {c: i for i, c in enumerate(cfg["classes"])}
    df = df[df[cfg["label_col"]].isin(class_to_idx)]
    df["label_idx"] = df[cfg["label_col"]].map(class_to_idx)

    if len(df) < 20:
        raise ValueError(
            f"Only {len(df)} usable rows for task '{task}'. Need more labeled "
            f"data — scrape more products or widen the label rules in config.py."
        )
    return df.reset_index(drop=True)


def _build_optimizer(net: nn.Module) -> torch.optim.Optimizer:
    """Smaller LR on the pretrained backbone, larger on the fresh head."""
    head_params = list(net.fc.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in net.parameters() if id(p) not in head_ids]
    return torch.optim.Adam(
        [
            {"params": backbone_params, "lr": config.LEARNING_RATE / 10},
            {"params": head_params, "lr": config.LEARNING_RATE},
        ]
    )


def _run_epoch(net, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    """One pass over loader. If optimizer is given, trains; else evaluates."""
    training = optimizer is not None
    net.train(training)
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def train_task(task: str, epochs: int = config.NUM_EPOCHS) -> dict:
    _set_seed()
    cfg = TASKS[task]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Training '{task}' classifier on {device} ===")

    df = _prepare_dataframe(task)
    print(f"Usable samples: {len(df)} "
          f"({df[cfg['label_col']].value_counts().to_dict()})")

    train_df, val_df = train_test_split(
        df, test_size=config.VAL_SPLIT, random_state=config.RANDOM_SEED,
        stratify=df["label_idx"],
    )

    train_ds = FashionDataset(train_df, model_lib.get_transforms(train=True))
    val_ds = FashionDataset(val_df, model_lib.get_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS)

    net = model_lib.build_model(num_classes=len(cfg["classes"])).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(net)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(net, train_loader, criterion, device, optimizer)
        val_loss, val_acc = _run_epoch(net, val_loader, criterion, device)
        print(f"epoch {epoch:2d}/{epochs}  "
              f"train_loss={tr_loss:.3f} acc={tr_acc:.3f}  |  "
              f"val_loss={val_loss:.3f} acc={val_acc:.3f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            model_lib.save_model(net, cfg["path"], cfg["classes"])

    # Final report using the best saved model.
    best_net, classes = model_lib.load_model(cfg["path"], device)
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            preds = best_net(images.to(device)).argmax(1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    print("Classification report (validation):")
    print(classification_report(y_true, y_pred, target_names=classes,
                                labels=list(range(len(classes))), zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))))
    print(f"Saved model -> {cfg['path']}")
    return {"task": task, "best_val_acc": best_val_acc, "n_samples": len(df)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune fashion classifiers.")
    parser.add_argument("--task", choices=["gender", "sleeve", "all"],
                        default="all", help="which classifier(s) to train")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    args = parser.parse_args()

    tasks = ["gender", "sleeve"] if args.task == "all" else [args.task]
    for task in tasks:
        train_task(task, epochs=args.epochs)


if __name__ == "__main__":
    main()
