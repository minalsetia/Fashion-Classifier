"""Model factory + image transforms for transfer learning.

We fine-tune a ResNet18 that was pretrained on ImageNet: the convolutional
backbone already knows generic visual features, so we only replace the final
fully-connected layer with a fresh 2-class head and fine-tune. This is the
"transfer learning" the brief asks for and works well with a few hundred images.

A checkpoint is a dict bundling the weights with the metadata needed to rebuild
and interpret the model (class order, backbone name, image size).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

from . import config


def build_model(num_classes: int = 2) -> nn.Module:
    """Return a ResNet18 with an ImageNet backbone and a fresh classifier head."""
    # `weights=...DEFAULT` downloads the pretrained ImageNet weights on first use.
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, num_classes)  # new, randomly-initialised head
    return net


def get_transforms(train: bool) -> transforms.Compose:
    """Image preprocessing. Training adds augmentation to fight overfitting."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),       # tops are left/right symmetric
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def save_model(model: nn.Module, path: Path, classes: list[str]) -> None:
    """Persist weights + the metadata needed to rebuild/interpret the model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": classes,
            "backbone": config.BACKBONE,
            "img_size": config.IMG_SIZE,
        },
        path,
    )


def load_model(path: Path, device: torch.device | str = "cpu") -> tuple[nn.Module, list[str]]:
    """Rebuild a model from a checkpoint and load its weights. Returns (model, classes)."""
    checkpoint = torch.load(path, map_location=device)
    classes = checkpoint["classes"]
    model = build_model(num_classes=len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, classes
