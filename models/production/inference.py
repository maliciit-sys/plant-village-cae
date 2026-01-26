#!/usr/bin/env python3
"""
Standalone inference script for Tomato Disease Classification.

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --folder path/to/folder --output results.csv
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd


class Encoder(nn.Module):
    def __init__(self, latent_channels=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, latent_channels, 3, stride=2, padding=1), nn.BatchNorm2d(latent_channels), nn.ReLU(True),
        )
    def forward(self, x): return self.encoder(x)


class TomatoClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = Encoder()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16, 512), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    def forward(self, x): return self.classifier(self.encoder(x))


def load_model(model_dir):
    model_dir = Path(model_dir)
    with open(model_dir / "model_config.json") as f:
        config = json.load(f)
    model = TomatoClassifier(config["num_classes"])
    model.load_state_dict(torch.load(model_dir / "model_weights.pth", map_location="cpu"))
    model.eval()
    return model, config


def predict(model, config, image_path):
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(config["norm_mean"], config["norm_std"])
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0].numpy()
    idx = probs.argmax()
    return config["class_names"][idx], float(probs[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=".", help="Directory with model files")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--folder", help="Folder with images")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV for batch")
    args = parser.parse_args()
    
    model, config = load_model(args.model_dir)
    
    if args.image:
        cls, conf = predict(model, config, args.image)
        print(f"Prediction: {cls} (Confidence: {conf:.1%})")
    elif args.folder:
        results = []
        for p in Path(args.folder).glob("*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                cls, conf = predict(model, config, p)
                results.append({"file": p.name, "class": cls, "confidence": conf})
        pd.DataFrame(results).to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
