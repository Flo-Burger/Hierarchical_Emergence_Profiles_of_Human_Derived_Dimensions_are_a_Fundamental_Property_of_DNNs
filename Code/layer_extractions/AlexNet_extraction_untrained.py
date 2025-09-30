#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# /…/Final_Code/layer_exctractions/feedforward_alexnet_untrained.py
# ─────────────────────────────────────────────────────────────────────────────
# Same as the pretrained extractor, but with random-initialized AlexNet.
# Uses ImageNet-style normalization for comparability.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T
from torchvision.models import alexnet

torch.manual_seed(123)

def extract_alexnet_untrained(images, device):
    """
    Extract per-layer activations from AlexNet (untrained / random init).

    Args:
        images: array-like of length N; each entry is H×W×3 uint8
        device: torch.device

    Returns:
        dict mapping:
          "alex_conv_0" … "alex_conv_4",
          "alex_fc_0", "alex_fc_1", "alex_fc_2"
        → lists of numpy arrays (layer-specific dims)
    """
    # 1) Model without pretrained weights
    model = alexnet(weights=None).to(device)
    model.eval()

    # ImageNet-style preprocessing (to match the pretrained version’s input stats)
    preprocess = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # 2) Prepare containers
    conv_names = [f"alex_conv_{i}" for i in range(5)]
    fc_names   = [f"alex_fc_{i}"   for i in range(3)]
    layer_names = conv_names + fc_names
    activations = {name: [] for name in layer_names}

    # 3) Forward per image (no batching)
    print("Extracting AlexNet (untrained) activations…")
    for i in tqdm(range(len(images)), desc="Images"):
        arr = np.asarray(images[i])
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")

        with torch.no_grad():
            x = preprocess(img).unsqueeze(0).to(device)

            # Conv stack
            conv_idx = 0
            for layer in model.features:
                x = layer(x)
                if isinstance(layer, torch.nn.Conv2d):
                    pooled = F.adaptive_avg_pool2d(x, output_size=1).squeeze(0).squeeze(-1).squeeze(-1)  # [C]
                    activations[f"alex_conv_{conv_idx}"].append(pooled.cpu().numpy())
                    conv_idx += 1

            # Flatten + classifier
            x = torch.flatten(x, 1)
            fc_idx = 0
            for layer in model.classifier:
                x = layer(x)
                if isinstance(layer, torch.nn.Linear):
                    activations[f"alex_fc_{fc_idx}"].append(x.squeeze(0).cpu().numpy())
                    fc_idx += 1

    return activations



