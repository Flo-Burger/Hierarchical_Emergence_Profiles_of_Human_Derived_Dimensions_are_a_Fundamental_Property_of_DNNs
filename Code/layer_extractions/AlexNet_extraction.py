import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T
from torchvision.models import alexnet, AlexNet_Weights

def extract_alexnet(images, device):
    """
    Extract per-layer activations from AlexNet (pretrained on ImageNet).

    Args:
        images: array-like of length N; each entry is H×W×3 uint8
        device: torch.device

    Returns:
        dict mapping:
          "alex_conv_0" … "alex_conv_4",
          "alex_fc_0", "alex_fc_1", "alex_fc_2"
        → lists of numpy arrays (layer-specific dims)
    """
    # 1) Model + preprocessing (pretrained ImageNet weights)
    weights = AlexNet_Weights.IMAGENET1K_V1
    model = alexnet(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()  # Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize

    # 2) Prepare containers
    conv_names = [f"alex_conv_{i}" for i in range(5)]
    fc_names   = [f"alex_fc_{i}"   for i in range(3)]
    layer_names = conv_names + fc_names
    activations = {name: [] for name in layer_names}

    # 3) Forward per image, capturing outputs layer-by-layer (no hooks)
    print("Extracting AlexNet (pretrained) activations…")
    for i in tqdm(range(len(images)), desc="Images"):
        arr = np.asarray(images[i])
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")

        with torch.no_grad():
            x = preprocess(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

            # AlexNet.features (conv stack)
            conv_idx = 0
            for layer in model.features:
                x = layer(x)
                # Capture after each Conv2d (before/after ReLU doesn't hugely matter; we take post-layer output)
                if isinstance(layer, torch.nn.Conv2d):
                    pooled = F.adaptive_avg_pool2d(x, output_size=1).squeeze(0).squeeze(-1).squeeze(-1)  # [C]
                    activations[f"alex_conv_{conv_idx}"].append(pooled.cpu().numpy())
                    conv_idx += 1

            # Flatten before classifier
            x = torch.flatten(x, 1)

            # AlexNet.classifier (Linear/Dropout/ReLU stack)
            fc_idx = 0
            for layer in model.classifier:
                x = layer(x)
                if isinstance(layer, torch.nn.Linear):
                    activations[f"alex_fc_{fc_idx}"].append(x.squeeze(0).cpu().numpy())
                    fc_idx += 1

    return activations