#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# /…/Final_Code/layer_exctractions/feedforward_clip_vision.py
# ─────────────────────────────────────────────────────────────────────────────

import torch
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

def extract_CLIP(images, device):
    """
    Extract [CLS] token activations from each CLIP-ViT vision encoder layer.

    Args:
        images: array-like of shape (N, H, W, 3), dtype uint8
        device: torch.device

    Returns:
        vision_activations: dict {
            "vision_layer_0": [hidden_dim array, ...],
            …,
            "vision_layer_11": [...]
        }
    """
    # 1) Load CLIP-ViT
    clip_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(clip_name)
    model     = CLIPModel.from_pretrained(clip_name).to(device)
    model.eval()

    # 2) Prepare containers & hooks
    num_layers = len(model.vision_model.encoder.layers)
    layer_names = [f"vision_layer_{i}" for i in range(num_layers)]
    vision_activations = {name: [] for name in layer_names}

    def make_hook(name):
        def hook(module, inp, outp):
            # outp may be ModelOutput or tuple
            hs = outp.last_hidden_state if hasattr(outp, "last_hidden_state") else outp[0]
            cls = hs[:, 0, :].detach().cpu().numpy().squeeze()
            vision_activations[name].append(cls)
        return hook

    for i, layer in enumerate(model.vision_model.encoder.layers):
        layer.register_forward_hook(make_hook(layer_names[i]))

    # 3) Preprocessing (use CLIP's mean/std)
    # 3) Preprocessing (CLIP-ViT base static mean/std)
    preprocess = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std =[0.26862954, 0.26130258, 0.27577711]
        ),
    ])

    # 4) Run images through the vision encoder
    print("Extracting CLIP-ViT vision activations…")
    for img_array in tqdm(images, desc="Images"):
        img = Image.fromarray(img_array.astype("uint8"))
        pixel_values = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model.vision_model(pixel_values=pixel_values)

    return vision_activations


if __name__ == "__main__":
    import scipy.io as sio
    import pickle
    import numpy as np

    # Load raw images from .mat
    mat = sio.loadmat(
        "/Users/22119216/Desktop/PhD_First_Year/Projects/Hebart_Dimensions/THINGS_Dimensions_OLD/im.mat"
    )
    images = mat["im"].flatten()  # expects shape (N,) of H×W×3 arrays

    # Force CPU for compatibility
    device = torch.device("cpu")
    print(f"Using device: {device}\n")

    # Extract vision activations
    vis_acts = extract_feedforward_clip_vision_activations(images, device)

    # Save to disk
    out_file = (
        "/Users/22119216/Desktop/PhD_First_Year/"
        "Projects/Hebart_Dimensions/Results/CLIP_Vision_activations.pkl"
    )
    with open(out_file, "wb") as f:
        pickle.dump(vis_acts, f)

    print(f"Saved vision activations → {out_file}")
