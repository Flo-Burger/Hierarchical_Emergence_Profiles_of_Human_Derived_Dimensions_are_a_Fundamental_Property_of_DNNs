#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# /…/Final_Code/layer_exctractions/feedforward_cornet.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import torch
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

# 1. point at your local CORnet checkout
sys.path.append("/Users/22119216/Desktop/PhD_First_Year/Projects/Old_or_Random/Generative_EEG/CORnet")
from cornet import cornet_rt

def extract_recurrent_cornet_activations(images, device):
    """
    images: list or array of N uint8 H×W×3 images
    device: torch.device

    returns: dict { "V1":[…], "V2":[…], "V4":[…], "IT":[…] }
    """

    # load & prep
    model = cornet_rt(pretrained=True, map_location=device).to(device)
    model.eval()

    layers = ["V1", "V2", "V4", "IT"]
    cornet_temp = {}
    activations = {L: [] for L in layers}

    # hook factory (uses .output just like your RT code)
    def create_final_hook(name):
        def hook(module, inp, outp):
            tensor = outp[0] if isinstance(outp, tuple) else outp
            flat   = tensor.view(tensor.size(0), -1)
            cornet_temp[name] = flat.detach().cpu().numpy().squeeze()
        return hook

    core = model.module if hasattr(model, "module") else model
    for L in layers:
        getattr(core, L).output.register_forward_hook(create_final_hook(L))

    # exactly your preprocessing
    preprocess = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])

    # extraction loop
    print("Extracting CORnet-RT feedforward activations…")
    for i in tqdm(range(len(images)), desc="Images"):
        img = Image.fromarray(images[i].astype("uint8"))
        inp = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(inp)
        for L in layers:
            activations[L].append(cornet_temp[L])

    return activations

