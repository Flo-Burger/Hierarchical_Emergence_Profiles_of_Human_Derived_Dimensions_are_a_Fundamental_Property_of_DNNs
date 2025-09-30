# feedforward_cornet_s.py

import sys
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Make sure Python can find the CORnet code
# ─────────────────────────────────────────────────────────────────────────────
sys.path.append("/Users/22119216/Desktop/PhD_First_Year/Projects/Old_or_Random/Generative_EEG/CORnet")
from cornet import cornet_s

def extract_skip_cornet_activations(images, device):
    """
    Extracts CORnet-S activations for each image.
    
    Parameters
    ----------
    images : array_like, shape (n_items, H, W, 3)
        Numpy array of uint8 images.
    device : torch.device
        The torch device to run the model on.
    
    Returns
    -------
    activations : dict
        Dictionary mapping layer names ("V1","V2","V4","IT") to lists of
        1D numpy arrays, one per image.
    """
    # 1) Load the pretrained model
    model = cornet_s(pretrained=True, map_location=device).to(device)
    model.eval()

    # 2) Layers we want
    layers = ["V1", "V2", "V4", "IT"]

    # 3) Hook container
    cornet_temp = {}
    def make_hook(name):
        def hook(module, inp, outp):
            tensor = outp[0] if isinstance(outp, tuple) else outp
            flat   = tensor.view(tensor.size(0), -1)
            cornet_temp[name] = flat.detach().cpu().numpy().squeeze()
        return hook

    # 4) Register hooks on each layer
    base = model.module if hasattr(model, "module") else model
    for L in layers:
        layer_module = getattr(base, L)
        layer_module.register_forward_hook(make_hook(L))

    # 5) Prepare storage & preprocessing
    activations = {L: [] for L in layers}
    preprocess = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    # 6) Loop over images
    for img_arr in tqdm(images, desc="Images"):
        img = Image.fromarray(img_arr.astype(np.uint8))
        inp = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(inp)
        for L in layers:
            activations[L].append(cornet_temp[L])

    return activations
