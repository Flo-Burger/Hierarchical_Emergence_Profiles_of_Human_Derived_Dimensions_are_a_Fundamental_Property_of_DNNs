import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# Needs some extra work from HuggingFace to run this one, 
# if you want to use dinov2 you can just call the function with "facebook/dinov2-base", 
# see https://huggingface.co/facebook/dinov2-base

def extract_dinov3(images, device, model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"):
    """
    Extract [CLS] token activations from each ViT encoder layer of a DINOv3 model.

    Args:
        images: array-like of length N; each entry is an (H, W, 3) uint8 image
        device: torch.device
        model_id: HF repo id (default: DINOv3 ViT-Base patch-16)

    Returns:
        dict mapping layer names ("vit_layer_0",…,"vit_layer_{L-1}")
        → lists of numpy arrays with shape (hidden_dim,)
    """
    
    # 1) Load processor & model
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    # Ensure per-layer outputs are returned
    model.config.output_hidden_states = True

    # 2) Prepare containers
    num_layers  = int(getattr(model.config, "num_hidden_layers", 12))
    layer_names = [f"vit_layer_{i}" for i in range(num_layers)]
    activations = {name: [] for name in layer_names}

    print(f"Extracting {model_id} activations…")
    for i in tqdm(range(len(images)), desc="Images"):
        arr = images[i]

        # Be robust to dtype/layout; ensure RGB uint8 for PIL
        if isinstance(arr, np.ndarray) and arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")

        # Use HF processor for exact resizing/normalization
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, output_hidden_states=True)

        # hidden_states = (embeddings_out, layer_0, ..., layer_{L-1})
        hidden_states = outputs.hidden_states
        for layer_idx in range(1, len(hidden_states)):  # skip embeddings at idx 0
            name    = f"vit_layer_{layer_idx-1}"
            cls_tok = hidden_states[layer_idx][:, 0, :]   # CLS token
            activations[name].append(cls_tok.squeeze(0).detach().cpu().numpy())

    return activations
