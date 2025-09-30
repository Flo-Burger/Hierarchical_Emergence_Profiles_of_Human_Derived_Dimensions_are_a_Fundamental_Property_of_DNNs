import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTModel

def extract_ViT(images, device, model_id: str = "google/vit-base-patch32-224-in21k"):
    """
    Extract [CLS] token activations from each ViT encoder layer (no hooks, no batching).

    Args:
        images: array-like of length N; each entry is H×W×3 uint8
        device: torch.device
        model_id: HF model name

    Returns:
        dict mapping "vit_layer_0"… "vit_layer_11" -> list of numpy arrays (hidden_dim,)
        The i-th list element corresponds to image i.
    """
    # 1) Load processor & model
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = ViTModel.from_pretrained(model_id, output_hidden_states=True).to(device)
    model.eval()

    # 2) Prepare containers
    num_layers = model.config.num_hidden_layers
    layer_names = [f"vit_layer_{i}" for i in range(num_layers)]
    activations = {name: [] for name in layer_names}

    # 3) Process each image individually
    print("Extracting ViT-Base activations…")
    for i in tqdm(range(len(images)), desc="Images"):
        arr = np.asarray(images[i]).astype("uint8")
        img = Image.fromarray(arr)

        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)  # [1, 3, 224, 224]
            outputs = model(pixel_values=pixel_values)

            # hidden_states: tuple of (embeddings, layer0, layer1, ..., layerN)
            hs = outputs.hidden_states[1:]  # skip embeddings

            for li, h in enumerate(hs):
                cls_tok = h[:, 0, :]  # shape [1, hidden_dim]
                activations[layer_names[li]].append(
                    cls_tok.squeeze(0).cpu().numpy()
                )

    return activations