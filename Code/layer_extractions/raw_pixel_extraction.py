import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T

def extract_raw_pixels(
    images,
    device,                         # kept for API symmetry; not used here
    size=(224, 224),
    interpolation=T.InterpolationMode.BICUBIC,
    to_grayscale=False,             # if True, converts to single channel then repeats to 3
    return_channel_first=False      # if True, keep (C,H,W) flattened by channel blocks
):
    """
    Parameters
    ----------
    images : array-like
        Iterable of images where each item can be turned into a uint8 RGB array via
        `Image.fromarray(images[i].astype('uint8'))`.
    device : torch.device
        Unused (for API compatibility with other extractors).
    size : tuple(int,int)
        Resize target (H, W).
    interpolation : torchvision.transforms.InterpolationMode
        Interpolation for resizing.
    to_grayscale : bool
        If True, convert to L (grayscale). The output will still be 3 channels
        by simple channel repeat to keep compatibility with 3-channel expectations.
    return_channel_first : bool
        If True, flatten in channel-first order (C*H*W). If False (default),
        flatten as H*W*3 for intuitive "image-like" ordering before flatten.

    Returns
    -------
    activations : dict
        {"Raw": [np.ndarray(shape=(3*H*W,), dtype=float32), ...]}
    """
    # One "layer" so your outer loop still works identically.
    activations = {"Raw": []}

    # Compose transform: resize → ToTensor (0..1 floats)
    # No mean/std normalization (keep raw-ish pixel scale).
    base_tfms = [
        T.Resize(size, interpolation=interpolation),
        T.ToTensor(),  # (C,H,W) in [0,1]
    ]
    transform = T.Compose(base_tfms)

    print("Extracting RAW pixel vectors…")
    for i in tqdm(range(len(images)), desc="Images"):
        # Ensure uint8 PIL and RGB/Grayscale as requested
        pil = Image.fromarray(images[i].astype("uint8"))

        if to_grayscale:
            pil = pil.convert("L")  # single channel
        else:
            pil = pil.convert("RGB")

        x = transform(pil)  # (C,H,W), float32

        if to_grayscale:
            # Repeat to keep 3 channels for downstream compatibility
            x = x.repeat(3, 1, 1)  # (3,H,W)

        if return_channel_first:
            # Flatten as C*H*W blocks
            flat = x.contiguous().view(-1).cpu().numpy().astype(np.float32)
        else:
            # Flatten as H*W*3 (more "image-like" ordering)
            x_hw3 = x.permute(1, 2, 0).contiguous()  # (H,W,C)
            flat  = x_hw3.view(-1).cpu().numpy().astype(np.float32)

        activations["Raw"].append(flat)

    return activations
