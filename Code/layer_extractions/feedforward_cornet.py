import torch
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append("/Users/22119216/Desktop/PhD_First_Year/Projects/Old_or_Random/Generative_EEG/CORnet")
from cornet import cornet_z

def extract_feedforward_cornet_activations(images, device):
    model = cornet_z(pretrained=True, map_location=device).to(device)
    model.eval()
    layers = ["V1", "V2", "V4", "IT"]
    activations = {L: [] for L in layers}

    def hook_fn(name):
        def hook(module, inp, outp):
            flat = outp[0] if isinstance(outp, tuple) else outp
            flat = flat.view(flat.size(0), -1)
            activations[name].append(flat.detach().cpu().numpy().squeeze())
        return hook

    core = model.module if hasattr(model, "module") else model
    for L in layers:
        getattr(core, L).register_forward_hook(hook_fn(L))

    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])

    print("Extracting CORnet-Z activations…")
    for i in tqdm(range(len(images)), desc="Images"):
        img = Image.fromarray(images[i].astype("uint8"))
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(x)

    return activations  # dict with {layer: [np.array, ...]}
