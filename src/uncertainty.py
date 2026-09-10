import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from src.config import Config

def extract_mc_uncertainty(model, loader, temp=1.0, n_mc=20, desc="MC Sampling"):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            
    raw_uncertainties, mc_confidences, all_preds = [], [], []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=desc):
            images = images.to(Config.DEVICE)
            batch_samples = []
            
            for _ in range(n_mc):
                logits = model(images)
                scaled_logits = logits / temp
                probs = F.softmax(scaled_logits, dim=1)
                batch_samples.append(probs.unsqueeze(0))
                
            mc_tensor = torch.cat(batch_samples, dim=0) # [n_mc, B, K]
            mean_probs = mc_tensor.mean(dim=0)          # [B, K]
            
            conf, _ = torch.max(mean_probs, dim=1)
            entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-12), dim=1)
            
            raw_uncertainties.extend(entropy.cpu().numpy())
            mc_confidences.extend(conf.cpu().numpy())
            
    return np.array(raw_uncertainties), np.array(mc_confidences)

def normalize_with_reference(u_raw, u_min_ref, u_max_ref):
    """
    Fixed in-domain reference normalization: clamps any test entropy
    against pristine in-domain extrema.
    """
    return np.clip((u_raw - u_min_ref) / (u_max_ref - u_min_ref + 1e-12), 0.0, 1.0)
