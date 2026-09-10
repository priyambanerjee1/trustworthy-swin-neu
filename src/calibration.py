import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from src.config import Config

class TemperatureScaler(nn.Module):
    def __init__(self, init_temp=1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits):
        return logits / self.temperature

def optimize_temperature(model, val_loader):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(Config.DEVICE)
            logits = model(images)
            logits_list.append(logits)
            labels_list.append(labels.to(Config.DEVICE))

    val_logits = torch.cat(logits_list)
    val_labels = torch.cat(labels_list)

    scaler = TemperatureScaler().to(Config.DEVICE)
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)

    def eval_fn():
        optimizer.zero_grad()
        scaled = scaler(val_logits)
        loss = nll_criterion(scaled, val_labels)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    optimal_T = float(scaler.temperature.item())
    return optimal_T

def compute_ece(confs, preds, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confs > lower) & (confs <= upper)
        prop = np.mean(mask)
        if prop > 0:
            acc = np.mean(preds[mask] == labels[mask])
            conf = np.mean(confs[mask])
            ece += prop * np.abs(acc - conf)
    return float(ece)
