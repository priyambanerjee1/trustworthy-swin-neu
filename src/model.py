import torch
import torch.nn as nn
from torchvision import models
from src.config import Config

def build_swin_t(num_classes=6, dropout_p=0.2, pretrained=True):
    weights = models.Swin_T_Weights.DEFAULT if pretrained else None
    model = models.swin_t(weights=weights)
    
    in_features = model.head.in_features
    # Custom head: Stochastic Dropout + Linear projection
    model.head = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes)
    )
    return model
