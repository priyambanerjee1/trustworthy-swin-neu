import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import ImageFilter
from src.config import Config

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Controlled distribution-shift transform (Gaussian blur stress-test)
    shift_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=2))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform, shift_transform

def build_dataloaders():
    train_tf, val_tf, shift_tf = get_transforms()
    
    train_set = ImageFolder(root=Config.TRAIN_DIR, transform=train_tf)
    val_set = ImageFolder(root=Config.VAL_DIR, transform=val_tf)
    shift_set = ImageFolder(root=Config.VAL_DIR, transform=shift_tf)
    
    train_loader = DataLoader(
        train_set, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=2, persistent_workers=False,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_set, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=2, persistent_workers=False,
        pin_memory=torch.cuda.is_available()
    )
    shift_loader = DataLoader(
        shift_set, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=2, persistent_workers=False,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, shift_loader, train_set.classes
