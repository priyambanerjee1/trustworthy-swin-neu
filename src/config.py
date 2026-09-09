import os
import torch

class Config:
    SEED = 42
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 6
    MC_SAMPLES = 20
    
    # 3-Tier Policy Thresholds
    HIGH_TRUST_THRESH = 0.90
    LOW_TRUST_THRESH = 0.60
    
    # Paths (adjust according to your environment)
    ROOT_DIR = "/kaggle/input/datasets/kaustubhdikshit/neu-surface-defect-database/NEU-DET"
    TRAIN_DIR = os.path.join(ROOT_DIR, "train/images")
    VAL_DIR = os.path.join(ROOT_DIR, "validation/images")
    OUTPUT_DIR = "/kaggle/working"
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "swin_neu_best.pth")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
