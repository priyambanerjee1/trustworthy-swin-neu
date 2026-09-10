import time
import torch
import torch.nn as nn
import torch.optim as optim
from src.config import Config
from src.dataset import build_dataloaders
from src.model import build_swin_t

def main():
    torch.manual_seed(Config.SEED)
    train_loader, val_loader, _, _ = build_dataloaders()
    model = build_swin_t(num_classes=Config.NUM_CLASSES, dropout_p=0.2, pretrained=True).to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    best_val_loss = float("inf")
    print(f"Training Swin-T on {Config.DEVICE} for {Config.EPOCHS} epochs...")
    
    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        ep_val_loss = val_loss / val_total
        ep_val_acc = (val_correct / val_total) * 100.0
        
        if ep_val_loss < best_val_loss:
            best_val_loss = ep_val_loss
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            print(f"Epoch [{epoch:02d}/{Config.EPOCHS:02d}] * Best Saved -> Val Loss: {ep_val_loss:.6f}, Val Acc: {ep_val_acc:.2f}%")
        else:
            print(f"Epoch [{epoch:02d}/{Config.EPOCHS:02d}]   Val Loss: {ep_val_loss:.6f}, Val Acc: {ep_val_acc:.2f}%")

    print(f"Training complete. Weights saved to: {Config.CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()
