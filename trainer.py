import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def mixup_data(x, y, alpha=0.2, device='cpu'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    epochs=50,
    patience=10,
    device="cpu",
    checkpoint_path="checkpoints/best_model.pth",
    mixup_alpha=0.2
):
    """
    Common training loop with validation, cosine annealing scheduler,
    early stopping based on Validation Macro F1, and Mixup augmentation.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model = model.to(device)
    
    best_val_f1 = 0.0
    epochs_no_improve = 0
    
    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_trues = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            mel_spec, acoustic_1d, labels = batch
            mel_spec = mel_spec.to(device)
            acoustic_1d = acoustic_1d.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            model_name = model.__class__.__name__
            
            # Apply Mixup during training
            if mixup_alpha > 0 and np.random.rand() < 0.5:
                # Reuse the permutation index for paired input alignment in Fusion network
                batch_size = mel_spec.size()[0]
                index = torch.randperm(batch_size).to(device)
                lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
                
                mixed_mel = lam * mel_spec + (1.0 - lam) * mel_spec[index]
                labels_a, labels_b = labels, labels[index]
                
                if model_name == "MultiFeatureFusionNet":
                    mixed_acoustic = lam * acoustic_1d + (1.0 - lam) * acoustic_1d[index]
                    logits = model(mixed_mel, mixed_acoustic)
                else:
                    logits = model(mixed_mel)
                    
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
            else:
                if model_name == "MultiFeatureFusionNet":
                    logits = model(mel_spec, acoustic_1d)
                else:
                    logits = model(mel_spec)
                loss = criterion(logits, labels)
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_trues.extend(labels.cpu().numpy())
            
        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_trues, train_preds)
        train_f1 = f1_score(train_trues, train_preds, average='macro')
        
        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_trues = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                mel_spec, acoustic_1d, labels = batch
                mel_spec = mel_spec.to(device)
                acoustic_1d = acoustic_1d.to(device)
                labels = labels.to(device)
                
                if model_name == "MultiFeatureFusionNet":
                    logits = model(mel_spec, acoustic_1d)
                else:
                    logits = model(mel_spec)
                    
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_trues.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_trues, val_preds)
        val_f1 = f1_score(val_trues, val_preds, average='macro')
        
        # 3. Learning Rate Scheduler step
        if scheduler is not None:
            scheduler.step()
            
        # Print progress
        print(f"Epoch {epoch:02d} Summary:")
        print(f"  [Train] Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Macro F1: {train_f1:.4f}")
        print(f"  [Val]   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Macro F1: {val_f1:.4f}")
        
        # 4. Checkpoint saving & Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc
            }, checkpoint_path)
            print(f"  => Saved checkpoint! New best Val Macro F1: {val_f1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  => No improvement. Patience counter: {epochs_no_improve}/{patience}")
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered. Training stopped at epoch {epoch}.")
            break
            
    print(f"Training completed. Best Validation Macro F1: {best_val_f1:.4f}")
    return best_val_f1
