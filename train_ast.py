import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import StratifiedKFold

from data.dataset import BabyCryDataset
from models.ast import BabyCryAST
from trainer import train_model

def main():
    # Configuration
    batch_size = 32
    epochs = 20
    lr = 2e-4
    patience = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # 1. Instantiate datasets
    full_train_ds = BabyCryDataset(split="train_val", augment=True)
    full_val_ds = BabyCryDataset(split="train_val", augment=False)
    
    # Get labels for Stratified K-Fold partitioning
    labels = [full_train_ds.CLASS_TO_IDX[lbl] for lbl in full_train_ds.labels]
    labels = np.array(labels)
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # 2. Loop over folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n==========================================")
        print(f"       TRAINING FOLD {fold + 1} / 3")
        print(f"==========================================")
        
        # Create Subsets for train and validation fold
        train_fold_ds = Subset(full_train_ds, train_idx)
        val_fold_ds = Subset(full_val_ds, val_idx)
        
        # Calculate dynamic class weights for weighted random sampling
        train_labels = labels[train_idx]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / (class_counts + 1e-5)
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(train_fold_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
        val_loader = DataLoader(val_fold_ds, batch_size=batch_size, shuffle=False)
        
        # 3. Model, Optimizer, Loss (with label smoothing)
        model = BabyCryAST(
            num_classes=len(full_train_ds.CLASS_NAMES),
            embed_dim=128,
            nhead=4,
            num_layers=4,
            dim_feedforward=512
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
        
        # Weight decay to prevent transformer overfitting
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        checkpoint_path = f"checkpoints/ast_fold_{fold}.pth"
        
        # 4. Start training this fold
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            epochs=epochs,
            patience=patience,
            device=device,
            checkpoint_path=checkpoint_path,
            mixup_alpha=0.2
        )

if __name__ == "__main__":
    main()
