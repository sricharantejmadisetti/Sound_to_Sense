import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tabulate import tabulate

from data.dataset import BabyCryDataset
from models.crnn import BabyCryCRNN
from models.fusion import MultiFeatureFusionNet
from models.ast import BabyCryAST

def evaluate_model(model, dataloader, model_name, device="cpu"):
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch in dataloader:
            mel_spec, acoustic_1d, labels = batch
            mel_spec = mel_spec.to(device)
            acoustic_1d = acoustic_1d.to(device)
            
            if model_name == "Fusion":
                logits = model(mel_spec, acoustic_1d)
            else:
                logits = model(mel_spec)
                
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(labels.numpy())
            
    return np.array(all_trues), np.array(all_preds)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on device: {device}")
    
    # 1. Load Test Dataset (We must set extract_acoustic=True since Fusion model needs it)
    test_ds = BabyCryDataset(split="test", extract_acoustic=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    # 2. Define models and checkpoint paths
    models_config = {
        "CRNN": {
            "class": BabyCryCRNN,
            "kwargs": {"num_classes": 5, "pretrained": False},
            "checkpoint": "checkpoints/crnn_best.pth"
        },
        "Fusion": {
            "class": MultiFeatureFusionNet,
            "kwargs": {"num_classes": 5, "pretrained": False},
            "checkpoint": "checkpoints/fusion_best.pth"
        },
        "AST": {
            "class": BabyCryAST,
            "kwargs": {
                "num_classes": 5,
                "embed_dim": 128,
                "nhead": 4,
                "num_layers": 4,
                "dim_feedforward": 512
            },
            "checkpoint": "checkpoints/ast_best.pth"
        }
    }
    
    results = []
    os.makedirs("results", exist_ok=True)
    
    for name, config in models_config.items():
        checkpoint_path = config["checkpoint"]
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {name}: Checkpoint {checkpoint_path} not found.")
            continue
            
        print(f"Evaluating {name}...")
        
        # Instantiate model and load checkpoint
        model = config["class"](**config["kwargs"])
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Run evaluation
        y_true, y_pred = evaluate_model(model, test_loader, name, device)
        
        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        _, _, w_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        results.append([name, f"{acc*100:.2f}%", f"{prec*100:.2f}%", f"{rec*100:.2f}%", f"{f1*100:.2f}%", f"{w_f1*100:.2f}%"])
        
        # Plot and save confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=test_ds.CLASS_NAMES,
            yticklabels=test_ds.CLASS_NAMES
        )
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        cm_path = f"results/confusion_matrix_{name.lower()}.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"Saved confusion matrix to {cm_path}")
        
    # Print comparison table
    if results:
        headers = ["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
        print("\n--- Side-by-Side Model Performance Comparison ---")
        print(tabulate(results, headers=headers, tablefmt="grid"))
    else:
        print("No models evaluated.")

if __name__ == "__main__":
    main()
