import torch
import torch.nn as nn
import torchvision.models as models

class MultiFeatureFusionNet(nn.Module):
    """
    Multi-Feature Fusion Network.
    Branch A: Mel-spectrogram processed by ResNet-18.
    Branch B: 1D acoustic vector (64 features) processed by a Dense feed-forward network.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(MultiFeatureFusionNet, self).__init__()
        
        # Branch A: 2D Mel-spectrogram deep feature extractor (ResNet-18 based)
        try:
            if pretrained:
                from torchvision.models import resnet18, ResNet18_Weights
                self.resnet_branch = resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                self.resnet_branch = resnet18(weights=None)
        except Exception:
            self.resnet_branch = models.resnet18(pretrained=pretrained)
            
        # Re-purpose ResNet pool & head to output 256-dimensional deep features
        in_features = self.resnet_branch.fc.in_features
        self.resnet_branch.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Freeze early conv layers to prevent overfitting
        for name, param in self.resnet_branch.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        
        # Branch B: 1D acoustic vector classifier (Dense Feed-Forward network)
        self.dense_branch = nn.Sequential(
            nn.Linear(74, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion classification head
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, mel_spec, acoustic_1d):
        # Branch A: Process Mel-spectrogram
        # Input shape: (batch, 1, H, W)
        if mel_spec.shape[1] == 1:
            mel_spec = mel_spec.repeat(1, 3, 1, 1)
        feat_a = self.resnet_branch(mel_spec) # Shape: (batch, 256)
        
        # Branch B: Process 1D features
        # Input shape: (batch, 64)
        feat_b = self.dense_branch(acoustic_1d) # Shape: (batch, 256)
        
        # Concat deep and physical features
        fused = torch.cat([feat_a, feat_b], dim=1) # Shape: (batch, 512)
        
        # Classification head
        logits = self.classifier(fused)
        return logits
