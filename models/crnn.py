import torch
import torch.nn as nn
import torchvision.models as models

class BabyCryCRNN(nn.Module):
    """
    CRNN Model: ResNet-18 + Bidirectional LSTM + Dense classification head.
    Expects input Mel-spectrogram of shape (batch, 1, 128, 256).
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(BabyCryCRNN, self).__init__()
        
        # Load ResNet-18 feature extractor
        try:
            if pretrained:
                # Modern torchvision way
                from torchvision.models import resnet18, ResNet18_Weights
                self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                self.resnet = resnet18(weights=None)
        except Exception:
            # Fallback for older torchvision versions or offline environments
            self.resnet = models.resnet18(pretrained=pretrained)
            
        # We need the conv layers up to the output of layer4
        # We will remove the original avgpool and fc layers
        self.features = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )
        
        # Freeze early conv layers to prevent overfitting on the small dataset
        for name, param in self.features.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False
        
        # ResNet-18 layer4 outputs 512 channels.
        # Height and Width of feature map for (128, 256) input will be (4, 8) because of 32x total downsampling.
        # We collapse the height dimension (4) and feed the width dimension (8) as the sequence length.
        lstm_input_size = 512
        hidden_size = 256
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Bidirectional LSTM has output size: hidden_size * 2 = 512
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Input shape: (batch, 1, H, W)
        # ResNet expects 3 channels, so replicate the single channel
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # Extract features from ResNet conv layers
        # Output shape: (batch, 512, H_feat, W_feat)
        feats = self.features(x)
        
        # Collapse the height dimension (H_feat) via average pooling
        # Output shape: (batch, 512, W_feat)
        temporal_seq = torch.mean(feats, dim=2)
        
        # Reshape for LSTM: (batch, seq_len, input_size) -> (batch, W_feat, 512)
        temporal_seq = temporal_seq.permute(0, 2, 1)
        
        # Pass through BiLSTM
        # lstm_out shape: (batch, seq_len, hidden_size * 2) = (batch, W_feat, 512)
        lstm_out, _ = self.lstm(temporal_seq)
        
        # Aggregate temporal representations via mean pooling across the sequence length (time)
        # pooled shape: (batch, 512)
        pooled = torch.mean(lstm_out, dim=1)
        
        # Classify
        logits = self.fc(pooled)
        return logits
