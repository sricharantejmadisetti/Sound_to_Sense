import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class BabyCryDataset(Dataset):
    """
    Custom Baby Cry Dataset for PyTorch.
    Loads raw waveforms, applies augmentations, pads/truncates, and extracts:
      1. Mel-spectrogram: (1, 128, 256)
      2. 1D Acoustic Feature Vector: (64,) [mean + std of (MFCC, Chroma, Tonnetz, Pitch)]
    """
    
    LABEL_MAP = {
        'hungry': 'hungry',
        'discomfort': 'discomfort',
        'cold_hot': 'discomfort',
        'belly_pain': 'belly_pain',
        'tired': 'tired',
        'lonely': 'tired',
        'burping': 'burping',
        'scared': 'burping'
    }
    
    CLASS_NAMES = ['hungry', 'discomfort', 'belly_pain', 'tired', 'burping']
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    
    def __init__(self, csv_path="D:/sound to sense/data/metadata.csv", split="train", sr=8000, duration=7, augment=False, extract_acoustic=False):
        self.csv_path = csv_path
        self.split = split
        self.sr = sr
        self.duration = duration
        self.target_len = sr * duration
        self.augment = augment and (split == "train")
        self.extract_acoustic = extract_acoustic
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")
            
        # Read metadata and filter by split
        df = pd.read_csv(csv_path)
        if split == "train_val":
            self.df = df[df['split'].isin(['train', 'val'])].reset_index(drop=True)
        else:
            self.df = df[df['split'] == split].reset_index(drop=True)
        
        # Verify and normalize file paths
        self.filepaths = []
        self.labels = []
        for idx, row in self.df.iterrows():
            path = row['filepath'].replace('\\', '/')
            # If path doesn't exist, check local workspace or fallback
            if not os.path.exists(path):
                # Fallback: check if we can find it in the folder D:/sound to sense/BABY CRY/
                base_name = os.path.basename(path)
                label_dir = row['label']
                fallback_path = f"D:/sound to sense/BABY CRY/{label_dir}/{base_name}"
                if os.path.exists(fallback_path):
                    path = fallback_path
            
            # Check mapped label
            raw_label = row['label']
            if raw_label in self.LABEL_MAP:
                self.filepaths.append(path)
                self.labels.append(self.LABEL_MAP[raw_label])
                
        print(f"Loaded {len(self.filepaths)} samples for split '{split}' from metadata CSV.")
        self.cache_file = f"data/cache_{split}_{self.extract_acoustic}_v3.pkl"
        if os.path.exists(self.cache_file):
            import pickle
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded {len(self.cache)} cached samples from {self.cache_file}")
            except Exception:
                self.cache = {}
        else:
            self.cache = {}
        self.wav_cache = {}

    def __len__(self):
        return len(self.filepaths)
        
    def __getitem__(self, idx):
        # 1. Check if the base unaugmented features are cached
        if idx in self.cache:
            mel_resized, acoustic_tensor, label_idx = self.cache[idx]
        else:
            path = self.filepaths[idx]
            mapped_label = self.labels[idx]
            label_idx = self.CLASS_TO_IDX[mapped_label]
            
            # Load audio (mono)
            if idx in self.wav_cache:
                y = self.wav_cache[idx]
            else:
                try:
                    import soundfile as sf
                    y, sr = sf.read(path, dtype='float32')
                    if y.ndim > 1:
                        y = np.mean(y, axis=1)
                except Exception as e:
                    y = np.zeros(self.target_len, dtype=np.float32)
                self.wav_cache[idx] = y
                
            # Pad or truncate to target duration (center crop/pad)
            if len(y) < self.target_len:
                pad_width = self.target_len - len(y)
                y = np.pad(y, (0, pad_width), 'constant')
            elif len(y) > self.target_len:
                crop_width = len(y) - self.target_len
                start = crop_width // 2
                y = y[start : start + self.target_len]
                
            # Extract Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=self.sr, n_fft=512, hop_length=160, n_mels=128
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            # Standardize to zero-mean and unit-variance for neural network stability
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-5)
            
            mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mel_resized = F.interpolate(mel_tensor, size=(128, 256), mode='bilinear', align_corners=False)
            mel_resized = mel_resized.squeeze(0) # (1, 128, 256)
            
            # Extract 1D acoustic features
            if not self.extract_acoustic:
                acoustic_tensor = torch.zeros(74, dtype=torch.float32)
            else:
                mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13, n_fft=512, hop_length=160)
                chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, n_fft=512, hop_length=160)
                tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=self.sr)
                centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr, n_fft=512, hop_length=160)
                bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr, n_fft=512, hop_length=160)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr, n_fft=512, hop_length=160)
                flatness = librosa.feature.spectral_flatness(y=y, n_fft=512, hop_length=160)
                zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=512, hop_length=160)
                
                try:
                    f0 = librosa.yin(y=y, fmin=150, fmax=600, sr=self.sr, hop_length=160)
                    f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    f0 = np.zeros(mfcc.shape[1], dtype=np.float32)
                    
                n_frames = mfcc.shape[1]
                
                def align_time(feat, target_len):
                    if feat.shape[-1] < target_len:
                        pad_w = target_len - feat.shape[-1]
                        if feat.ndim == 1:
                            return np.pad(feat, (0, pad_w), 'edge')
                        else:
                            return np.pad(feat, ((0, 0), (0, pad_w)), 'edge')
                    elif feat.shape[-1] > target_len:
                        if feat.ndim == 1:
                            return feat[:target_len]
                        else:
                            return feat[:, :target_len]
                    return feat
                    
                chroma = align_time(chroma, n_frames)
                tonnetz = align_time(tonnetz, n_frames)
                centroid = align_time(centroid, n_frames)
                bandwidth = align_time(bandwidth, n_frames)
                rolloff = align_time(rolloff, n_frames)
                flatness = align_time(flatness, n_frames)
                zcr = align_time(zcr, n_frames)
                f0 = align_time(f0, n_frames)
                
                combined_feats = np.concatenate([
                    mfcc, 
                    chroma, 
                    tonnetz, 
                    centroid, 
                    bandwidth, 
                    rolloff, 
                    flatness, 
                    zcr, 
                    f0[np.newaxis, :]
                ], axis=0)
                mean_feats = np.mean(combined_feats, axis=1)
                std_feats = np.std(combined_feats, axis=1)
                acoustic_1d = np.concatenate([mean_feats, std_feats], axis=0)
                acoustic_tensor = torch.tensor(acoustic_1d, dtype=torch.float32)
                
            # Cache the base features
            self.cache[idx] = (mel_resized, acoustic_tensor, label_idx)
            
            # Periodically write cache to disk to save warm restarts
            if len(self.cache) % 50 == 0 or len(self.cache) == len(self.filepaths):
                try:
                    import pickle
                    os.makedirs("data", exist_ok=True)
                    with open(self.cache_file, 'wb') as f:
                        pickle.dump(self.cache, f)
                except Exception:
                    pass
            
        # 2. Apply on-the-fly fast SpecAugment (only during training)
        if self.augment:
            mel_augmented = mel_resized.clone()
            
            # Frequency masking
            if np.random.rand() < 0.5:
                # Apply 1 or 2 masks
                for _ in range(np.random.randint(1, 3)):
                    mask_w = np.random.randint(6, 18)
                    mask_start = np.random.randint(0, 128 - mask_w)
                    mel_augmented[:, mask_start:mask_start+mask_w, :] = mel_augmented.min()
                    
            # Time masking
            if np.random.rand() < 0.5:
                # Apply 1 or 2 masks
                for _ in range(np.random.randint(1, 3)):
                    mask_w = np.random.randint(12, 36)
                    mask_start = np.random.randint(0, 256 - mask_w)
                    mel_augmented[:, :, mask_start:mask_start+mask_w] = mel_augmented.min()
                    
            # Gaussian Noise Injection
            if np.random.rand() < 0.5:
                noise = torch.randn_like(mel_augmented) * 0.05
                mel_augmented = mel_augmented + noise
                
            # Random Gain / Scale Contrast
            if np.random.rand() < 0.5:
                gain = np.random.uniform(0.8, 1.2)
                mel_augmented = mel_augmented * gain
                
            # Vector augmentation (add subtle Gaussian noise to acoustic vector)
            if self.extract_acoustic and np.random.rand() < 0.5:
                acoustic_augmented = acoustic_tensor + torch.randn_like(acoustic_tensor) * 0.02
            else:
                acoustic_augmented = acoustic_tensor
                
            return mel_augmented, acoustic_augmented, label_idx
            
        # For validation / test splits
        return mel_resized, acoustic_tensor, label_idx
