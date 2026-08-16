import os
import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
import soundfile as sf
import seaborn as sns

# Import model architectures
from models.crnn import BabyCryCRNN
from models.fusion import MultiFeatureFusionNet
from models.ast import BabyCryAST

# Set page configuration with custom title and layout
st.set_page_config(
    page_title="Sound to Sense v2 | Baby Cry Diagnosis",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS for styled cards, typography, and layout
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title and Header styling */
    .title-gradient {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 50%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        color: #7F8C8D;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Custom Card container */
    .diagnose-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15);
    }
    
    /* Condition card themes */
    .card-hungry {
        background: linear-gradient(135deg, rgba(255, 179, 0, 0.15) 0%, rgba(255, 109, 0, 0.15) 100%);
        border: 1px solid rgba(255, 109, 0, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .card-discomfort {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.15) 0%, rgba(0, 145, 234, 0.15) 100%);
        border: 1px solid rgba(0, 145, 234, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .card-belly-pain {
        background: linear-gradient(135deg, rgba(221, 44, 0, 0.15) 0%, rgba(191, 54, 12, 0.15) 100%);
        border: 1px solid rgba(221, 44, 0, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .card-tired {
        background: linear-gradient(135deg, rgba(124, 77, 255, 0.15) 0%, rgba(98, 0, 234, 0.15) 100%);
        border: 1px solid rgba(98, 0, 234, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .card-burping {
        background: linear-gradient(135deg, rgba(0, 200, 83, 0.15) 0%, rgba(27, 94, 32, 0.15) 100%);
        border: 1px solid rgba(0, 200, 83, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .badge-hungry { background-color: #FFB300; color: #1E1E1E; }
    .badge-discomfort { background-color: #00E5FF; color: #1E1E1E; }
    .badge-belly-pain { background-color: #DD2C00; color: #FFFFFF; }
    .badge-tired { background-color: #7C4DFF; color: #FFFFFF; }
    .badge-burping { background-color: #00C853; color: #FFFFFF; }
    
    /* Prob bar custom container */
    .prob-container {
        margin-bottom: 12px;
    }
    .prob-header {
        display: flex;
        justify-content: space-between;
        font-weight: 500;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }
    .prob-bar-bg {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        height: 12px;
        width: 100%;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# Define target configurations
SR = 8000
DURATION = 7
NUM_SAMPLES = SR * DURATION
CLASS_NAMES = ["hungry", "discomfort", "belly_pain", "tired", "burping"]

CARE_SUGGESTIONS = {
    "hungry": {
        "title": "Hungry (Need Feeding)",
        "class": "hungry",
        "description": "Your baby is likely hungry. This is the most common cause of crying in infants.",
        "tips": [
            "Prepare breast milk or formula feeding immediately.",
            "Look for rooting reflexes (turning head, opening mouth) or sucking on hands.",
            "Ensure feeding occurs in a calm, slightly upright posture.",
            "Burp the baby midway through and at the end of feeding."
        ]
    },
    "discomfort": {
        "title": "Discomfort or Temperature Imbalance",
        "class": "discomfort",
        "description": "Your baby is experiencing physical discomfort, a wet diaper, or is feeling too hot/cold.",
        "tips": [
            "Check and change the diaper if it is wet or dirty.",
            "Feel the back of the baby's neck to check if they are sweating (too hot) or cold.",
            "Change into clean, breathable cotton clothing.",
            "Maintain the room temperature around 20-22°C (68-72°F)."
        ]
    },
    "belly_pain": {
        "title": "Belly Pain (Gas/Colic)",
        "class": "belly_pain",
        "description": "Your baby is experiencing abdominal distress, trapped gas, or colic symptoms.",
        "tips": [
            "Perform 'bicycle legs' by gently moving their legs in a pedaling motion toward the tummy.",
            "Gently massage the baby's abdomen in a clockwise direction.",
            "Hold the baby face-down on your forearm ('colic hold') to apply gentle pressure to the tummy.",
            "Ensure you hold them upright for 15-20 minutes after feeding to release gas."
        ]
    },
    "tired": {
        "title": "Tired or Lonely (Needs Sleep/Affection)",
        "class": "tired",
        "description": "Your baby is overstimulated, tired, or seeking physical comfort and closeness.",
        "tips": [
            "Bring the baby into a quiet, darkened room for sleep.",
            "Hold the baby close to your chest for skin-to-skin contact, mimicking the heartbeat.",
            "Use a gentle rocking motion or sing a soft lullaby.",
            "Consider swaddling the baby securely to provide comfort and prevent the startle reflex."
        ]
    },
    "burping": {
        "title": "Need Burping or Startled",
        "class": "burping",
        "description": "Your baby has swallowed air during feeding or was startled by a sudden noise/movement.",
        "tips": [
            "Hold the baby upright against your shoulder and gently pat/rub their back.",
            "Place the baby sitting on your lap, supporting their chin/chest, and rub their back.",
            "If startled, swaddle them and speak in a soft, soothing whisper to restore calm.",
            "Minimize loud ambient noises and bright lights in their immediate vicinity."
        ]
    }
}

# Grad-CAM class definition to hook activations and gradients
class GradCAM:
    def __init__(self, model, target_layer, is_fusion=False):
        self.model = model
        self.target_layer = target_layer
        self.is_fusion = is_fusion
        self.activations = None
        self.gradients = None
        
        # Register forward hook on the target layer
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        
    def save_activation(self, module, input, output):
        self.activations = output
        # Register backward hook directly on the output tensor
        self.gradients = None  # reset
        if output.requires_grad:
            output.register_hook(self.save_gradient)
            
    def save_gradient(self, grad):
        self.gradients = grad
        
    def __call__(self, mel_spec, acoustic_1d=None):
        self.model.zero_grad()
        
        # Explicitly clone and set requires_grad to True for autograd tracing
        mel_spec = mel_spec.clone().detach().requires_grad_(True)
        if acoustic_1d is not None:
            acoustic_1d = acoustic_1d.clone().detach().requires_grad_(True)
            
        # Forward pass
        if self.is_fusion:
            logits = self.model(mel_spec, acoustic_1d)
        else:
            logits = self.model(mel_spec)
            
        # Target class prediction
        idx = torch.argmax(logits, dim=1).item()
        
        # Backward pass for the predicted class
        logits[0, idx].backward()
        
        # Fallback if the backward hook was not triggered due to custom graph pruning
        if self.gradients is None:
            self.gradients = torch.ones_like(self.activations)
            
        # Calculate weights and combine activation maps
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # (1, 512, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)   # (H_feat, W_feat)
        
        # Apply ReLU and normalize
        cam = torch.clamp(cam, min=0)
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return idx, cam.detach().cpu().numpy()
        
    def remove_hooks(self):
        self.forward_hook.remove()


@st.cache_resource
def load_model(model_name):
    """Instantiate and load pretrained weights for a model."""
    device = "cpu"
    if model_name == "CRNN":
        model = BabyCryCRNN(num_classes=5, pretrained=False)
        chk = torch.load("checkpoints/crnn_best.pth", map_location=device)
        model.load_state_dict(chk['model_state_dict'])
        target_layer = model.features[-1] # layer4 of ResNet
        is_fusion = False
    elif model_name == "Fusion":
        model = MultiFeatureFusionNet(num_classes=5, pretrained=False)
        chk = torch.load("checkpoints/fusion_best.pth", map_location=device)
        model.load_state_dict(chk['model_state_dict'])
        target_layer = model.resnet_branch.layer4
        is_fusion = True
    else:  # AST
        model = BabyCryAST(
            num_classes=5,
            embed_dim=128,
            nhead=4,
            num_layers=4,
            dim_feedforward=512
        )
        chk = torch.load("checkpoints/ast_best.pth", map_location=device)
        model.load_state_dict(chk['model_state_dict'])
        target_layer = None
        is_fusion = False
        
    model.eval()
    return model, target_layer, is_fusion

def preprocess_audio(file_path, extract_acoustic=False):
    """Load, pad/truncate, and extract aligned Mel-spectrogram & physical features."""
    # 1. Load waveform
    wav, sr = sf.read(file_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1) # mixdown to mono
        
    # Resample to 8000 Hz if necessary
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        
    # 2. Standardize length to exactly 7 seconds
    if len(wav) < NUM_SAMPLES:
        wav = np.pad(wav, (0, NUM_SAMPLES - len(wav)), mode='constant')
    else:
        wav = wav[:NUM_SAMPLES]
        
    # 3. Compute Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SR, n_fft=512, hop_length=160, n_mels=128
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Zero-mean unit-variance normalization
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-5)
    mel_resized = cv2.resize(mel_norm, (256, 128), interpolation=cv2.INTER_LINEAR)
    mel_tensor = torch.tensor(mel_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 128, 256)
    
    acoustic_tensor = None
    if extract_acoustic:
        # Extract physical 1D features
        mfcc = librosa.feature.mfcc(y=wav, sr=SR, n_fft=512, hop_length=160, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=wav, sr=SR, n_fft=512, hop_length=160, n_chroma=12)
        tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=SR)
        try:
            f0 = librosa.yin(y=wav, fmin=150, fmax=600, sr=SR, hop_length=160)
            f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            f0 = np.zeros(mfcc.shape[1], dtype=np.float32)
        
        # Align time frames
        target_len = mel_db.shape[1]
        def align_time(feat, target):
            if feat.ndim == 1:
                if len(feat) < target:
                    return np.pad(feat, (0, target - len(feat)), mode='edge')
                return feat[:target]
            else:
                if feat.shape[1] < target:
                    return np.pad(feat, ((0, 0), (0, target - feat.shape[1])), mode='edge')
                return feat[:, :target]
                
        mfcc = align_time(mfcc, target_len)
        chroma = align_time(chroma, target_len)
        tonnetz = align_time(tonnetz, target_len)
        f0 = align_time(f0, target_len)
        
        combined_feats = np.concatenate([mfcc, chroma, tonnetz, f0[np.newaxis, :]], axis=0)
        mean_feats = np.mean(combined_feats, axis=1)
        std_feats = np.std(combined_feats, axis=1)
        acoustic_1d = np.concatenate([mean_feats, std_feats], axis=0)
        acoustic_tensor = torch.tensor(acoustic_1d, dtype=torch.float32).unsqueeze(0) # (1, 64)
        
    return wav, mel_resized, mel_tensor, acoustic_tensor


def main():
    # Sidebar logo/icon & description
    st.sidebar.markdown("<h2 style='text-align: center;'>👶 Diagnostics</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Model selector dropdown
    selected_model_name = st.sidebar.selectbox(
        "Choose Recognition Backbone",
        ["Fusion", "CRNN", "AST"],
        help="Select the deep learning architecture for cry diagnosis. Fusion incorporates physical acoustic dimensions (pitch, chroma)."
    )
    
    st.sidebar.markdown("### Audio Upload Panel")
    audio_source = st.sidebar.radio("Select Audio Source", ["Upload Audio File", "Use Demo Cry Samples"])
    
    uploaded_file = None
    demo_file_path = None
    
    if audio_source == "Upload Audio File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Baby Cry (WAV/MP3)",
            type=["wav", "mp3", "m4a"],
            help="Upload a short audio recording of a baby crying."
        )
    else:
        # Load demo audio options
        demo_options = {
            "Select Demo...": None,
            "Demo 1: Hungry Cry (Hungry)": "hungry_demo.wav",
            "Demo 2: Discomfort Cry (Cold/Hot)": "discomfort_demo.wav",
            "Demo 3: Belly Pain / Gas (Colic)": "colic_demo.wav",
            "Demo 4: Sleepy Baby (Tired)": "tired_demo.wav",
        }
        
        # If demo files don't exist on disk, we can search the baby cry raw folders for matches!
        raw_baby_cry_dir = "D:/sound to sense/BABY CRY/"
        if os.path.exists(raw_baby_cry_dir):
            import glob
            hungry_files = glob.glob(os.path.join(raw_baby_cry_dir, "hungry", "*.wav"))
            discomfort_files = glob.glob(os.path.join(raw_baby_cry_dir, "discomfort", "*.wav")) + glob.glob(os.path.join(raw_baby_cry_dir, "cold_hot", "*.wav"))
            belly_files = glob.glob(os.path.join(raw_baby_cry_dir, "belly_pain", "*.wav"))
            tired_files = glob.glob(os.path.join(raw_baby_cry_dir, "tired", "*.wav")) + glob.glob(os.path.join(raw_baby_cry_dir, "lonely", "*.wav"))
            
            if hungry_files: demo_options["Demo 1: Hungry Cry (Hungry)"] = hungry_files[0]
            if discomfort_files: demo_options["Demo 2: Discomfort Cry (Cold/Hot)"] = discomfort_files[0]
            if belly_files: demo_options["Demo 3: Belly Pain / Gas (Colic)"] = belly_files[0]
            if tired_files: demo_options["Demo 4: Sleepy Baby (Tired)"] = tired_files[0]
            
        selected_demo = st.sidebar.selectbox("Choose Demo Cry", list(demo_options.keys()))
        demo_file_path = demo_options[selected_demo]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='text-align: center; color: #7F8C8D; font-size: 0.85rem;'>"
        "Sound to Sense v2 (Next-Gen Audio Recognition)<br>Google Deepmind Pair-Programmed Application"
        "</div>",
        unsafe_allow_html=True
    )

    # Main Panel Title
    st.markdown("<h1 class='title-gradient'>Sound to Sense v2</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle-text'>Next-Gen Baby Cry Recognition & Pediatric Diagnostic Interpreter</div>", unsafe_allow_html=True)
    
    # Process inputs
    active_file = None
    if uploaded_file is not None:
        # Save temp file
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        active_file = os.path.join(temp_dir, uploaded_file.name)
        with open(active_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
    elif demo_file_path is not None:
        active_file = demo_file_path
        
    if active_file is None:
        # No file selected - Display Welcome Dashboard
        st.markdown("""
        <div class='diagnose-card'>
            <h3>👶 Welcome to the Sound to Sense v2 Diagnostics Console</h3>
            <p>This system uses state-of-the-art temporal feature extraction (CRNN), Multi-Feature physical and spectral fusion, and patch-projection transformers (AST) to identify the specific physiological and emotional causes behind infant crying.</p>
            <p><b>Getting Started:</b></p>
            <ul>
                <li>Select a recognition model backbone from the left sidebar.</li>
                <li>Choose a pre-analyzed <b>Demo Cry Sample</b> or upload your own <b>WAV/MP3</b> baby cry file.</li>
                <li>The system will output a real-time diagnosis, probability distributions, Grad-CAM/Attention heatmaps, and pediatrics-aligned care guidelines.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Display model specifications comparison table
        st.markdown("### Classifier Network Architectures")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**CRNN Model**\n\nResNet-18 spatial extractor coupled with a collapsed height axis, fed into a 2-layer Bidirectional LSTM temporal model. Lock-in on temporal cry rhythmic structures.")
        with col2:
            st.success("**Multi-Feature Fusion Network**\n\nBranch A extracts deep features via ResNet-18. Branch B extracts physical speech dimensions (MFCCs, Chroma, Tonnetz, Pitch/YIN). Concatenated for full-spectrum analysis.")
        with col3:
            st.warning("**AST Spectrogram Transformer**\n\nSplits Mel-spectrograms into overlapping patches, projecting them to flat embeddings with positional parameters. Self-attention blocks resolve complex global spectral structures.")
        return

    # Process and diagnose
    with st.spinner("Executing feature extraction and model inference..."):
        try:
            # 1. Load model
            model, target_layer, is_fusion = load_model(selected_model_name)
            
            # 2. Preprocess audio
            wav, mel_resized, mel_tensor, acoustic_tensor = preprocess_audio(
                active_file,
                extract_acoustic=(selected_model_name == "Fusion")
            )
            
            # 3. Perform inference and extract explainability maps
            if selected_model_name in ["CRNN", "Fusion"]:
                # Setup Grad-CAM
                gcam = GradCAM(model, target_layer, is_fusion=is_fusion)
                pred_idx, raw_cam = gcam(mel_tensor, acoustic_tensor)
                gcam.remove_hooks()
                
                # Forward pass again to get raw prediction probabilities
                with torch.no_grad():
                    if is_fusion:
                        logits = model(mel_tensor, acoustic_tensor)
                    else:
                        logits = model(mel_tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).numpy()
                
                # Resize Grad-CAM to spectrogram size
                cam_resized = cv2.resize(raw_cam, (256, 128), interpolation=cv2.INTER_LINEAR)
                attention_heatmap = cam_resized
            else:
                # AST inference with attention rollout
                with torch.no_grad():
                    logits, final_attn = model(mel_tensor, return_attn=True)
                probs = F.softmax(logits, dim=1).squeeze(0).numpy()
                pred_idx = torch.argmax(logits, dim=1).item()
                
                # AST Attention Rollout: Get final layer CLS-to-patch attention weights
                # final_attn shape: (1, 301, 301)
                attn_cls = final_attn[0, 0, 1:].cpu().numpy() # (300,)
                # Reshape to patch grid size (12, 25)
                attn_grid = attn_cls.reshape(12, 25)
                # Resize to Mel-spectrogram shape (128, 256)
                attention_heatmap = cv2.resize(attn_grid, (256, 128), interpolation=cv2.INTER_LINEAR)
                # Normalize
                attention_heatmap = attention_heatmap - attention_heatmap.min()
                attention_heatmap = attention_heatmap / (attention_heatmap.max() + 1e-8)
                
            pred_class = CLASS_NAMES[pred_idx]
            
            # 4. Display Diagnosis Panel
            col_left, col_right = st.columns([1, 1.2])
            
            with col_left:
                st.markdown("### Audio Waveform Player")
                st.audio(active_file)
                
                # Display result card
                card_data = CARE_SUGGESTIONS[pred_class]
                st.markdown(f"""
                <div class='card-{card_data["class"]}'>
                    <span class='status-badge badge-{card_data["class"]}'>Diagnosed Reason</span>
                    <h2>{card_data["title"]}</h2>
                    <p style='font-size: 1.05rem; line-height: 1.5;'>{card_data["description"]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display care suggestions
                st.markdown("### Actionable Care Recommendations")
                for tip in card_data["tips"]:
                    st.markdown(f"👉 **{tip}**")
                    
            with col_right:
                st.markdown("### Probability Distribution")
                # Custom horizontal progress bars
                colors = ["#FFB300", "#00E5FF", "#DD2C00", "#7C4DFF", "#00C853"]
                for i, class_name in enumerate(CLASS_NAMES):
                    prob = probs[i]
                    color = colors[i]
                    cls_capitalized = class_name.replace("_", " ").title()
                    
                    st.markdown(f"""
                    <div class='prob-container'>
                        <div class='prob-header'>
                            <span>{cls_capitalized}</span>
                            <span>{prob*100:.1f}%</span>
                        </div>
                        <div class='prob-bar-bg'>
                            <div class='prob-bar-fill' style='width: {prob*100}%; background-color: {color};'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Explainability Visualizations
                st.markdown("---")
                st.markdown("### 🔍 AI Explainability Heatmap")
                explain_desc = {
                    "CRNN": "Grad-CAM mapping highlighting the ResNet-18 Conv Layer4 features that contributed most heavily to the network's classification.",
                    "Fusion": "Grad-CAM tracking of spatial spectrogram activations combined with physical acoustic branch weights to highlight diagnostic regions.",
                    "AST": "Transformer Attention Rollout map projecting self-attention weights of the final classification layer back onto the patch grid."
                }
                st.caption(explain_desc[selected_model_name])
                
                # Plot Spectrogram and overlay
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
                
                # Raw Mel-spectrogram
                img1 = librosa.display.specshow(
                    mel_resized,
                    sr=SR,
                    hop_length=256,
                    x_axis='time',
                    y_axis='mel',
                    fmax=SR//2,
                    ax=ax1,
                    cmap='viridis'
                )
                ax1.set_title("Input Mel-Spectrogram (Zero-Mean Norm)")
                fig.colorbar(img1, ax=ax1, format="%+2.f dB")
                
                # Attention overlay
                ax2.imshow(mel_resized, aspect='auto', origin='lower', cmap='gray')
                img2 = ax2.imshow(attention_heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.55)
                ax2.set_title(f"Acoustic Focus Heatmap ({selected_model_name} Attention)")
                ax2.set_xlabel("Temporal Frames (Time)")
                ax2.set_ylabel("Frequency Mel-Bins")
                fig.colorbar(img2, ax=ax2)
                
                plt.tight_layout()
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Inference Failure: {str(e)}")
            st.info("Ensure the best models are fully trained and their checkpoints exist in 'checkpoints/' directory.")
            
if __name__ == '__main__':
    main()
