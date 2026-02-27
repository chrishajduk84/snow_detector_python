# ===========================================================================
# Copyright (C) 2022 Infineon Technologies AG
#
# Radar Material Classification using Deep Learning
#
# This script uses radar signatures to classify materials.
# Supports multiple model architectures:
#   - 1D CNN with Self-Attention (recommended for spectral classification)
#   - LSTM for temporal analysis
#   - Hybrid CNN-LSTM for complex signatures
#
# Usage:
#   1. Collect data: python range_material-map.py collect <material_name>
#   2. Train model:  python range_material-map.py train
#   3. Live predict: python range_material-map.py predict
# ===========================================================================

import os
import sys
import json
import pprint
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import joblib
import seaborn as sns

from ifxAvian import Avian
from helpers.fft_spectrum import *
from helpers.DopplerAlgo import DopplerAlgo


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class AttentionBlock(nn.Module):
    """Self-attention for highlighting important frequency components."""
    
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch, C, L = x.shape
        
        q = self.query(x).view(batch, -1, L).permute(0, 2, 1)  # B x L x C'
        k = self.key(x).view(batch, -1, L)  # B x C' x L
        v = self.value(x).view(batch, -1, L)  # B x C x L
        
        attention = F.softmax(torch.bmm(q, k), dim=-1)  # B x L x L
        out = torch.bmm(v, attention.permute(0, 2, 1))  # B x C x L
        
        return self.gamma * out + x


class CNNAttentionClassifier(nn.Module):
    """
    1D CNN with Self-Attention for radar material classification.
    
    Architecture:
    - 3 Conv1D blocks with BatchNorm and ReLU
    - Self-attention layer to focus on important frequencies
    - Global average pooling
    - Fully connected classifier
    
    Best for: Spectral patterns, frequency-domain features
    """
    
    def __init__(self, input_size, num_classes, num_channels=64):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, num_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_channels, num_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(num_channels * 2, num_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_channels * 4),
            nn.ReLU(),
        )
        
        self.attention = AttentionBlock(num_channels * 4)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(num_channels * 4, num_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_channels * 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, input_size) -> (batch, 1, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attention(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        
        return x
    
    def get_attention_weights(self, x):
        """Extract attention weights for interpretability."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Get attention
        batch, C, L = x.shape
        q = self.attention.query(x).view(batch, -1, L).permute(0, 2, 1)
        k = self.attention.key(x).view(batch, -1, L)
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        
        return attention


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for radar material classification.
    
    Good for: Temporal patterns, sequences, time-varying materials
    """
    
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, input_size) -> (batch, input_size, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state from both directions
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        hidden = torch.cat([h_forward, h_backward], dim=1)
        
        out = self.classifier(hidden)
        return out


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM for complex radar signatures.
    
    CNN extracts spectral features, LSTM captures temporal dependencies.
    Best for: Complex materials with both spectral and temporal characteristics
    """
    
    def __init__(self, input_size, num_classes, cnn_channels=32, lstm_hidden=64):
        super().__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden, num_classes)
        )
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # CNN features: (batch, channels, length)
        cnn_out = self.cnn(x)
        
        # Reshape for LSTM: (batch, length, channels)
        cnn_out = cnn_out.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)
        
        # Use last hidden state
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        hidden = torch.cat([h_forward, h_backward], dim=1)
        
        out = self.classifier(hidden)
        return out


class SimpleCNNClassifier(nn.Module):
    """
    Lightweight CNN for small datasets (<1000 samples).
    Much less prone to overfitting than CNNAttentionClassifier.
    ~20,000 parameters.
    """
    
    def __init__(self, input_size, num_classes):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.3),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.3),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(8)
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TinyClassifier(nn.Module):
    """
    Minimal neural network for very small datasets (<500 samples).
    ~2,000-40,000 parameters depending on input size.
    """
    
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


# =============================================================================
# 2D CNN MODELS (Paper-style: raw time-domain input)
# =============================================================================

class PaperCNN2D(nn.Module):
    """
    2D CNN matching the IEEE Sensors paper architecture.
    Uses RAW time-domain signals as input (not FFT-processed).
    
    Input: (batch, 1, num_rx, num_samples) e.g., (batch, 1, 3, 128)
    Paper used: 4 RX channels, 256 samples
    Your setup: 3 RX channels, 128 samples
    
    ~10,000-50,000 parameters depending on input size.
    """
    
    def __init__(self, num_rx=3, num_samples=128, num_classes=6):
        super().__init__()
        
        self.num_rx = num_rx
        self.num_samples = num_samples
        
        # Conv layer 1: 64 filters, 3x3 kernel (like paper)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25)  # Spatial dropout to prevent overfitting
        )
        
        # Conv layer 2: 32 filters, 2x2 kernel (like paper)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(2, 2), padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.25)  # Spatial dropout
        )
        
        # Calculate flattened size
        # After conv1 (padding=1): (64, num_rx, num_samples)
        # After conv2 (padding=0): (32, num_rx-1, num_samples-1)
        flat_size = 32 * (num_rx - 1) * (num_samples - 1)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)  # Heavy dropout before FC layer
        self.fc = nn.Linear(flat_size, num_classes)
        
    def forward(self, x):
        # Input: (batch, 1, num_rx, num_samples)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class PaperCNN2DDeeper(nn.Module):
    """
    Deeper 2D CNN with pooling - more robust version of paper's architecture.
    Uses global average pooling to handle variable input sizes.
    
    Input: (batch, 1, num_rx, num_samples)
    ~5,000-10,000 parameters.
    """
    
    def __init__(self, num_rx=3, num_samples=128, num_classes=6):
        super().__init__()
        
        self.num_rx = num_rx
        self.num_samples = num_samples
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # Pool only in time dimension
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            
            # Block 3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Global average pooling - makes it input-size agnostic
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# =============================================================================
# TRADITIONAL ML MODELS (sklearn-based)
# =============================================================================

class SklearnModelWrapper:
    """
    Wrapper for sklearn models to provide consistent interface.
    Recommended for small datasets (<1000 samples).
    """
    
    AVAILABLE_MODELS = ['random_forest', 'svm', 'gradient_boosting']
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = self._create_model(model_type)
        
    def _create_model(self, model_type):
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'svm':
            return SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X, y):
        """Train the model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        """Predict class labels."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def score(self, X, y):
        """Return accuracy score."""
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)
    
    def save(self, filepath):
        """Save model to file."""
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'model_type': self.model_type}, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load model from file."""
        data = joblib.load(filepath)
        wrapper = cls(data['model_type'])
        wrapper.model = data['model']
        wrapper.scaler = data['scaler']
        return wrapper


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_radar_signals(data_dir: str, num_samples_per_class: int = 3):
    """
    Visualize radar signals from each material class to diagnose data issues.
    
    This helps identify:
    - If materials have distinguishable signatures
    - Collection artifacts or noise
    - Inconsistent data between samples
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' not found.")
        return
    
    material_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    num_materials = len(material_dirs)
    
    if num_materials == 0:
        print("No materials found in data directory.")
        return
    
    print(f"\nVisualizing {num_samples_per_class} samples from {num_materials} materials...")
    print("="*60)
    
    # Create figure with subplots for each material
    fig, axes = plt.subplots(num_materials, num_samples_per_class + 2, 
                             figsize=(4 * (num_samples_per_class + 2), 3 * num_materials))
    
    if num_materials == 1:
        axes = axes.reshape(1, -1)
    
    # Color maps for different visualizations
    cmap = 'viridis'
    
    for i, material_dir in enumerate(material_dirs):
        material_name = material_dir.name
        npy_files = sorted(list(material_dir.glob("*.npy")))
        
        if len(npy_files) == 0:
            print(f"  {material_name}: No samples found")
            continue
        
        print(f"  {material_name}: {len(npy_files)} samples")
        
        # Load samples
        samples = []
        for f in npy_files[:min(num_samples_per_class * 10, len(npy_files))]:
            data = np.load(f)
            if np.iscomplexobj(data):
                data = np.abs(data)
            samples.append(data)
        
        # Plot individual samples
        for j in range(min(num_samples_per_class, len(samples))):
            ax = axes[i, j]
            sample = samples[j]
            
            if sample.ndim == 2:
                # 2D data: (num_rx, num_samples)
                im = ax.imshow(sample, aspect='auto', cmap=cmap)
                ax.set_xlabel('Sample')
                ax.set_ylabel('RX Channel')
            else:
                # 1D data: flattened features
                ax.plot(sample)
                ax.set_xlabel('Feature Index')
                ax.set_ylabel('Value')
            
            if j == 0:
                ax.set_title(f'{material_name.upper()}\nSample {j+1}')
            else:
                ax.set_title(f'Sample {j+1}')
        
        # Plot mean signal across all samples
        ax_mean = axes[i, num_samples_per_class]
        samples_array = np.array(samples[:20])  # Use up to 20 samples for mean
        mean_signal = samples_array.mean(axis=0)
        std_signal = samples_array.std(axis=0)
        
        if mean_signal.ndim == 2:
            im = ax_mean.imshow(mean_signal, aspect='auto', cmap=cmap)
            ax_mean.set_title(f'Mean (n={len(samples_array)})')
            ax_mean.set_xlabel('Sample')
            ax_mean.set_ylabel('RX Channel')
        else:
            ax_mean.plot(mean_signal)
            ax_mean.fill_between(range(len(mean_signal)), 
                                mean_signal - std_signal, 
                                mean_signal + std_signal, alpha=0.3)
            ax_mean.set_title(f'Mean ± Std (n={len(samples_array)})')
        
        # Plot standard deviation (shows variability)
        ax_std = axes[i, num_samples_per_class + 1]
        if std_signal.ndim == 2:
            im = ax_std.imshow(std_signal, aspect='auto', cmap='hot')
            ax_std.set_title('Std Dev (variability)')
            ax_std.set_xlabel('Sample')
            ax_std.set_ylabel('RX Channel')
        else:
            ax_std.plot(std_signal, color='red')
            ax_std.set_title('Std Dev')
    
    plt.tight_layout()
    plt.savefig('radar_signals_visualization.png', dpi=150)
    plt.show()
    
    print(f"\nVisualization saved to 'radar_signals_visualization.png'")
    
    # Also create a comparison plot showing all materials overlaid
    _plot_material_comparison(data_path, material_dirs)


def _plot_material_comparison(data_path, material_dirs):
    """Create overlay comparison of all materials."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(material_dirs)))
    
    for i, material_dir in enumerate(material_dirs):
        material_name = material_dir.name
        npy_files = list(material_dir.glob("*.npy"))[:20]
        
        if len(npy_files) == 0:
            continue
        
        # Load and average samples
        samples = []
        for f in npy_files:
            data = np.load(f)
            if np.iscomplexobj(data):
                data = np.abs(data)
            samples.append(data)
        
        samples_array = np.array(samples)
        mean_signal = samples_array.mean(axis=0)
        
        if mean_signal.ndim == 2:
            # Plot each RX channel
            for ch in range(min(3, mean_signal.shape[0])):
                axes[ch].plot(mean_signal[ch], label=material_name, 
                            color=colors[i], linewidth=2, alpha=0.8)
                axes[ch].set_title(f'RX Channel {ch+1}')
                axes[ch].set_xlabel('Sample Index')
                axes[ch].set_ylabel('Amplitude')
                axes[ch].legend(loc='upper right')
                axes[ch].grid(True, alpha=0.3)
        else:
            # 1D data
            axes[0].plot(mean_signal, label=material_name, color=colors[i])
            axes[0].legend()
    
    plt.suptitle('Material Comparison: Mean Signals by RX Channel', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('material_comparison.png', dpi=150)
    plt.show()
    
    print(f"Comparison saved to 'material_comparison.png'")
    print("\nLook for:")
    print("  - Materials with similar curves = hard to distinguish")
    print("  - High std dev = inconsistent data collection")
    print("  - Distinct patterns = good discriminative features")


# =============================================================================
# DATASET
# =============================================================================

class RadarMaterialDataset(Dataset):
    """Dataset for radar material classification."""
    
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        
        self._load_data()
        
    def _load_data(self):
        """Load all .npy files from data directory."""
        material_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        all_labels = []
        for material_dir in material_dirs:
            material_name = material_dir.name
            npy_files = list(material_dir.glob("*.npy"))
            
            for npy_file in npy_files:
                data = np.load(npy_file)
                self.samples.append(data)
                all_labels.append(material_name)
        
        if len(all_labels) == 0:
            raise ValueError(f"No data found in {self.data_dir}. Collect data first!")
        
        # Encode labels
        self.labels = self.label_encoder.fit_transform(all_labels)
        self.class_names = list(self.label_encoder.classes_)
        
        print(f"Loaded {len(self.samples)} samples from {len(self.class_names)} classes")
        print(f"Classes: {self.class_names}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx].astype(np.float32)
        label = self.labels[idx]
        
        # Normalize
        sample = (sample - sample.mean()) / (sample.std() + 1e-8)
        
        if self.transform:
            sample = self.transform(sample)
        
        return torch.from_numpy(sample), torch.tensor(label, dtype=torch.long)


class RawRadarDataset(Dataset):
    """
    Dataset using RAW time-domain signals like the IEEE Sensors research paper.
    No FFT processing - let the 2D CNN learn the features directly.
    
    This approach preserves spatial relationships between antenna channels
    and temporal relationships in ADC samples.
    """
    
    def __init__(self, data_dir: str, num_rx: int = 3, num_samples: int = 128):
        self.data_dir = Path(data_dir)
        self.num_rx = num_rx
        self.num_samples = num_samples
        self.samples = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        
        self._load_data()
        
    def _load_data(self):
        """Load raw radar frames from data directory."""
        material_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        all_labels = []
        for material_dir in material_dirs:
            material_name = material_dir.name
            npy_files = list(material_dir.glob("*.npy"))
            
            for npy_file in npy_files:
                data = np.load(npy_file)
                self.samples.append(data)
                all_labels.append(material_name)
        
        if len(all_labels) == 0:
            raise ValueError(f"No data found in {self.data_dir}. Collect data first!")
        
        # Encode labels
        self.labels = self.label_encoder.fit_transform(all_labels)
        self.class_names = list(self.label_encoder.classes_)
        
        # Detect data format from first sample
        sample_shape = self.samples[0].shape
        if len(sample_shape) == 2 and sample_shape[0] == self.num_rx:
            self.data_format = 'raw_2d'  # (num_rx, num_samples)
        elif len(sample_shape) == 1:
            self.data_format = 'processed_1d'  # Flattened features
        else:
            self.data_format = 'unknown'
        
        print(f"Loaded {len(self.samples)} samples from {len(self.class_names)} classes")
        print(f"Data format: {self.data_format}, shape: {sample_shape}")
        print(f"Classes: {self.class_names}")
        
        # Print class distribution
        print("Class distribution:")
        for i, name in enumerate(self.class_names):
            count = sum(1 for l in self.labels if l == i)
            print(f"  {name}: {count} samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        raw = self.samples[idx]
        label = self.labels[idx]
        
        # Handle different data formats
        if self.data_format == 'raw_2d':
            # Raw 2D data: (num_rx, num_samples)
            sample = raw.astype(np.float32)
            
            # If complex, take magnitude
            if np.iscomplexobj(sample):
                sample = np.abs(sample)
            
            # Standardize each channel independently (like the paper)
            for ch in range(sample.shape[0]):
                sample[ch] = (sample[ch] - sample[ch].mean()) / (sample[ch].std() + 1e-8)
            
            # Add channel dimension for Conv2d: (1, num_rx, num_samples)
            sample = sample[np.newaxis, :, :]
            
        else:
            # Processed 1D data - reshape to 2D for compatibility
            sample = raw.astype(np.float32)
            sample = (sample - sample.mean()) / (sample.std() + 1e-8)
            
            # Reshape to (1, 1, length) - treat as 1 channel, 1 row
            sample = sample.reshape(1, 1, -1)
        
        return torch.from_numpy(sample), torch.tensor(label, dtype=torch.long)


# =============================================================================
# TRAINING
# =============================================================================

class MaterialClassifierTrainer:
    """Trainer for radar material classification models."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', is_2d_model=True):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        self.is_2d_model = is_2d_model  # 2D models keep shape, 1D models need flatten
        
    def _prepare_input(self, inputs):
        """Prepare input for model - flatten for 1D models."""
        if self.is_2d_model:
            return inputs  # Keep (batch, 1, num_rx, num_samples)
        else:
            # Flatten for 1D models: (batch, 1, num_rx, num_samples) -> (batch, num_rx * num_samples)
            return inputs.view(inputs.size(0), -1)
        
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, class_weights=None):
        """Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            lr: Learning rate
            class_weights: Optional array of weights for each class to handle imbalanced data
        """
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            for inputs, labels in train_loader:
                inputs = self._prepare_input(inputs)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = self._prepare_input(inputs)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss / len(train_loader))
            self.history['val_loss'].append(val_loss / len(val_loader))
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_material_model.pth')
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.1f}% | "
                      f"Val Loss={val_loss/len(val_loader):.4f}, Acc={val_acc:.1f}%")
        
        print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
        return self.history
    
    def evaluate(self, test_loader, class_names, is_2d_model=None):
        """Evaluate model and show confusion matrix."""
        # Allow override for evaluate call
        if is_2d_model is not None:
            self.is_2d_model = is_2d_model
            
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = self._prepare_input(inputs)
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Material Classification Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
    def plot_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history['train_loss'], label='Train')
        ax1.plot(self.history['val_loss'], label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        
        ax2.plot(self.history['train_acc'], label='Train')
        ax2.plot(self.history['val_acc'], label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()


# =============================================================================
# DATA COLLECTION
# =============================================================================

class MaterialDataCollector:
    """Collect radar data for training material classifier."""
    
    def __init__(self, save_dir: str = "material_data"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Radar configuration optimized for material sensing
        self.config = Avian.DeviceConfig(
            sample_rate_Hz = 1_000_000,
            rx_mask = 7,                      # All 3 RX antennas
            tx_mask = 1,
            if_gain_dB = 33,
            tx_power_level = 31,
            start_frequency_Hz = 60e9,
            end_frequency_Hz = 61.5e9,
            num_chirps_per_frame = 64,        # Fewer chirps for faster capture
            num_samples_per_chirp = 128,      # More samples for better range resolution
            chirp_repetition_time_s = 0.0005,
            frame_repetition_time_s = 0.1,    # 10 Hz
            mimo_mode = 'off'
        )
        
    def collect_material_samples(self, material_name: str, num_samples: int = 100):
        """
        Collect radar samples for a specific material.
        
        Place the radar sensor on/near the material and run this function.
        """
        material_dir = self.save_dir / material_name
        material_dir.mkdir(exist_ok=True)
        
        existing_samples = len(list(material_dir.glob("*.npy")))
        
        print(f"\n{'='*50}")
        print(f"Collecting samples for: {material_name.upper()}")
        print(f"Existing samples: {existing_samples}")
        print(f"Target: {num_samples} new samples")
        print(f"{'='*50}")
        print("\nPlace sensor on material and press ENTER to start...")
        input()
        
        with Avian.Device() as device:
            device.set_config(self.config)
            
            num_rx = 3  # From rx_mask = 7
            doppler = DopplerAlgo(
                self.config.num_samples_per_chirp,
                num_chirps_per_frame=self.config.num_chirps_per_frame,
                num_ant=num_rx
            )
            
            for i in range(num_samples):
                frame = device.get_next_frame()
                
                # Extract features from radar data
                features = self._extract_features(frame, doppler, num_rx)
                
                # Save sample
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = material_dir / f"{material_name}_{timestamp}.npy"
                np.save(filename, features)
                
                if (i + 1) % 10 == 0:
                    print(f"  Collected {i + 1}/{num_samples} samples")
            
        print(f"\nDone! Saved {num_samples} samples to {material_dir}")
        
    def _extract_features(self, frame, doppler, num_rx):
        """
        Extract features from radar frame for material classification.
        
        Features include:
        - Range profile (amplitude vs range)
        - Phase information across antennas
        - Frequency response characteristics
        """
        features = []
        
        for i_ant in range(num_rx):
            mat = frame[i_ant, :, :]
            
            # Range-Doppler map
            rd_map = doppler.compute_doppler_map(mat, i_ant)
            
            # Range profile: max across Doppler (target regardless of velocity)
            range_profile = np.max(np.abs(rd_map), axis=1)
            features.append(range_profile)
            
            # Doppler profile: max across range (velocity signature)
            doppler_profile = np.max(np.abs(rd_map), axis=0)
            features.append(doppler_profile)
            
            # Phase profile at peak range
            peak_range_idx = np.argmax(range_profile)
            phase_profile = np.angle(rd_map[peak_range_idx, :])
            features.append(phase_profile)
        
        # Concatenate all features
        return np.concatenate(features)
    
    def list_collected_materials(self):
        """List all collected materials and sample counts."""
        print("\nCollected Materials:")
        print("-" * 40)
        
        total = 0
        for material_dir in sorted(self.save_dir.iterdir()):
            if material_dir.is_dir():
                count = len(list(material_dir.glob("*.npy")))
                print(f"  {material_dir.name}: {count} samples")
                total += count
        
        print("-" * 40)
        print(f"  Total: {total} samples")
        
        return total
    
    def collect_raw_samples(self, material_name: str, num_samples: int = 100):
        """
        Collect RAW radar samples (like the IEEE Sensors research paper).
        Saves raw time-domain ADC samples without FFT processing.
        
        This is the RECOMMENDED method for training 2D CNN models.
        """
        material_dir = self.save_dir / material_name
        material_dir.mkdir(exist_ok=True)
        
        existing_samples = len(list(material_dir.glob("*.npy")))
        
        print(f"\n{'='*50}")
        print(f"Collecting RAW samples for: {material_name.upper()}")
        print(f"Existing samples: {existing_samples}")
        print(f"Target: {num_samples} new samples")
        print(f"{'='*50}")
        print("\nThis collects RAW time-domain data (no FFT processing).")
        print("Use with 'paper_cnn' or 'paper_cnn_deep' models.")
        print("\nPlace sensor on material and press ENTER to start...")
        input()
        
        with Avian.Device() as device:
            device.set_config(self.config)
            
            num_rx = bin(self.config.rx_mask).count('1')  # Count active RX antennas
            
            for i in range(num_samples):
                frame = device.get_next_frame()
                
                # Save RAW data: take first chirp, all RX channels
                # frame shape: (num_rx, num_samples_per_chirp, num_chirps_per_frame)
                # We want: (num_rx, num_samples_per_chirp) from first chirp
                raw_data = self._extract_raw_frame(frame)
                
                # Save sample
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = material_dir / f"{material_name}_raw_{timestamp}.npy"
                np.save(filename, raw_data)
                
                if (i + 1) % 10 == 0:
                    print(f"  Collected {i + 1}/{num_samples} RAW samples")
            
        print(f"\nDone! Saved {num_samples} RAW samples to {material_dir}")
        print(f"Data shape: {raw_data.shape} (num_rx x num_samples)")
    
    def _extract_raw_frame(self, frame):
        """
        Extract raw radar frame with preprocessing for better material discrimination.
        
        Improvements over basic extraction:
        - Average across all chirps for better SNR (not just first chirp)
        - Apply Range-FFT to separate reflections by distance
        - Remove DC offset which dominates the signal
        
        Args:
            frame: Raw radar frame (num_rx, num_samples_per_chirp, num_chirps_per_frame)
            
        Returns:
            Processed frame data: (num_rx, num_range_bins) - range profile per antenna
        """
        # Average across ALL chirps for better SNR (instead of just first chirp)
        # frame shape: (num_rx, num_samples_per_chirp, num_chirps_per_frame)
        raw = np.mean(frame, axis=2)  # (num_rx, num_samples_per_chirp)
        
        if np.iscomplexobj(raw):
            # Apply Hanning window to reduce spectral leakage
            window = np.hanning(raw.shape[1])
            raw_windowed = raw * window
            
            # Range FFT - converts time-domain to range-domain
            # This separates the material reflection from antenna coupling
            range_fft = np.fft.fft(raw_windowed, axis=1)
            
            # Take magnitude, keep only positive frequencies (actual range bins)
            raw = np.abs(range_fft[:, :raw.shape[1]//2])
        else:
            raw = np.abs(raw)
        
        # Remove DC offset from each channel
        # The DC component dominates and hides material-specific differences
        for ch in range(raw.shape[0]):
            raw[ch] = raw[ch] - raw[ch].mean()
        
        return raw.astype(np.float32)


# =============================================================================
# LIVE INFERENCE
# =============================================================================

class MaterialClassifier:
    """Live material classification using trained model."""
    
    def __init__(self, model_path: str, class_names: list, model_type: str = 'cnn_attention'):
        self.class_names = class_names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Determine input size (will be set after first frame)
        self.input_size = None
        self.model = None
        self.model_type = model_type
        self.model_path = model_path
        self.is_sklearn_model = model_type in TRADITIONAL_ML_MODELS
        self.is_2d_model = model_type in MODELS_2D  # paper_cnn, paper_cnn_deep
        
        # Load sklearn model immediately (doesn't need input size)
        if self.is_sklearn_model:
            self.model = SklearnModelWrapper.load(model_path)
            print(f"Loaded sklearn model: {model_type}")
        
        # Radar configuration (must match training config)
        self.config = Avian.DeviceConfig(
            sample_rate_Hz = 1_000_000,
            rx_mask = 7,
            tx_mask = 1,
            if_gain_dB = 33,
            tx_power_level = 31,
            start_frequency_Hz = 60e9,
            end_frequency_Hz = 61.5e9,
            num_chirps_per_frame = 64,
            num_samples_per_chirp = 128,
            chirp_repetition_time_s = 0.0005,
            frame_repetition_time_s = 0.1,
            mimo_mode = 'off'
        )
        
    def _init_model(self, num_rx=3, num_samples=128):
        """Initialize PyTorch model with correct input size."""
        if self.is_sklearn_model:
            return  # sklearn models don't need initialization
        
        # All deep learning models use raw data
        # Calculate flattened size for 1D models
        input_size_1d = num_rx * num_samples
        
        if self.model_type == 'paper_cnn':
            self.model = PaperCNN2D(num_rx, num_samples, len(self.class_names))
        elif self.model_type == 'paper_cnn_deep':
            self.model = PaperCNN2DDeeper(num_rx, num_samples, len(self.class_names))
        elif self.model_type == 'cnn_attention':
            self.model = CNNAttentionClassifier(input_size_1d, len(self.class_names))
        elif self.model_type == 'lstm':
            self.model = LSTMClassifier(input_size_1d, len(self.class_names))
        elif self.model_type == 'hybrid':
            self.model = HybridCNNLSTM(input_size_1d, len(self.class_names))
        elif self.model_type == 'simple_cnn':
            self.model = SimpleCNNClassifier(input_size_1d, len(self.class_names))
        elif self.model_type == 'tiny':
            self.model = TinyClassifier(input_size_1d, len(self.class_names))
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def _extract_raw_frame(self, frame):
        """Extract raw frame for all deep learning models."""
        # Take first chirp from each RX channel
        raw = frame[:, :, 0]  # (num_rx, num_samples_per_chirp)
        
        if np.iscomplexobj(raw):
            raw = np.abs(raw)
        
        # Standardize each channel
        for ch in range(raw.shape[0]):
            raw[ch] = (raw[ch] - raw[ch].mean()) / (raw[ch].std() + 1e-8)
        
        return raw.astype(np.float32)
        
    def _extract_features(self, frame, doppler, num_rx):
        """Extract processed features for sklearn models only."""
        features = []
        
        for i_ant in range(num_rx):
            mat = frame[i_ant, :, :]
            rd_map = doppler.compute_doppler_map(mat, i_ant)
            
            range_profile = np.max(np.abs(rd_map), axis=1)
            features.append(range_profile)
            
            doppler_profile = np.max(np.abs(rd_map), axis=0)
            features.append(doppler_profile)
            
            peak_range_idx = np.argmax(range_profile)
            phase_profile = np.angle(rd_map[peak_range_idx, :])
            features.append(phase_profile)
        
        return np.concatenate(features)
    
    def run_live(self):
        """Run live material classification with visualization."""
        print("\n" + "="*50)
        print("LIVE MATERIAL CLASSIFICATION")
        print("="*50)
        print(f"Model: {self.model_type}")
        print(f"Classes: {self.class_names}")
        print(f"Device: {self.device}")
        print("Press Ctrl+C to stop\n")
        
        # Set up live plot
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.canvas.manager.set_window_title("Radar Material Classifier")
        
        with Avian.Device() as device:
            device.set_config(self.config)
            
            num_rx = 3
            num_samples = self.config.num_samples_per_chirp
            doppler = DopplerAlgo(
                self.config.num_samples_per_chirp,
                num_chirps_per_frame=self.config.num_chirps_per_frame,
                num_ant=num_rx
            )
            
            # For smoothing predictions
            prediction_history = []
            history_size = 5
            confidence_history = {name: [] for name in self.class_names}
            max_history = 50
            
            try:
                while True:
                    frame = device.get_next_frame()
                    
                    # Initialize PyTorch model on first frame (sklearn already loaded)
                    if self.model is None:
                        self._init_model(num_rx=num_rx, num_samples=num_samples)
                        print(f"Model initialized: {self.model_type}")
                    
                    # Extract data and predict based on model type
                    if self.is_sklearn_model:
                        # Processed features for sklearn
                        features = self._extract_features(frame, doppler, num_rx)
                        features_norm = (features - features.mean()) / (features.std() + 1e-8)
                        probs = self.model.predict_proba(features_norm.reshape(1, -1))[0]
                        pred_idx = np.argmax(probs)
                        pred_class = self.class_names[pred_idx]
                        confidence = probs[pred_idx] * 100
                    else:
                        # Raw time-domain for ALL deep learning models
                        raw_data = self._extract_raw_frame(frame)
                        
                        if self.is_2d_model:
                            # Shape: (1, 1, num_rx, num_samples) for Conv2d
                            x = torch.from_numpy(raw_data).unsqueeze(0).unsqueeze(0).to(self.device)
                        else:
                            # Flatten for 1D models: (1, num_rx * num_samples)
                            x = torch.from_numpy(raw_data.flatten()).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(x)
                            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
                            pred_idx = np.argmax(probs)
                            pred_class = self.class_names[pred_idx]
                            confidence = probs[pred_idx] * 100
                    
                    # Update prediction history for smoothing
                    prediction_history.append(pred_idx)
                    if len(prediction_history) > history_size:
                        prediction_history.pop(0)
                    
                    # Smoothed prediction (majority vote)
                    smoothed_pred_idx = max(set(prediction_history), key=prediction_history.count)
                    smoothed_pred_class = self.class_names[smoothed_pred_idx]
                    
                    # Update confidence history
                    for i, name in enumerate(self.class_names):
                        confidence_history[name].append(probs[i] * 100)
                        if len(confidence_history[name]) > max_history:
                            confidence_history[name].pop(0)
                    
                    # Clear and update plots
                    ax1.clear()
                    ax2.clear()
                    
                    # Bar chart of current probabilities
                    colors = ['green' if i == smoothed_pred_idx else 'steelblue' 
                             for i in range(len(self.class_names))]
                    bars = ax1.bar(self.class_names, probs * 100, color=colors)
                    ax1.set_ylim(0, 100)
                    ax1.set_ylabel('Confidence (%)')
                    ax1.set_title(f'Current Prediction: {smoothed_pred_class.upper()} ({confidence:.1f}%)',
                                 fontsize=14, fontweight='bold')
                    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5)
                    
                    # Add value labels on bars
                    for bar, prob in zip(bars, probs):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                                f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=10)
                    
                    # Time series of confidence
                    for name in self.class_names:
                        ax2.plot(confidence_history[name], label=name, linewidth=2)
                    ax2.set_ylim(0, 100)
                    ax2.set_xlabel('Frame')
                    ax2.set_ylabel('Confidence (%)')
                    ax2.set_title('Confidence History')
                    ax2.legend(loc='upper right')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.pause(0.01)
                    
                    # Print to console
                    print(f"\rPrediction: {smoothed_pred_class.upper():15s} | Confidence: {confidence:5.1f}%", end='')
                    
            except KeyboardInterrupt:
                print("\n\nStopped by user")
            finally:
                plt.ioff()
                plt.close()


# =============================================================================
# DATA CONVERSION
# =============================================================================

def convert_existing_data(data_dir: str = "material_data", backup: bool = True):
    """
    Convert existing raw data files by applying preprocessing.
    
    This applies:
    - Range-FFT to separate reflections by distance
    - DC offset removal to reveal material-specific differences
    
    After running this, you can remove the TEMPORARY PREPROCESSING block
    from RawRadarDataset.__getitem__().
    
    Args:
        data_dir: Directory containing material data
        backup: If True, create backup of original files
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' not found.")
        return
    
    # Create backup directory if needed
    if backup:
        backup_dir = data_path.parent / f"{data_path.name}_backup"
        if backup_dir.exists():
            print(f"Backup directory already exists: {backup_dir}")
            response = input("Continue without new backup? (y/n): ")
            if response.lower() != 'y':
                return
        else:
            import shutil
            print(f"Creating backup at: {backup_dir}")
            shutil.copytree(data_path, backup_dir)
            print("Backup complete.")
    
    print(f"\nConverting data in: {data_dir}")
    print("="*60)
    
    total_converted = 0
    total_skipped = 0
    
    for material_dir in sorted(data_path.iterdir()):
        if not material_dir.is_dir():
            continue
        
        material_name = material_dir.name
        npy_files = list(material_dir.glob("*.npy"))
        
        if len(npy_files) == 0:
            continue
        
        converted = 0
        skipped = 0
        
        for npy_file in npy_files:
            try:
                data = np.load(npy_file)
                
                # Apply preprocessing based on data type
                if np.iscomplexobj(data):
                    # Complex data: Apply Hanning window + Range FFT
                    window = np.hanning(data.shape[1])
                    data_windowed = data * window
                    
                    # Range FFT
                    range_fft = np.fft.fft(data_windowed, axis=1)
                    
                    # Take magnitude, positive frequencies only
                    processed = np.abs(range_fft[:, :data.shape[1]//2])
                else:
                    # Real data: just ensure it's float
                    processed = data.astype(np.float32)
                
                # ALWAYS remove DC offset from each channel (mean subtraction)
                for ch in range(processed.shape[0]):
                    processed[ch] = processed[ch] - processed[ch].mean()
                
                # Save processed data (overwrite original)
                np.save(npy_file, processed.astype(np.float32))
                converted += 1
                
            except Exception as e:
                print(f"  Error processing {npy_file.name}: {e}")
                skipped += 1
        
        print(f"  {material_name}: {converted} converted, {skipped} skipped")
        total_converted += converted
        total_skipped += skipped
    
    print("="*60)
    print(f"Total: {total_converted} files converted, {total_skipped} skipped")
    print(f"\nOriginal data backed up to: {data_path.parent / f'{data_path.name}_backup'}")
    print("\n" + "!"*60)
    print("IMPORTANT: Now remove the TEMPORARY PREPROCESSING block from")
    print("RawRadarDataset.__getitem__() in range_material-map.py")
    print("!"*60)


# =============================================================================
# MAIN - COMMAND LINE INTERFACE
# =============================================================================

# Available model types - ALL deep learning models use raw time-domain data
DEEP_LEARNING_MODELS = ['paper_cnn', 'paper_cnn_deep', 'cnn_attention', 'lstm', 'hybrid', 'simple_cnn', 'tiny']
# Models designed for 2D input (use Conv2d)
MODELS_2D = ['paper_cnn', 'paper_cnn_deep']
# Models designed for 1D input (will flatten raw data)
MODELS_1D = ['cnn_attention', 'lstm', 'hybrid', 'simple_cnn', 'tiny']
TRADITIONAL_ML_MODELS = ['random_forest', 'svm', 'gradient_boosting']
ALL_MODELS = DEEP_LEARNING_MODELS + TRADITIONAL_ML_MODELS


def print_usage():
    """Print usage instructions."""
    print("""
Radar Material Classification System
=====================================

Usage:
    python range_material-map.py <command> [options]

Commands:
    collect_raw <material> [n]  - Collect RAW data (time-domain, RECOMMENDED)
                                  Example: python range_material-map.py collect_raw wood 100

    collect <material> [n]      - Collect PROCESSED data (FFT features, for sklearn)
                                  Example: python range_material-map.py collect wood 100
                         
    list                        - List all collected materials and sample counts
    
    convert                     - Convert existing data with Range-FFT preprocessing
                                  Creates backup, then removes DC offset and applies FFT
                                  Example: python range_material-map.py convert
    
    visualize [n]               - Visualize radar signals from each material class
                                  Example: python range_material-map.py visualize 5
    
    train <model>               - Train the classification model
                                  Example: python range_material-map.py train paper_cnn
                         
    predict <model>             - Run live material classification
                                  Example: python range_material-map.py predict paper_cnn

Available Models:
    Deep Learning (all use raw time-domain data):
        paper_cnn        - 2D CNN matching IEEE Sensors paper (~10-50k params, RECOMMENDED)
        paper_cnn_deep   - Deeper 2D CNN with pooling (~5-10k params)
        cnn_attention    - 1D CNN with self-attention (~190k params)
        lstm             - Bidirectional LSTM (~130k params)
        hybrid           - CNN + LSTM combined (~80k params)
        simple_cnn       - Lightweight 1D CNN (~20k params)
        tiny             - Minimal MLP (~40k params)
    
    Traditional ML (use processed FFT features from 'collect'):
        random_forest     - Random Forest classifier
        svm               - Support Vector Machine with RBF kernel
        gradient_boosting - Gradient Boosting classifier

Recommendations:
    1. Use 'collect_raw' + 'paper_cnn' for best results (matches research paper)
    2. For small datasets (<1000): paper_cnn, paper_cnn_deep, or random_forest
    3. For large datasets (>5000): paper_cnn_deep or cnn_attention

Workflow (Recommended):
    1. Collect RAW samples for each material:
       python range_material-map.py collect_raw wood 100
       python range_material-map.py collect_raw metal 100
       python range_material-map.py collect_raw plastic 100
       
    2. Check collected data:
       python range_material-map.py list
       
    3. Train the model:
       python range_material-map.py train paper_cnn
       
    4. Run live classification:
       python range_material-map.py predict paper_cnn
""")


def main():
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'collect':
        if len(sys.argv) < 3:
            print("Error: Please specify material name")
            print("Example: python range_material-map.py collect wood")
            return
        
        material_name = sys.argv[2].lower()
        num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        
        collector = MaterialDataCollector()
        collector.collect_material_samples(material_name, num_samples)
    
    elif command == 'collect_raw':
        if len(sys.argv) < 3:
            print("Error: Please specify material name")
            print("Example: python range_material-map.py collect_raw wood")
            return
        
        material_name = sys.argv[2].lower()
        num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        
        collector = MaterialDataCollector()
        collector.collect_raw_samples(material_name, num_samples)
        
    elif command == 'list':
        collector = MaterialDataCollector()
        collector.list_collected_materials()
    
    elif command == 'visualize':
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        visualize_radar_signals('material_data', num_samples)
    
    elif command == 'convert':
        backup = '--no-backup' not in sys.argv
        convert_existing_data('material_data', backup=backup)
        
    elif command == 'train':
        model_type = sys.argv[2] if len(sys.argv) > 2 else 'paper_cnn'
        
        if model_type not in ALL_MODELS:
            print(f"Unknown model type: {model_type}")
            print(f"Available models: {', '.join(ALL_MODELS)}")
            return
        
        print(f"\nTraining {model_type} model...")
        print("="*50)
        
        # Load dataset
        data_dir = "material_data"
        if not Path(data_dir).exists():
            print(f"Error: No data directory found at '{data_dir}'")
            print("Please collect data first using: python range_material-map.py collect_raw <material>")
            return
        
        # Use appropriate dataset based on model type
        if model_type in DEEP_LEARNING_MODELS:
            # ALL deep learning models use raw time-domain data
            dataset = RawRadarDataset(data_dir, num_rx=3, num_samples=128)
        else:
            # Traditional ML models use processed FFT features
            dataset = RadarMaterialDataset(data_dir)
        
        if len(dataset) < 10:
            print("Error: Not enough samples. Collect at least 10 samples per class.")
            return
        
        # Save class names and model type for inference
        with open('class_names.json', 'w') as f:
            json.dump({'class_names': dataset.class_names, 'model_type': model_type}, f)
        
        num_classes = len(dataset.class_names)
        print(f"Number of classes: {num_classes}")
        print(f"Total samples: {len(dataset)}")
        
        # Different training paths based on model type
        if model_type in TRADITIONAL_ML_MODELS:
            # =====================
            # SKLEARN MODEL TRAINING
            # =====================
            print(f"\nUsing sklearn {model_type} model...")
            
            # Prepare data as numpy arrays (flatten if needed)
            X = []
            for i in range(len(dataset)):
                sample = dataset.samples[i].astype(np.float32)
                if sample.ndim > 1:
                    sample = sample.flatten()
                X.append(sample)
            X = np.array(X)
            y = np.array(dataset.labels)
            
            print(f"Feature vector size: {X.shape[1]}")
            
            # Normalize features
            X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
            
            # Split
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples: {len(X_val)}")
            
            # Create and train model
            model = SklearnModelWrapper(model_type)
            print(f"\nTraining {model_type}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            train_acc = model.score(X_train, y_train) * 100
            val_acc = model.score(X_val, y_val) * 100
            
            print(f"\nTraining accuracy: {train_acc:.1f}%")
            print(f"Validation accuracy: {val_acc:.1f}%")
            
            # Confusion matrix and classification report
            y_pred = model.predict(X_val)
            
            cm = confusion_matrix(y_val, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=dataset.class_names, yticklabels=dataset.class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'{model_type} - Material Classification Confusion Matrix')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png')
            plt.show()
            
            print("\nClassification Report:")
            print(classification_report(y_val, y_pred, target_names=dataset.class_names))
            
            # Save model
            model.save('best_material_model.joblib')
            print(f"\nModel saved to 'best_material_model.joblib'")
        
        else:
            # ========================
            # DEEP LEARNING TRAINING (all models use raw data)
            # ========================
            print(f"\nUsing deep learning model: {model_type}")
            
            # Get data shape from first sample
            sample, _ = dataset[0]
            _, num_rx, num_samples = sample.shape  # (1, num_rx, num_samples)
            print(f"Input shape: (1, {num_rx}, {num_samples})")
            
            # For 1D models, we need the flattened size
            input_size_1d = num_rx * num_samples
            
            # Split dataset
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            print(f"Training samples: {train_size}")
            print(f"Validation samples: {val_size}")
            
            # Create model based on type
            if model_type == 'paper_cnn':
                model = PaperCNN2D(num_rx, num_samples, num_classes)
            elif model_type == 'paper_cnn_deep':
                model = PaperCNN2DDeeper(num_rx, num_samples, num_classes)
            elif model_type == 'cnn_attention':
                model = CNNAttentionClassifier(input_size_1d, num_classes)
            elif model_type == 'lstm':
                model = LSTMClassifier(input_size_1d, num_classes)
            elif model_type == 'hybrid':
                model = HybridCNNLSTM(input_size_1d, num_classes)
            elif model_type == 'simple_cnn':
                model = SimpleCNNClassifier(input_size_1d, num_classes)
            elif model_type == 'tiny':
                model = TinyClassifier(input_size_1d, num_classes)
            else:
                print(f"Unknown model type: {model_type}")
                return
            
            # Print model parameter count
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {num_params:,}")
            
            # Calculate class weights for imbalanced data
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(dataset.labels),
                y=dataset.labels
            )
            print(f"\nClass weights (to handle imbalanced data):")
            for name, weight in zip(dataset.class_names, class_weights):
                print(f"  {name}: {weight:.2f}")
            
            # Train
            trainer = MaterialClassifierTrainer(model, is_2d_model=(model_type in MODELS_2D))
            trainer.train(train_loader, val_loader, epochs=100, class_weights=class_weights)
            trainer.plot_history()
            trainer.evaluate(val_loader, dataset.class_names, is_2d_model=(model_type in MODELS_2D))
            
            print(f"\nModel saved to 'best_material_model.pth'")
        
        print(f"Class names saved to 'class_names.json'")
        
    elif command == 'predict':
        # Load class names and model type
        if not Path('class_names.json').exists():
            print("Error: No trained model found.")
            print("Please train a model first using: python range_material-map.py train <model>")
            return
        
        with open('class_names.json', 'r') as f:
            config = json.load(f)
        
        # Handle both old and new format
        if isinstance(config, list):
            # Old format: just a list of class names
            class_names = config
            model_type = sys.argv[2] if len(sys.argv) > 2 else 'cnn_attention'
        else:
            # New format: dict with class_names and model_type
            class_names = config['class_names']
            model_type = sys.argv[2] if len(sys.argv) > 2 else config.get('model_type', 'cnn_attention')
        
        print(f"Using model type: {model_type}")
        
        # Check for appropriate model file
        if model_type in TRADITIONAL_ML_MODELS:
            if not Path('best_material_model.joblib').exists():
                print("Error: Model file 'best_material_model.joblib' not found.")
                print(f"Please train a {model_type} model first.")
                return
            classifier = MaterialClassifier('best_material_model.joblib', class_names, model_type)
        else:
            if not Path('best_material_model.pth').exists():
                print("Error: Model file 'best_material_model.pth' not found.")
                print(f"Please train a {model_type} model first.")
                return
            classifier = MaterialClassifier('best_material_model.pth', class_names, model_type)
        
        classifier.run_live()
        
    else:
        print(f"Unknown command: {command}")
        print_usage()


if __name__ == '__main__':
    main()
