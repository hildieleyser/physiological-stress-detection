#!/usr/bin/env python3
"""
WESAD Dataset Formatter for TimesNet (Version 2)
Formats WESAD physiological data from CSV files for TimesNet forecasting and classification tasks.
"""

import numpy as np
import pandas as pd
import os
import zipfile
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class WESADTimesNetFormatterV2:
    def __init__(self, data_path="WESAD", output_path="timesnet_data"):
        """
        Initialize the WESAD TimesNet formatter for CSV-based data.
        
        Args:
            data_path: Path to WESAD dataset
            output_path: Path to save formatted data
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_path / "train").mkdir(exist_ok=True)
        (self.output_path / "val").mkdir(exist_ok=True)
        (self.output_path / "test").mkdir(exist_ok=True)
        
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        
    def load_wesad_data(self, subject_id="S2"):
        """
        Load WESAD data for a specific subject from CSV files.
        
        Args:
            subject_id: Subject ID (e.g., 'S2', 'S3', etc.)
            
        Returns:
            dict: Dictionary containing signals and labels
        """
        print(f"Loading WESAD data for subject {subject_id}...")
        
        subject_path = self.data_path / subject_id
        zip_file = subject_path / f"{subject_id}_E4_Data.zip"
        
        if not zip_file.exists():
            raise FileNotFoundError(f"Subject data not found: {zip_file}")
        
        # Extract and load data from zip file
        data = {}
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # List contents
            file_list = zip_ref.namelist()
            print(f"Files in archive: {file_list}")
            
            # Load each signal
            for filename in file_list:
                if filename.endswith('.csv'):
                    signal_name = filename.replace('.csv', '').upper()
                    
                    # Read CSV file
                    with zip_ref.open(filename) as f:
                        df = pd.read_csv(f, header=None)
                    
                    # Extract signal data
                    if signal_name in ['ACC', 'BVP', 'EDA', 'TEMP']:
                        # These are the main physiological signals
                        if signal_name == 'ACC':
                            # ACC has 3 columns (X, Y, Z)
                            signal_data = df.values
                        else:
                            # Single column signals
                            signal_data = df.iloc[:, 0].values
                        
                        data[signal_name] = signal_data
                        print(f"Loaded {signal_name}: {signal_data.shape}")
        
        # Load labels from tags.csv if available
        try:
            with zip_ref.open('tags.csv') as f:
                tags_df = pd.read_csv(f, header=None)
                print(f"Loaded tags: {tags_df.shape}")
                # Process tags to create labels
                data['tags'] = tags_df
        except:
            print("No tags.csv found, will create synthetic labels")
            data['tags'] = None
        
        return data
    
    def create_labels_from_tags(self, data, signal_length):
        """
        Create labels from tags or generate synthetic labels.
        
        Args:
            data: Dictionary containing signals and tags
            signal_length: Length of the longest signal
            
        Returns:
            np.ndarray: Array of labels
        """
        if data.get('tags') is not None:
            # Create labels from tags
            tags_df = data['tags']
            labels = np.zeros(signal_length)
            
            # Map tags to labels (this would need to be customized based on your tag format)
            # For now, create synthetic labels
            print("Creating synthetic labels from tags...")
            
            # Simple synthetic labeling: alternate between states
            segment_length = signal_length // 4
            for i in range(4):
                start_idx = i * segment_length
                end_idx = min((i + 1) * segment_length, signal_length)
                labels[start_idx:end_idx] = i
            
        else:
            # Create synthetic labels for demonstration
            print("Creating synthetic labels...")
            labels = np.zeros(signal_length)
            
            # Create 4 segments with different labels
            segment_length = signal_length // 4
            for i in range(4):
                start_idx = i * segment_length
                end_idx = min((i + 1) * segment_length, signal_length)
                labels[start_idx:end_idx] = i
        
        return labels
    
    def align_signals(self, data):
        """
        Align all signals to the same length.
        
        Args:
            data: Dictionary containing signals
            
        Returns:
            tuple: (aligned_signals, labels, signal_names)
        """
        print("Aligning signals...")
        
        # Get available signals
        available_signals = {}
        for signal_name, signal_data in data.items():
            if signal_name in ['ACC', 'BVP', 'EDA', 'TEMP']:
                available_signals[signal_name] = signal_data
                print(f"Signal {signal_name}: {signal_data.shape}")
        
        if not available_signals:
            raise ValueError("No valid signals found")
        
        # Find minimum length for alignment
        min_length = float('inf')
        for signal_name, signal_data in available_signals.items():
            if signal_name == 'ACC':
                # ACC has 3 columns
                length = len(signal_data)
            else:
                length = len(signal_data)
            min_length = min(min_length, length)
        
        print(f"Minimum signal length: {min_length}")
        
        # Align signals to minimum length
        aligned_signals = {}
        for signal_name, signal_data in available_signals.items():
            if signal_name == 'ACC':
                # ACC: keep all 3 columns
                aligned_signals[signal_name] = signal_data[:min_length]
            else:
                # Single column signals
                aligned_signals[signal_name] = signal_data[:min_length]
        
        # Create labels
        labels = self.create_labels_from_tags(data, min_length)
        
        return aligned_signals, labels, list(aligned_signals.keys())
    
    def extract_signals(self, data, window_size=1000, overlap=0.5):
        """
        Extract and align signals from WESAD data.
        
        Args:
            data: WESAD data dictionary
            window_size: Size of time windows
            overlap: Overlap ratio between windows
            
        Returns:
            tuple: (signals, labels, signal_names)
        """
        print("Extracting and aligning signals...")
        
        # Align signals
        aligned_signals, labels, signal_names = self.align_signals(data)
        
        print(f"Available signals: {signal_names}")
        print(f"Signal length: {len(labels)}")
        
        # Create windows
        step_size = int(window_size * (1 - overlap))
        windows = []
        window_labels = []
        
        for i in range(0, len(labels) - window_size + 1, step_size):
            window_data = []
            
            for signal_name in sorted(aligned_signals.keys()):
                if signal_name == 'ACC':
                    # ACC has 3 columns
                    acc_data = aligned_signals[signal_name][i:i+window_size]
                    window_data.extend(acc_data.flatten())  # Flatten to 1D
                else:
                    # Single column signals
                    signal_data = aligned_signals[signal_name][i:i+window_size]
                    window_data.extend(signal_data)
            
            # Get majority label for the window
            window_label_data = labels[i:i+window_size]
            unique_labels, counts = np.unique(window_label_data, return_counts=True)
            majority_label = unique_labels[np.argmax(counts)]
            
            windows.append(np.array(window_data))
            window_labels.append(majority_label)
        
        # Reshape to [batch, sequence_length, num_features]
        # Calculate number of features
        num_features = 0
        for signal_name in sorted(aligned_signals.keys()):
            if signal_name == 'ACC':
                num_features += 3  # X, Y, Z
            else:
                num_features += 1
        
        reshaped_windows = []
        
        for window in windows:
            # Reshape from [window_size * num_features] to [window_size, num_features]
            reshaped = window.reshape(window_size, num_features)
            reshaped_windows.append(reshaped)
        
        return np.array(reshaped_windows), np.array(window_labels), signal_names
    
    def normalize_data(self, data, method='standard'):
        """
        Normalize the data.
        
        Args:
            data: Input data of shape [batch, seq_len, features]
            method: Normalization method ('standard', 'minmax', 'per_subject')
            
        Returns:
            np.ndarray: Normalized data
        """
        print(f"Normalizing data using {method} method...")
        
        if method == 'standard':
            # Global standardization
            scaler = StandardScaler()
            original_shape = data.shape
            data_reshaped = data.reshape(-1, data.shape[-1])
            data_normalized = scaler.fit_transform(data_reshaped)
            data_normalized = data_normalized.reshape(original_shape)
            self.scalers['global'] = scaler
            
        return data_normalized
    
    def encode_labels(self, labels):
        """
        Encode categorical labels.
        
        Args:
            labels: Raw labels
            
        Returns:
            np.ndarray: Encoded labels
        """
        print("Encoding labels...")
        encoded_labels = self.label_encoder.fit_transform(labels)
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        return encoded_labels
    
    def split_data(self, data, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split data into train/val/test sets.
        
        Args:
            data: Input data
            labels: Labels
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            tuple: (train_data, val_data, test_data, train_labels, val_labels, test_labels)
        """
        print("Splitting data into train/val/test sets...")
        
        # First split: train vs (val + test)
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            data, labels, test_size=(val_ratio + test_ratio), 
            stratify=labels, random_state=42
        )
        
        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_data, test_data, val_labels, test_labels = train_test_split(
            temp_data, temp_labels, test_size=(1 - val_ratio_adjusted),
            stratify=temp_labels, random_state=42
        )
        
        print(f"Train set: {train_data.shape[0]} samples")
        print(f"Validation set: {val_data.shape[0]} samples") 
        print(f"Test set: {test_data.shape[0]} samples")
        
        return train_data, val_data, test_data, train_labels, val_labels, test_labels
    
    def save_data(self, train_data, val_data, test_data, 
                  train_labels, val_labels, test_labels, 
                  signal_names, subject_id):
        """
        Save formatted data to files.
        
        Args:
            train_data, val_data, test_data: Data arrays
            train_labels, val_labels, test_labels: Label arrays
            signal_names: List of signal names
            subject_id: Subject identifier
        """
        print("Saving formatted data...")
        
        # Save data
        np.save(self.output_path / "train" / f"{subject_id}_train_data.npy", train_data)
        np.save(self.output_path / "val" / f"{subject_id}_val_data.npy", val_data)
        np.save(self.output_path / "test" / f"{subject_id}_test_data.npy", test_data)
        
        # Save labels
        np.save(self.output_path / "train" / f"{subject_id}_train_labels.npy", train_labels)
        np.save(self.output_path / "val" / f"{subject_id}_val_labels.npy", val_labels)
        np.save(self.output_path / "test" / f"{subject_id}_test_labels.npy", test_labels)
        
        # Save metadata
        metadata = {
            'signal_names': signal_names,
            'num_classes': len(np.unique(train_labels)),
            'sequence_length': train_data.shape[1],
            'num_features': train_data.shape[2],
            'label_mapping': dict(zip(self.label_encoder.classes_, 
                                    self.label_encoder.transform(self.label_encoder.classes_)))
        }
        
        import json
        with open(self.output_path / f"{subject_id}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Data saved to {self.output_path}")
        
    def create_timesnet_config(self, subject_id, task='classification'):
        """
        Create TimesNet configuration file.
        
        Args:
            subject_id: Subject identifier
            task: Task type ('classification' or 'forecasting')
        """
        print("Creating TimesNet configuration...")
        
        config = {
            'task_name': task,
            'model': 'TimesNet',
            'mode': 'forecast' if task == 'forecasting' else 'classification',
            'data': 'custom',
            'root_path': str(self.output_path),
            'data_path': f'{subject_id}_train_data.npy',
            'features': 'M',  # Multivariate
            'target': 'OT',
            'freq': 'h',
            'checkpoints': './checkpoints/',
            'num_workers': 0,
            'train_epochs': 10,
            'batch_size': 32,
            'patch_size': 16,
            'stride': 8,
            'max_seq_len': 1000,
            'hidden_size': 64,
            'num_layers': 2,
            'num_heads': 8,
            'dropout': 0.1,
            'learning_rate': 0.0001,
            'loss': 'MSE' if task == 'forecasting' else 'CrossEntropy',
            'lradj': 'type1',
            'use_amp': False,
            'inverse': False,
            'use_gpu': True,
            'gpu': 0,
            'use_multi_gpu': False,
            'devices': '0,1,2,3',
            'detail_freq': 100,
            'save': 'checkpoints',
            'model_id': f'{subject_id}_{task}',
            'model': 'TimesNet',
            'd_model': 64,
            'e_layers': 2,
            'd_ff': 256,
            'factor': 5,
            'factor_temporal': 5,
            'top_k': 5,
            'num_kernels': 6
        }
        
        import json
        config_path = self.output_path / f"{subject_id}_{task}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"Configuration saved to {config_path}")
        return config_path
    
    def visualize_data(self, data, labels, signal_names, subject_id):
        """
        Create visualizations of the formatted data.
        
        Args:
            data: Formatted data
            labels: Labels
            signal_names: Signal names
            subject_id: Subject identifier
        """
        print("Creating data visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sample time series
        sample_idx = 0
        sample_data = data[sample_idx]
        
        # Plot sample signals
        ax1 = axes[0, 0]
        for i, signal_name in enumerate(signal_names):
            if signal_name == 'ACC':
                # Plot ACC X, Y, Z separately
                for j in range(3):
                    ax1.plot(sample_data[:, i*3 + j], label=f'{signal_name}_{["X","Y","Z"][j]}', alpha=0.7)
            else:
                ax1.plot(sample_data[:, i], label=signal_name, alpha=0.7)
        ax1.set_title(f'Sample Time Series (Subject {subject_id})')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Signal Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Label distribution
        ax2 = axes[0, 1]
        unique_labels, counts = np.unique(labels, return_counts=True)
        ax2.bar(unique_labels, counts)
        ax2.set_title('Label Distribution')
        ax2.set_xlabel('Label')
        ax2.set_ylabel('Count')
        
        # Signal correlation
        ax3 = axes[1, 0]
        correlation_matrix = np.corrcoef(sample_data.T)
        im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_title('Signal Correlation Matrix')
        ax3.set_xticks(range(len(sample_data.T)))
        ax3.set_yticks(range(len(sample_data.T)))
        plt.colorbar(im, ax=ax3)
        
        # Data shape info
        ax4 = axes[1, 1]
        ax4.axis('off')
        info_text = f"""
        Data Shape: {data.shape}
        Sequence Length: {data.shape[1]}
        Features: {data.shape[2]}
        Samples: {data.shape[0]}
        Labels: {len(np.unique(labels))}
        """
        ax4.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(self.output_path / f"{subject_id}_data_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {self.output_path / f'{subject_id}_data_visualization.png'}")
    
    def format_single_subject(self, subject_id="S2"):
        """
        Format data for a single subject.
        
        Args:
            subject_id: Subject ID to process
        """
        print(f"Formatting data for subject {subject_id}...")
        
        # Load data
        data = self.load_wesad_data(subject_id)
        
        # Extract signals
        signals, labels, signal_names = self.extract_signals(data)
        
        # Normalize
        signals_normalized = self.normalize_data(signals, method='standard')
        
        # Encode labels
        labels_encoded = self.encode_labels(labels)
        
        # Split data
        train_data, val_data, test_data, train_labels, val_labels, test_labels = self.split_data(
            signals_normalized, labels_encoded
        )
        
        # Save data
        self.save_data(train_data, val_data, test_data, 
                      train_labels, val_labels, test_labels, 
                      signal_names, subject_id)
        
        # Create config
        self.create_timesnet_config(subject_id, 'classification')
        
        # Visualize
        self.visualize_data(signals_normalized, labels_encoded, signal_names, subject_id)
        
        return signals_normalized, labels_encoded, signal_names

def main():
    """Main function to run the formatter."""
    formatter = WESADTimesNetFormatterV2()
    
    # Format single subject
    print("=== Formatting Single Subject ===")
    formatter.format_single_subject("S2")
    
    print("\n=== Formatting Complete ===")
    print(f"Data saved to: {formatter.output_path}")
    print("Next steps:")
    print("1. Clone TimesNet repository")
    print("2. Install dependencies")
    print("3. Use the generated config files to run TimesNet")

if __name__ == "__main__":
    main() 