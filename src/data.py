#!/usr/bin/env python3
"""
Data loading and preprocessing module for WESAD-like physiological data.
Handles data loading, windowing, feature matrix creation, and normalization.
"""

import numpy as np
import pandas as pd
import zipfile
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml
import logging
from tqdm import tqdm
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Load and preprocess WESAD-like physiological data."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize data loader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_path = Path(self.config['data']['data_path'])
        self.output_path = Path(self.config['data']['output_path'])
        self.output_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        for subdir in ['cv', 'shap', 'robustness', 'corr', 'hparams']:
            (self.output_path / subdir).mkdir(exist_ok=True)
    
    def load_wesad_data(self, subject_id: str = None) -> Dict[str, np.ndarray]:
        """Load WESAD data for a specific subject from CSV files."""
        if subject_id is None:
            subject_id = self.config['data']['subject_id']
        
        logger.info(f"Loading WESAD data for subject {subject_id}...")
        
        subject_path = self.data_path / subject_id
        zip_file = subject_path / f"{subject_id}_E4_Data.zip"
        
        if not zip_file.exists():
            raise FileNotFoundError(f"Subject data not found: {zip_file}")
        
        # Extract and load data from zip file
        data = {}
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # List contents
            file_list = zip_ref.namelist()
            logger.info(f"Files in archive: {file_list}")
            
            # Load each signal
            for filename in file_list:
                if filename.endswith('.csv') and filename != 'tags.csv':
                    signal_name = filename.replace('.csv', '').upper()
                    
                    # Read CSV file
                    with zip_ref.open(filename) as f:
                        try:
                            df = pd.read_csv(f, header=None)
                            logger.info(f"Loaded {signal_name}: {df.shape}")
                            
                            # Extract signal data
                            if signal_name in ['ACC', 'BVP', 'EDA', 'TEMP']:
                                if signal_name == 'ACC':
                                    # ACC has 3 columns (X, Y, Z)
                                    signal_data = df.values
                                    data['ACC_X'] = signal_data[:, 0]
                                    data['ACC_Y'] = signal_data[:, 1]
                                    data['ACC_Z'] = signal_data[:, 2]
                                else:
                                    # Single column signals
                                    signal_data = df.iloc[:, 0].values
                                    data[signal_name] = signal_data
                                    
                        except Exception as e:
                            logger.warning(f"Error loading {signal_name}: {e}")
                            continue
        
        return data
    
    def create_windows(self, data: Dict[str, np.ndarray], 
                      window_len: int = None, stride: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows from time series data."""
        if window_len is None:
            window_len = self.config['preprocessing']['window_len']
        if stride is None:
            stride = self.config['preprocessing']['stride']
        
        logger.info(f"Creating windows with length {window_len} and stride {stride}")
        
        # Find the minimum length across all signals
        min_length = min(len(signal) for signal in data.values())
        logger.info(f"Minimum signal length: {min_length}")
        
        # Create windows
        windows = []
        labels = []
        
        # Create synthetic labels for demonstration
        # In practice, you would load actual labels from tags.csv or similar
        n_windows = (min_length - window_len) // stride + 1
        
        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + window_len
            
            # Extract features for this window
            window_features = []
            for signal_name, signal_data in data.items():
                window_data = signal_data[start_idx:end_idx]
                
                # Compute basic statistics as features
                features = [
                    np.mean(window_data),
                    np.std(window_data),
                    np.min(window_data),
                    np.max(window_data),
                    np.percentile(window_data, 25),
                    np.percentile(window_data, 75),
                    np.median(window_data)
                ]
                window_features.extend(features)
            
            windows.append(window_features)
            
            # Create synthetic labels (0=baseline, 1=stress, 2=amusement, 3=meditation)
            # In practice, load actual labels
            label = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
            labels.append(label)
        
        X = np.array(windows, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        
        logger.info(f"Created {len(windows)} windows with {X.shape[1]} features")
        return X, y
    
    def apply_class_filtering(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply class filtering and aggregation based on config."""
        drop_classes = self.config['classes']['drop_classes']
        aggregate_to_one = self.config['classes']['aggregate_to_one']
        class_mapping = self.config['classes']['class_mapping']
        
        # Check configuration validity
        if drop_classes and aggregate_to_one:
            raise ValueError("Cannot have both drop_classes and aggregate_to_one active")
        
        # Create class balance summary
        unique_classes, counts = np.unique(y, return_counts=True)
        class_balance_before = pd.DataFrame({
            'class': unique_classes,
            'count': counts,
            'percentage': counts / len(y) * 100
        })
        
        logger.info("Class balance before filtering:")
        logger.info(class_balance_before)
        
        if drop_classes:
            # Remove specified classes
            mask = ~np.isin(y, drop_classes)
            X = X[mask]
            y = y[mask]
            logger.info(f"Removed classes {drop_classes}")
            
        elif aggregate_to_one:
            # Aggregate specified classes to "resting"
            # First, relabel classes to be consecutive
            unique_classes = np.unique(y)
            class_mapping_dict = {old: new for new, old in enumerate(unique_classes)}
            y = np.array([class_mapping_dict[label] for label in y])
            
            # Now aggregate the specified classes
            for old_class in aggregate_to_one:
                if int(old_class) in y:
                    # Map to the first class (0)
                    y[y == int(old_class)] = 0
            logger.info(f"Aggregated classes {aggregate_to_one} to 'resting'")
        
        # Create class balance summary after
        unique_classes, counts = np.unique(y, return_counts=True)
        class_balance_after = pd.DataFrame({
            'class': unique_classes,
            'count': counts,
            'percentage': counts / len(y) * 100
        })
        
        logger.info("Class balance after filtering:")
        logger.info(class_balance_after)
        
        # Save class balance summary
        class_balance_df = pd.concat([
            class_balance_before.assign(stage='before'),
            class_balance_after.assign(stage='after')
        ])
        class_balance_df.to_csv(self.output_path / 'class_balance.csv', index=False)
        
        return X, y
    
    def normalize_features(self, X: np.ndarray, method: str = None) -> Tuple[np.ndarray, StandardScaler]:
        """Normalize features using specified method."""
        if method is None:
            method = self.config['preprocessing']['normalize_method']
        
        logger.info(f"Normalizing features using {method} method")
        
        if method == "global":
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X)
        elif method == "per_subject":
            # For per-subject normalization, we'd need subject IDs
            # For now, use global normalization
            logger.warning("Per-subject normalization not implemented yet, using global")
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Save scaler
        import joblib
        joblib.dump(scaler, self.output_path / 'scaler.joblib')
        
        return X_normalized, scaler
    
    def make_splits(self, X: np.ndarray, y: np.ndarray, 
                   groups: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Create train/val/test splits based on configuration."""
        cv_config = self.config['cv']
        random_state = cv_config['random_state']
        
        if cv_config['type'] == 'repeated':
            from sklearn.model_selection import RepeatedStratifiedKFold
            
            cv = RepeatedStratifiedKFold(
                n_splits=cv_config['n_splits'],
                n_repeats=cv_config['n_repeats'],
                random_state=random_state
            )
            splits = list(cv.split(X, y))
        else:
            # Simple train/test split
            train_idx, test_idx = train_test_split(
                range(len(X)), test_size=0.2, 
                stratify=y, random_state=random_state
            )
            splits = [(train_idx, test_idx)]
        
        return splits
    
    def prepare_data(self, subject_id: str = None) -> Tuple[np.ndarray, np.ndarray, List]:
        """Complete data preparation pipeline."""
        # Load raw data
        raw_data = self.load_wesad_data(subject_id)
        
        # Create windows
        X, y = self.create_windows(raw_data)
        
        # Apply class filtering
        X, y = self.apply_class_filtering(X, y)
        
        # Normalize features
        X, scaler = self.normalize_features(X)
        
        # Create splits
        splits = self.make_splits(X, y)
        
        logger.info(f"Final data shape: {X.shape}")
        logger.info(f"Number of splits: {len(splits)}")
        
        return X, y, splits

def main():
    """CLI entry point for data preparation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare WESAD-like data for analysis")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--subject", help="Subject ID to analyze")
    
    args = parser.parse_args()
    
    # Load and prepare data
    loader = DataLoader(args.config)
    X, y, splits = loader.prepare_data(args.subject)
    
    # Save prepared data
    np.save(loader.output_path / "X.npy", X)
    np.save(loader.output_path / "y.npy", y)
    
    # Save splits
    import pickle
    with open(loader.output_path / "splits.pkl", 'wb') as f:
        pickle.dump(splits, f)
    
    logger.info("Data preparation completed successfully")

if __name__ == "__main__":
    main() 