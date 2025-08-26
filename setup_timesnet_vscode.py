#!/usr/bin/env python3
"""
TimesNet Setup Script for VS Code
Sets up TimesNet environment and organizes WESAD data for training.
"""

import os
import subprocess
import sys
from pathlib import Path
import json

class TimesNetVSCodeSetup:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.timesnet_path = self.project_root / "TimesNet"
        self.data_path = self.project_root / "timesnet_data"
        
    def clone_timesnet_repo(self):
        """Clone the TimesNet repository."""
        print("Cloning TimesNet repository...")
        
        if self.timesnet_path.exists():
            print("TimesNet repository already exists, skipping clone...")
            return
            
        try:
            subprocess.run([
                "git", "clone", "https://github.com/thuml/TimesNet.git"
            ], cwd=self.project_root, check=True)
            print("✓ TimesNet repository cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error cloning repository: {e}")
            return False
            
        return True
    
    def install_dependencies(self):
        """Install TimesNet dependencies."""
        print("Installing TimesNet dependencies...")
        
        requirements_file = self.timesnet_path / "requirements.txt"
        
        if not requirements_file.exists():
            print("Creating basic requirements.txt...")
            basic_requirements = [
                "torch>=1.8.0",
                "torchvision>=0.9.0",
                "numpy>=1.19.0",
                "pandas>=1.2.0",
                "scikit-learn>=0.24.0",
                "matplotlib>=3.3.0",
                "seaborn>=0.11.0",
                "tqdm>=4.60.0",
                "tensorboard>=2.4.0"
            ]
            
            with open(requirements_file, 'w') as f:
                f.write('\n'.join(basic_requirements))
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            print("✓ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing dependencies: {e}")
            return False
            
        return True
    
    def create_data_loader(self):
        """Create a custom data loader for WESAD data."""
        print("Creating custom data loader...")
        
        data_loader_path = self.timesnet_path / "data_provider" / "wesad_loader.py"
        data_loader_path.parent.mkdir(exist_ok=True)
        
        data_loader_code = '''"""
Custom Data Loader for WESAD Dataset
Adapted for TimesNet framework.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path

class WESADDataset(Dataset):
    def __init__(self, data_path, split='train', task='classification'):
        """
        Initialize WESAD dataset.
        
        Args:
            data_path: Path to formatted data
            split: 'train', 'val', or 'test'
            task: 'classification' or 'forecasting'
        """
        self.data_path = Path(data_path)
        self.split = split
        self.task = task
        
        # Load data
        self.data = np.load(self.data_path / split / f"all_subjects_{split}_data.npy")
        self.labels = np.load(self.data_path / split / f"all_subjects_{split}_labels.npy")
        
        # Load metadata
        with open(self.data_path / "all_subjects_metadata.json", 'r') as f:
            self.metadata = json.load(f)
            
        print(f"Loaded {split} data: {self.data.shape}")
        print(f"Labels: {np.unique(self.labels, return_counts=True)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        
        if self.task == 'classification':
            y = torch.LongTensor([self.labels[idx]])
        else:  # forecasting
            # For forecasting, use the next timestep as target
            if idx < len(self.data) - 1:
                y = torch.FloatTensor(self.data[idx + 1])
            else:
                y = torch.FloatTensor(self.data[idx])
                
        return x, y
    
    def get_metadata(self):
        return self.metadata

def create_wesad_dataloader(data_path, batch_size=32, task='classification'):
    """
    Create data loaders for WESAD dataset.
    
    Args:
        data_path: Path to formatted data
        batch_size: Batch size
        task: Task type
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = WESADDataset(data_path, 'train', task)
    val_dataset = WESADDataset(data_path, 'val', task)
    test_dataset = WESADDataset(data_path, 'test', task)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the data loader
    data_path = "../timesnet_data"
    train_loader, val_loader, test_loader = create_wesad_dataloader(data_path)
    
    print("Data loader test:")
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: data shape {data.shape}, labels shape {labels.shape}")
        if batch_idx >= 2:  # Just test first few batches
            break
'''
        
        with open(data_loader_path, 'w') as f:
            f.write(data_loader_code)
            
        print(f"✓ Custom data loader created: {data_loader_path}")
    
    def create_training_script(self):
        """Create a training script for TimesNet with WESAD data."""
        print("Creating training script...")
        
        training_script_path = self.project_root / "train_timesnet_wesad.py"
        
        training_script = '''#!/usr/bin/env python3
"""
TimesNet Training Script for WESAD Dataset
Trains TimesNet model on WESAD physiological data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add TimesNet to path
sys.path.append('./TimesNet')

from data_provider.wesad_loader import create_wesad_dataloader
from models import TimesNet

class TimesNetTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader = create_wesad_dataloader(
            config['data_path'], 
            batch_size=config['batch_size'],
            task=config['task']
        )
        
        # Initialize model
        self.model = self._create_model()
        self.criterion = self._create_criterion()
        self.optimizer = self._create_optimizer()
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def _create_model(self):
        """Create TimesNet model."""
        # Load metadata for model configuration
        with open(Path(self.config['data_path']) / "all_subjects_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        model_config = {
            'seq_len': metadata['sequence_length'],
            'pred_len': 1,  # For classification
            'enc_in': metadata['num_features'],
            'dec_in': metadata['num_features'],
            'c_out': metadata['num_classes'] if self.config['task'] == 'classification' else metadata['num_features'],
            'd_model': self.config['d_model'],
            'e_layers': self.config['e_layers'],
            'd_ff': self.config['d_ff'],
            'factor': self.config['factor'],
            'factor_temporal': self.config['factor_temporal'],
            'top_k': self.config['top_k'],
            'num_kernels': self.config['num_kernels'],
            'dropout': self.config['dropout']
        }
        
        model = TimesNet.Model(model_config)
        model.to(self.device)
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        return model
    
    def _create_criterion(self):
        """Create loss function."""
        if self.config['task'] == 'classification':
            return nn.CrossEntropyLoss()
        else:
            return nn.MSELoss()
    
    def _create_optimizer(self):
        """Create optimizer."""
        return optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, labels) in enumerate(tqdm(self.train_loader, desc="Training")):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            if self.config['task'] == 'classification':
                labels = labels.squeeze()
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            if self.config['task'] == 'classification':
                loss = self.criterion(outputs, labels)
            else:
                loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in tqdm(self.val_loader, desc="Validating"):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                if self.config['task'] == 'classification':
                    labels = labels.squeeze()
                
                outputs = self.model(data)
                
                if self.config['task'] == 'classification':
                    loss = self.criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                else:
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        if self.config['task'] == 'classification':
            accuracy = 100 * correct / total
            print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            return avg_loss, accuracy
        else:
            print(f"Validation Loss: {avg_loss:.4f}")
            return avg_loss
    
    def train(self):
        """Train the model."""
        print("Starting training...")
        
        for epoch in range(self.config['epochs']):
            print(f"\\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            if self.config['task'] == 'classification':
                val_loss, val_acc = self.validate()
                self.val_losses.append(val_loss)
            else:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"best_model_{self.config['task']}.pth")
                print("✓ Best model saved")
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training curves."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Curves - {self.config["task"]}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'training_curves_{self.config["task"]}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Training curves saved")

def main():
    """Main training function."""
    # Configuration
    config = {
        'data_path': './timesnet_data',
        'task': 'classification',  # or 'forecasting'
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'd_model': 64,
        'e_layers': 2,
        'd_ff': 256,
        'factor': 5,
        'factor_temporal': 5,
        'top_k': 5,
        'num_kernels': 6,
        'dropout': 0.1
    }
    
    # Create trainer and train
    trainer = TimesNetTrainer(config)
    trainer.train()
    
    print("Training completed!")

if __name__ == "__main__":
    main()
'''
        
        with open(training_script_path, 'w') as f:
            f.write(training_script)
            
        print(f"✓ Training script created: {training_script_path}")
    
    def create_launch_config(self):
        """Create VS Code launch configuration."""
        print("Creating VS Code launch configuration...")
        
        vscode_path = self.project_root / ".vscode"
        vscode_path.mkdir(exist_ok=True)
        
        launch_config = {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Train TimesNet WESAD",
                    "type": "python",
                    "request": "launch",
                    "program": "${workspaceFolder}/train_timesnet_wesad.py",
                    "console": "integratedTerminal",
                    "cwd": "${workspaceFolder}",
                    "env": {
                        "PYTHONPATH": "${workspaceFolder}"
                    }
                },
                {
                    "name": "Format WESAD Data",
                    "type": "python",
                    "request": "launch",
                    "program": "${workspaceFolder}/wesad_timesnet_formatter.py",
                    "console": "integratedTerminal",
                    "cwd": "${workspaceFolder}"
                }
            ]
        }
        
        launch_path = vscode_path / "launch.json"
        with open(launch_path, 'w') as f:
            json.dump(launch_config, f, indent=2)
            
        print(f"✓ VS Code launch configuration created: {launch_path}")
    
    def create_readme(self):
        """Create a README file with setup instructions."""
        print("Creating README...")
        
        readme_content = '''# TimesNet with WESAD Dataset

This project sets up TimesNet for training on the WESAD (Wearable Stress and Affect Detection) dataset.

## Setup Instructions

### 1. Data Formatting
First, format your WESAD data for TimesNet:
```bash
python wesad_timesnet_formatter.py
```

This will:
- Load WESAD data from the `WESAD/` directory
- Extract and align physiological signals
- Normalize the data
- Split into train/val/test sets
- Save formatted data to `timesnet_data/`
- Create TimesNet configuration files

### 2. Environment Setup
The setup script will:
- Clone the TimesNet repository
- Install dependencies
- Create custom data loaders
- Set up VS Code configurations

Run the setup:
```bash
python setup_timesnet_vscode.py
```

### 3. Training
Train TimesNet on your WESAD data:
```bash
python train_timesnet_wesad.py
```

Or use VS Code debugger with the provided launch configurations.

## Data Structure

After formatting, your data will be organized as:
```
timesnet_data/
├── train/
│   ├── all_subjects_train_data.npy
│   └── all_subjects_train_labels.npy
├── val/
│   ├── all_subjects_val_data.npy
│   └── all_subjects_val_labels.npy
├── test/
│   ├── all_subjects_test_data.npy
│   └── all_subjects_test_labels.npy
└── all_subjects_metadata.json
```

## Model Configuration

The model is configured for:
- **Task**: Classification (stress/emotion detection)
- **Input**: Multivariate time series (7 physiological signals)
- **Sequence Length**: 1000 time steps
- **Features**: 7 (ACC, BVP, EDA, EMG, ECG, Resp, Temp)
- **Classes**: 4 (baseline, amusement, stress, meditation)

## Troubleshooting

### Memory Issues
- Reduce batch size in `train_timesnet_wesad.py`
- Use CPU if GPU memory is insufficient
- Reduce sequence length in data formatting

### CUDA Errors
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Use CPU: Change device in training script

### Data Shape Errors
- Verify data format: `python -c "import numpy as np; data=np.load('timesnet_data/train/all_subjects_train_data.npy'); print(data.shape)"`
- Expected shape: `(samples, 1000, 7)`

## Performance Tips

1. **GPU Usage**: Enable GPU in training script
2. **Batch Size**: Start with 32, adjust based on memory
3. **Sequence Length**: 1000 is good for WESAD, can be reduced for speed
4. **Data Augmentation**: Consider adding noise or time shifts

## Results

Training will save:
- Best model: `best_model_classification.pth`
- Training curves: `training_curves_classification.png`
- Logs: Console output with loss/accuracy metrics
'''
        
        readme_path = self.project_root / "README_TimesNet_WESAD.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
            
        print(f"✓ README created: {readme_path}")
    
    def setup_complete(self):
        """Run complete setup."""
        print("=== TimesNet VS Code Setup ===")
        
        # Clone repository
        if not self.clone_timesnet_repo():
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Create custom components
        self.create_data_loader()
        self.create_training_script()
        self.create_launch_config()
        self.create_readme()
        
        print("\\n=== Setup Complete ===")
        print("Next steps:")
        print("1. Format your WESAD data: python wesad_timesnet_formatter.py")
        print("2. Train the model: python train_timesnet_wesad.py")
        print("3. Or use VS Code debugger with the provided launch configurations")
        
        return True

def main():
    """Main setup function."""
    setup = TimesNetVSCodeSetup()
    setup.setup_complete()

if __name__ == "__main__":
    main() 