#!/usr/bin/env python3
"""
TimesNet Training Script for WESAD Dataset (Final Version)
Uses Time-Series-Library to train TimesNet on WESAD physiological data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# Add Time-Series-Library to path
sys.path.append('./Time-Series-Library')

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
        self.data = np.load(self.data_path / split / f"S2_{split}_data.npy")
        self.labels = np.load(self.data_path / split / f"S2_{split}_labels.npy")
        
        # Load metadata
        with open(self.data_path / "S2_metadata.json", 'r') as f:
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
        """Create TimesNet model using Time-Series-Library."""
        # Load metadata for model configuration
        with open(Path(self.config['data_path']) / "S2_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Import TimesNet from Time-Series-Library
        from models import TimesNet
        
        # Create config object
        class Config:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        model_config = Config(
            task_name=self.config['task'],
            seq_len=metadata['sequence_length'],
            label_len=metadata['sequence_length'] // 2,
            pred_len=1,  # For classification
            enc_in=metadata['num_features'],
            dec_in=metadata['num_features'],
            c_out=metadata['num_classes'] if self.config['task'] == 'classification' else metadata['num_features'],
            d_model=self.config['d_model'],
            e_layers=self.config['e_layers'],
            d_ff=self.config['d_ff'],
            factor=self.config['factor'],
            factor_temporal=self.config['factor_temporal'],
            top_k=self.config['top_k'],
            num_kernels=self.config['num_kernels'],
            dropout=self.config['dropout'],
            embed='timeF',
            freq='h',
            activation='gelu',
            num_class=metadata['num_classes']
        )
        
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
            
            # Create dummy mark enc for classification
            if self.config['task'] == 'classification':
                # For classification, create a mask of ones
                x_mark_enc = torch.ones(data.shape[0], data.shape[1]).to(self.device)
                outputs = self.model(data, x_mark_enc, None, None)
            else:
                # For forecasting, use the data as both encoder and decoder
                outputs = self.model(data, None, data, None)
            
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
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
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
    
    def test(self):
        """Test the model on test set."""
        print("Testing model...")
        
        # Load best model
        self.model.load_state_dict(torch.load(f"best_model_{self.config['task']}.pth"))
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in tqdm(self.test_loader, desc="Testing"):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                if self.config['task'] == 'classification':
                    labels = labels.squeeze()
                
                # Create dummy mark enc for classification
                if self.config['task'] == 'classification':
                    x_mark_enc = torch.ones(data.shape[0], data.shape[1]).to(self.device)
                    outputs = self.model(data, x_mark_enc, None, None)
                else:
                    outputs = self.model(data, None, data, None)
                
                if self.config['task'] == 'classification':
                    loss = self.criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                else:
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_loader)
        
        if self.config['task'] == 'classification':
            accuracy = 100 * correct / total
            print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            return avg_loss, accuracy
        else:
            print(f"Test Loss: {avg_loss:.4f}")
            return avg_loss

def main():
    """Main training function."""
    # Configuration
    config = {
        'data_path': './timesnet_data',
        'task': 'classification',  # or 'forecasting'
        'batch_size': 16,  # Reduced for memory
        'epochs': 20,  # Reduced for quick training
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
    
    print("=== TimesNet WESAD Training ===")
    print(f"Task: {config['task']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    # Create trainer and train
    trainer = TimesNetTrainer(config)
    trainer.train()
    
    # Test the model
    trainer.test()
    
    print("\n=== Training Complete ===")
    print("Files generated:")
    print(f"- best_model_{config['task']}.pth")
    print(f"- training_curves_{config['task']}.png")

if __name__ == "__main__":
    main() 