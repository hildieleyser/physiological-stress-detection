#!/usr/bin/env python3
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
