#!/usr/bin/env python3
"""
Simple TimesNet Training for WESAD Dataset
Uses Time-Series-Library's existing infrastructure.
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

class SimpleWESADDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = Path(data_path)
        self.split = split
        
        # Load data
        self.data = np.load(self.data_path / split / f"S2_{split}_data.npy")
        self.labels = np.load(self.data_path / split / f"S2_{split}_labels.npy")
        
        print(f"Loaded {split} data: {self.data.shape}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = torch.LongTensor([self.labels[idx]])
        return x, y

def create_simple_dataloader(data_path, batch_size=16):
    """Create simple data loaders."""
    train_dataset = SimpleWESADDataset(data_path, 'train')
    val_dataset = SimpleWESADDataset(data_path, 'val')
    test_dataset = SimpleWESADDataset(data_path, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

class SimpleTimesNet(nn.Module):
    """Simplified TimesNet for WESAD classification."""
    
    def __init__(self, input_dim=6, seq_len=1000, num_classes=4, d_model=64):
        super(SimpleTimesNet, self).__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output projection
        self.output_projection = nn.Linear(d_model * seq_len, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.shape[0]
        
        # Project to d_model
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Apply transformer
        x = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        # Flatten and project to classes
        x = x.reshape(batch_size, -1)  # (batch_size, seq_len * d_model)
        x = self.dropout(x)
        x = self.output_projection(x)  # (batch_size, num_classes)
        
        return x

class SimpleTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader = create_simple_dataloader(
            config['data_path'], 
            batch_size=config['batch_size']
        )
        
        # Initialize model
        self.model = SimpleTimesNet(
            input_dim=6,
            seq_len=1000,
            num_classes=4,
            d_model=config['d_model']
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, labels) in enumerate(tqdm(self.train_loader, desc="Training")):
            data = data.to(self.device)
            labels = labels.squeeze().to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
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
                labels = labels.squeeze().to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy
    
    def train(self):
        """Train the model."""
        print("Starting training...")
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model_simple.pth")
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
        plt.title('Training Curves - Simple TimesNet')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curves_simple.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Training curves saved")
    
    def test(self):
        """Test the model on test set."""
        print("Testing model...")
        
        # Load best model
        self.model.load_state_dict(torch.load("best_model_simple.pth"))
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in tqdm(self.test_loader, desc="Testing"):
                data = data.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy

def main():
    """Main training function."""
    # Configuration
    config = {
        'data_path': './timesnet_data',
        'batch_size': 16,
        'epochs': 20,
        'learning_rate': 0.001,
        'd_model': 64
    }
    
    print("=== Simple TimesNet WESAD Training ===")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    # Create trainer and train
    trainer = SimpleTrainer(config)
    trainer.train()
    
    # Test the model
    trainer.test()
    
    print("\n=== Training Complete ===")
    print("Files generated:")
    print("- best_model_simple.pth")
    print("- training_curves_simple.png")

if __name__ == "__main__":
    main() 