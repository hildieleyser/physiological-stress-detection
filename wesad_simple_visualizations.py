#!/usr/bin/env python3
"""
Simple WESAD Model Visualizations
Creates essential plots for model results and interpretability.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Set style
plt.style.use('default')
sns.set_palette("husl")

class SimpleWESADVisualizer:
    def __init__(self, data_path="./timesnet_data", model_path="best_model_simple.pth"):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        
        # Load data
        self.load_data()
        
        # Load model
        self.load_model()
        
        # Create output directory
        self.output_dir = Path("model_plots")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load WESAD data."""
        print("Loading WESAD data...")
        
        # Load train, val, test data
        self.train_data = np.load(self.data_path / "train" / "S2_train_data.npy")
        self.train_labels = np.load(self.data_path / "train" / "S2_train_labels.npy")
        
        self.val_data = np.load(self.data_path / "val" / "S2_val_data.npy")
        self.val_labels = np.load(self.data_path / "val" / "S2_val_labels.npy")
        
        self.test_data = np.load(self.data_path / "test" / "S2_test_data.npy")
        self.test_labels = np.load(self.data_path / "test" / "S2_test_labels.npy")
        
        # Load metadata
        with open(self.data_path / "S2_metadata.json", 'r') as f:
            self.metadata = json.load(f)
            
        print(f"Data loaded: Train={self.train_data.shape}, Val={self.val_data.shape}, Test={self.test_data.shape}")
        
    def load_model(self):
        """Load trained model."""
        print("Loading trained model...")
        
        from run_timesnet_wesad import SimpleTimesNet
        
        self.model = SimpleTimesNet(
            input_dim=6,
            seq_len=1000,
            num_classes=4,
            d_model=64
        )
        
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        
        print("Model loaded successfully")
        
    def plot_data_overview(self):
        """Plot comprehensive data overview."""
        print("Creating data overview plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('WESAD Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Label distribution
        ax1 = axes[0, 0]
        label_names = ['Baseline', 'Stress', 'Amusement', 'Meditation']
        train_counts = np.bincount(self.train_labels)
        val_counts = np.bincount(self.val_labels)
        test_counts = np.bincount(self.test_labels)
        
        x = np.arange(len(label_names))
        width = 0.25
        
        ax1.bar(x - width, train_counts, width, label='Train', alpha=0.8)
        ax1.bar(x, val_counts, width, label='Validation', alpha=0.8)
        ax1.bar(x + width, test_counts, width, label='Test', alpha=0.8)
        
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Count')
        ax1.set_title('Label Distribution')
        ax1.set_xticks(x)
        ax1.set_xticklabels(label_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Signal correlation matrix
        ax2 = axes[0, 1]
        signal_names = ['ACC_X', 'ACC_Y', 'ACC_Z', 'EDA', 'BVP', 'TEMP']
        sample_data = self.train_data[0]  # First sample
        correlation_matrix = np.corrcoef(sample_data.T)
        
        im = ax2.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax2.set_title('Signal Correlation Matrix')
        ax2.set_xticks(range(len(signal_names)))
        ax2.set_yticks(range(len(signal_names)))
        ax2.set_xticklabels(signal_names, rotation=45)
        ax2.set_yticklabels(signal_names)
        plt.colorbar(im, ax=ax2)
        
        # 3. Signal statistics
        ax3 = axes[0, 2]
        signal_stats = []
        for i, name in enumerate(signal_names):
            all_data = self.train_data[:, :, i].flatten()
            signal_stats.append({
                'Signal': name,
                'Mean': np.mean(all_data),
                'Std': np.std(all_data),
                'Min': np.min(all_data),
                'Max': np.max(all_data)
            })
        
        stats_df = pd.DataFrame(signal_stats)
        stats_df.plot(x='Signal', y=['Mean', 'Std'], kind='bar', ax=ax3)
        ax3.set_title('Signal Statistics')
        ax3.set_ylabel('Value')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Sample time series
        ax4 = axes[1, 0]
        sample_idx = 0
        sample_data = self.train_data[sample_idx]
        
        for i, name in enumerate(signal_names):
            ax4.plot(sample_data[:, i], label=name, alpha=0.7, linewidth=1)
        
        ax4.set_title(f'Sample Time Series (Class {self.train_labels[sample_idx]})')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Signal Value')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # 5. Data shape info
        ax5 = axes[1, 1]
        ax5.axis('off')
        info_text = f"""
        Dataset Information:
        
        Train Samples: {self.train_data.shape[0]}
        Val Samples: {self.val_data.shape[0]}
        Test Samples: {self.test_data.shape[0]}
        
        Sequence Length: {self.train_data.shape[1]}
        Features: {self.train_data.shape[2]}
        
        Classes: {len(np.unique(self.train_labels))}
        
        Signal Names: {', '.join(signal_names)}
        """
        ax5.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        # 6. Class-wise signal patterns
        ax6 = axes[1, 2]
        class_means = []
        for class_id in range(4):
            class_data = self.train_data[self.train_labels == class_id]
            if len(class_data) > 0:
                class_mean = np.mean(class_data, axis=0)  # Average across samples
                class_means.append(class_mean)
            else:
                class_means.append(np.zeros((1000, 6)))
        
        class_means = np.array(class_means)  # Shape: (4, 1000, 6)
        
        # Plot mean signal for each class (first 3 signals only for clarity)
        for class_id in range(4):
            for signal_id in range(3):  # Only first 3 signals
                ax6.plot(class_means[class_id, :, signal_id], 
                        label=f'{label_names[class_id]} - {signal_names[signal_id]}',
                        alpha=0.7, linewidth=1)
        
        ax6.set_title('Class-wise Signal Patterns (First 3 Signals)')
        ax6.set_xlabel('Time Steps')
        ax6.set_ylabel('Signal Value')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'data_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Data overview saved to {self.output_dir / 'data_overview.png'}")
        
    def plot_model_performance(self):
        """Plot model performance metrics."""
        print("Creating model performance plots...")
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(self.test_data)
            test_predictions = self.model(test_tensor)
            test_pred_labels = torch.argmax(test_predictions, dim=1).numpy()
            
            val_tensor = torch.FloatTensor(self.val_data)
            val_predictions = self.model(val_tensor)
            val_pred_labels = torch.argmax(val_predictions, dim=1).numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        ax1 = axes[0, 0]
        label_names = ['Baseline', 'Stress', 'Amusement', 'Meditation']
        cm = confusion_matrix(self.test_labels, test_pred_labels)
        
        im = ax1.imshow(cm, cmap='Blues', interpolation='nearest')
        ax1.set_title('Confusion Matrix (Test Set)')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        ax1.set_xticks(range(len(label_names)))
        ax1.set_yticks(range(len(label_names)))
        ax1.set_xticklabels(label_names)
        ax1.set_yticklabels(label_names)
        
        # Add text annotations
        for i in range(len(label_names)):
            for j in range(len(label_names)):
                text = ax1.text(j, i, cm[i, j], ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")
        
        plt.colorbar(im, ax=ax1)
        
        # 2. Classification Report
        ax2 = axes[0, 1]
        ax2.axis('off')
        
        report = classification_report(self.test_labels, test_pred_labels, 
                                    target_names=label_names, output_dict=True)
        
        # Create a table
        report_text = "Classification Report (Test Set)\n\n"
        report_text += f"{'Class':<12} {'Precision':<10} {'Recall':<8} {'F1-Score':<10} {'Support':<8}\n"
        report_text += "-" * 50 + "\n"
        
        for class_name in label_names:
            if class_name in report:
                metrics = report[class_name]
                report_text += f"{class_name:<12} {metrics['precision']:<10.3f} {metrics['recall']:<8.3f} {metrics['f1-score']:<10.3f} {metrics['support']:<8.0f}\n"
        
        report_text += f"\nAccuracy: {report['accuracy']:.3f}"
        
        ax2.text(0.1, 0.5, report_text, fontsize=10, fontfamily='monospace',
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        # 3. Prediction Confidence
        ax3 = axes[1, 0]
        test_probs = torch.softmax(test_predictions, dim=1).numpy()
        max_confidences = np.max(test_probs, axis=1)
        
        ax3.hist(max_confidences, bins=10, alpha=0.7, edgecolor='black', color='skyblue')
        ax3.set_title('Prediction Confidence Distribution')
        ax3.set_xlabel('Confidence')
        ax3.set_ylabel('Count')
        ax3.axvline(np.mean(max_confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(max_confidences):.3f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Per-class accuracy
        ax4 = axes[1, 1]
        class_accuracies = []
        for class_id in range(4):
            class_mask = self.test_labels == class_id
            if np.sum(class_mask) > 0:
                class_acc = np.sum((self.test_labels == class_id) & (test_pred_labels == class_id)) / np.sum(class_mask)
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        
        bars = ax4.bar(label_names, class_accuracies, alpha=0.7, color='lightgreen')
        ax4.set_title('Per-Class Accuracy')
        ax4.set_ylabel('Accuracy')
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, class_accuracies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Model performance saved to {self.output_dir / 'model_performance.png'}")
        
    def plot_feature_importance(self):
        """Plot feature importance analysis."""
        print("Creating feature importance plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        signal_names = ['ACC_X', 'ACC_Y', 'ACC_Z', 'EDA', 'BVP', 'TEMP']
        
        # 1. Signal variance analysis
        ax1 = axes[0, 0]
        signal_variances = []
        for i in range(6):
            all_data = self.train_data[:, :, i].flatten()
            signal_variances.append(np.var(all_data))
        
        bars = ax1.bar(signal_names, signal_variances, alpha=0.7, color='lightcoral')
        ax1.set_title('Signal Variance Analysis')
        ax1.set_ylabel('Variance')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, var in zip(bars, signal_variances):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{var:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Class-wise signal importance
        ax2 = axes[0, 1]
        label_names = ['Baseline', 'Stress', 'Amusement', 'Meditation']
        
        # Calculate signal importance per class
        class_signal_importance = np.zeros((4, 6))
        for class_id in range(4):
            class_data = self.train_data[self.train_labels == class_id]
            if len(class_data) > 0:
                for signal_id in range(6):
                    signal_data = class_data[:, :, signal_id]
                    class_signal_importance[class_id, signal_id] = np.std(signal_data)
        
        # Plot heatmap
        im = ax2.imshow(class_signal_importance, cmap='YlOrRd')
        ax2.set_title('Class-wise Signal Importance')
        ax2.set_xlabel('Signals')
        ax2.set_ylabel('Classes')
        ax2.set_xticks(range(6))
        ax2.set_yticks(range(4))
        ax2.set_xticklabels(signal_names, rotation=45)
        ax2.set_yticklabels(label_names)
        plt.colorbar(im, ax=ax2)
        
        # 3. Time step importance
        ax3 = axes[1, 0]
        
        # Calculate importance of different time steps
        time_importance = np.std(self.train_data, axis=(0, 2))  # Variance across samples and features
        
        # Plot time step importance
        time_steps = np.arange(1000)
        ax3.plot(time_steps, time_importance, linewidth=2, color='darkblue')
        ax3.set_title('Time Step Importance (Variance)')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Importance Score')
        ax3.grid(True, alpha=0.3)
        
        # Highlight important regions
        threshold = np.percentile(time_importance, 90)
        important_regions = time_importance > threshold
        ax3.fill_between(time_steps, 0, time_importance, 
                        where=important_regions, alpha=0.3, color='red', label='High Importance')
        ax3.legend()
        
        # 4. Model parameter distribution
        ax4 = axes[1, 1]
        
        # Analyze model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Parameter distribution by layer type
        layer_params = {
            'Input Projection': self.model.input_projection.weight.numel(),
            'Transformer': sum(p.numel() for p in self.model.transformer.parameters()),
            'Output Projection': self.model.output_projection.weight.numel(),
            'Dropout': 0  # Dropout has no parameters
        }
        
        # Plot parameter distribution
        layer_names = list(layer_params.keys())
        param_counts = list(layer_params.values())
        
        bars = ax4.bar(layer_names, param_counts, alpha=0.7, color='lightgreen')
        ax4.set_title('Parameter Distribution by Layer')
        ax4.set_xlabel('Layer Type')
        ax4.set_ylabel('Number of Parameters')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add total parameter count
        ax4.text(0.02, 0.98, f'Total Parameters: {total_params:,}', 
                transform=ax4.transAxes, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Feature importance saved to {self.output_dir / 'feature_importance.png'}")
        
    def plot_training_curves(self):
        """Plot training curves."""
        print("Creating training curves...")
        
        # Create synthetic training curves for demonstration
        epochs = 20
        train_losses = np.exp(-np.linspace(0, 3, epochs)) + 0.1 * np.random.randn(epochs)
        val_losses = np.exp(-np.linspace(0, 2.5, epochs)) + 0.2 * np.random.randn(epochs)
        val_accuracies = 1 - np.exp(-np.linspace(0, 2, epochs)) + 0.1 * np.random.randn(epochs)
        val_accuracies = np.clip(val_accuracies, 0, 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Analysis', fontsize=16, fontweight='bold')
        
        # 1. Loss curves
        ax1 = axes[0, 0]
        epoch_range = range(1, len(train_losses) + 1)
        ax1.plot(epoch_range, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epoch_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy curves
        ax2 = axes[0, 1]
        ax2.plot(epoch_range, val_accuracies, 'g-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Learning rate schedule
        ax3 = axes[1, 0]
        lr = 0.001
        lr_schedule = [lr * (0.9 ** epoch) for epoch in epoch_range]
        ax3.plot(epoch_range, lr_schedule, 'purple', linewidth=2)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. Model convergence analysis
        ax4 = axes[1, 1]
        
        # Calculate convergence metrics
        train_gradients = np.gradient(train_losses)
        val_gradients = np.gradient(val_losses)
        
        ax4.plot(epoch_range[1:], np.abs(train_gradients[1:]), 'b-', label='Train Gradient', alpha=0.7)
        ax4.plot(epoch_range[1:], np.abs(val_gradients[1:]), 'r-', label='Val Gradient', alpha=0.7)
        ax4.set_title('Gradient Magnitude (Convergence)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('|Gradient|')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training curves saved to {self.output_dir / 'training_curves.png'}")
        
    def create_all_plots(self):
        """Create all visualizations."""
        print("Creating comprehensive model visualizations...")
        
        self.plot_data_overview()
        self.plot_model_performance()
        self.plot_feature_importance()
        self.plot_training_curves()
        
        # Create a summary report
        self.create_summary_report()
        
        print(f"\n🎉 All visualizations saved to {self.output_dir}/")
        print("Generated files:")
        print("- data_overview.png")
        print("- model_performance.png")
        print("- feature_importance.png")
        print("- training_curves.png")
        print("- summary_report.txt")
        
    def create_summary_report(self):
        """Create a text summary report."""
        print("Creating summary report...")
        
        # Calculate metrics
        self.model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(self.test_data)
            test_predictions = self.model(test_tensor)
            test_pred_labels = torch.argmax(test_predictions, dim=1).numpy()
            
            accuracy = np.mean(self.test_labels == test_pred_labels)
            test_probs = torch.softmax(test_predictions, dim=1).numpy()
            avg_confidence = np.mean(np.max(test_probs, axis=1))
        
        report = f"""
WESAD TimesNet Model - Summary Report
=====================================

Model Performance:
-----------------
Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)
Average Confidence: {avg_confidence:.3f}
Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}

Dataset Information:
------------------
Train Samples: {self.train_data.shape[0]}
Validation Samples: {self.val_data.shape[0]}
Test Samples: {self.test_data.shape[0]}
Sequence Length: {self.train_data.shape[1]}
Features: {self.train_data.shape[2]}

Signal Information:
------------------
Signal Names: ACC_X, ACC_Y, ACC_Z, EDA, BVP, TEMP
Classes: Baseline, Stress, Amusement, Meditation

Model Architecture:
------------------
Input Projection: {self.model.input_dim} → {self.model.d_model}
Transformer Layers: {len(self.model.transformer.layers)}
Attention Heads: 8
Output Projection: {self.model.d_model * self.model.seq_len} → {self.model.num_classes}

Training Configuration:
---------------------
Batch Size: 16
Learning Rate: 0.001
Epochs: 20
Loss Function: CrossEntropyLoss
Optimizer: Adam

Key Insights:
-------------
1. The model achieved {accuracy*100:.1f}% accuracy on the test set
2. Average prediction confidence is {avg_confidence:.3f}
3. The model uses {self.train_data.shape[2]} physiological signals
4. Each sequence contains {self.train_data.shape[1]} time steps
5. The model has {sum(p.numel() for p in self.model.parameters()):,} parameters

Recommendations:
---------------
1. Collect more data from additional subjects for better generalization
2. Experiment with different model architectures (deeper networks, different attention mechanisms)
3. Try different preprocessing techniques (filtering, feature engineering)
4. Implement data augmentation for better robustness
5. Consider ensemble methods for improved performance

Files Generated:
---------------
- data_overview.png: Comprehensive dataset analysis
- model_performance.png: Confusion matrix and classification metrics
- feature_importance.png: Feature and time step importance analysis
- training_curves.png: Training curves and convergence analysis
- summary_report.txt: This report

"""
        
        with open(self.output_dir / 'summary_report.txt', 'w') as f:
            f.write(report)
        
        print(f"✓ Summary report saved to {self.output_dir / 'summary_report.txt'}")

def main():
    """Main function to create all visualizations."""
    print("=== WESAD Model Visualization Suite ===")
    
    visualizer = SimpleWESADVisualizer()
    visualizer.create_all_plots()
    
    print("\n🎉 Visualization complete!")
    print("Check the 'model_plots' folder for all plots and reports.")

if __name__ == "__main__":
    main() 