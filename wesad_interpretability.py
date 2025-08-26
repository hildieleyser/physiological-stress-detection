#!/usr/bin/env python3
"""
WESAD Model Interpretability Analysis
Creates mechanistic interpretability visualizations using simple methods.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

# Set style
plt.style.use('default')
sns.set_palette("husl")

class WESADInterpretability:
    def __init__(self, data_path="./timesnet_data", model_path="best_model_simple.pth"):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        
        # Load data and model
        self.load_data()
        self.load_model()
        
        # Create output directory
        self.output_dir = Path("interpretability_plots")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load WESAD data."""
        print("Loading WESAD data...")
        
        self.test_data = np.load(self.data_path / "test" / "S2_test_data.npy")
        self.test_labels = np.load(self.data_path / "test" / "S2_test_labels.npy")
        
        with open(self.data_path / "S2_metadata.json", 'r') as f:
            self.metadata = json.load(f)
            
        print(f"Test data loaded: {self.test_data.shape}")
        
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
        
    def plot_saliency_maps(self):
        """Plot saliency maps for model interpretability."""
        print("Creating saliency maps...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Saliency Maps - Feature Importance', fontsize=16, fontweight='bold')
        
        signal_names = ['ACC_X', 'ACC_Y', 'ACC_Z', 'EDA', 'BVP', 'TEMP']
        label_names = ['Baseline', 'Stress', 'Amusement', 'Meditation']
        
        # Analyze first 3 test samples
        for sample_idx in range(3):
            if sample_idx >= len(self.test_data):
                break
                
            # Get sample data
            sample_data = torch.FloatTensor(self.test_data[sample_idx:sample_idx+1])
            sample_data.requires_grad_(True)
            
            # Get prediction
            self.model.eval()
            prediction = self.model(sample_data)
            predicted_class = torch.argmax(prediction, dim=1).item()
            true_class = self.test_labels[sample_idx]
            
            # Calculate gradients
            prediction[0, predicted_class].backward()
            gradients = sample_data.grad.abs().squeeze().numpy()  # Shape: (1000, 6)
            
            # Plot saliency map
            ax = axes[sample_idx // 3, sample_idx % 3]
            im = ax.imshow(gradients.T, cmap='hot', aspect='auto')
            ax.set_title(f'Sample {sample_idx+1}: True={label_names[true_class]}, Pred={label_names[predicted_class]}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Signals')
            ax.set_yticks(range(6))
            ax.set_yticklabels(signal_names)
            plt.colorbar(im, ax=ax)
        
        # Hide unused subplots
        for i in range(3, 6):
            ax = axes[i // 3, i % 3]
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'saliency_maps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saliency maps saved to {self.output_dir / 'saliency_maps.png'}")
        
    def plot_feature_ablation(self):
        """Plot feature ablation analysis."""
        print("Creating feature ablation analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Ablation Analysis', fontsize=16, fontweight='bold')
        
        signal_names = ['ACC_X', 'ACC_Y', 'ACC_Z', 'EDA', 'BVP', 'TEMP']
        label_names = ['Baseline', 'Stress', 'Amusement', 'Meditation']
        
        # 1. Feature importance by ablation
        ax1 = axes[0, 0]
        
        # Calculate importance by removing each feature
        baseline_accuracy = self.calculate_accuracy(self.test_data, self.test_labels)
        feature_importance = []
        
        for i in range(6):
            # Create data with feature i zeroed out
            ablated_data = self.test_data.copy()
            ablated_data[:, :, i] = 0
            ablated_accuracy = self.calculate_accuracy(ablated_data, self.test_labels)
            importance = baseline_accuracy - ablated_accuracy
            feature_importance.append(importance)
        
        bars = ax1.bar(signal_names, feature_importance, alpha=0.7, color='lightcoral')
        ax1.set_title('Feature Importance (Ablation)')
        ax1.set_ylabel('Accuracy Drop')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, importance in zip(bars, feature_importance):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{importance:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Time step importance
        ax2 = axes[0, 1]
        
        # Calculate importance of different time segments
        segment_size = 100
        n_segments = 1000 // segment_size
        segment_importance = []
        
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size
            
            # Create data with segment zeroed out
            ablated_data = self.test_data.copy()
            ablated_data[:, start_idx:end_idx, :] = 0
            ablated_accuracy = self.calculate_accuracy(ablated_data, self.test_labels)
            importance = baseline_accuracy - ablated_accuracy
            segment_importance.append(importance)
        
        time_segments = np.arange(n_segments) * segment_size
        ax2.plot(time_segments, segment_importance, linewidth=2, color='darkblue')
        ax2.set_title('Time Segment Importance (Ablation)')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Accuracy Drop')
        ax2.grid(True, alpha=0.3)
        
        # Highlight important regions
        threshold = np.percentile(segment_importance, 80)
        important_regions = np.array(segment_importance) > threshold
        ax2.fill_between(time_segments, 0, segment_importance, 
                        where=important_regions, alpha=0.3, color='red', label='High Importance')
        ax2.legend()
        
        # 3. Class-wise feature importance
        ax3 = axes[1, 0]
        
        class_feature_importance = np.zeros((4, 6))
        for class_id in range(4):
            class_mask = self.test_labels == class_id
            if np.sum(class_mask) > 0:
                class_data = self.test_data[class_mask]
                class_baseline = self.calculate_accuracy(class_data, self.test_labels[class_mask])
                
                for feature_id in range(6):
                    ablated_data = class_data.copy()
                    ablated_data[:, :, feature_id] = 0
                    ablated_accuracy = self.calculate_accuracy(ablated_data, self.test_labels[class_mask])
                    class_feature_importance[class_id, feature_id] = class_baseline - ablated_accuracy
        
        im = ax3.imshow(class_feature_importance, cmap='YlOrRd')
        ax3.set_title('Class-wise Feature Importance')
        ax3.set_xlabel('Features')
        ax3.set_ylabel('Classes')
        ax3.set_xticks(range(6))
        ax3.set_yticks(range(4))
        ax3.set_xticklabels(signal_names, rotation=45)
        ax3.set_yticklabels(label_names)
        plt.colorbar(im, ax=ax3)
        
        # 4. Model confidence analysis
        ax4 = axes[1, 1]
        
        # Get model predictions and confidence
        self.model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(self.test_data)
            predictions = self.model(test_tensor)
            probabilities = torch.softmax(predictions, dim=1).numpy()
            max_confidences = np.max(probabilities, axis=1)
            predicted_classes = np.argmax(probabilities, axis=1)
        
        # Plot confidence vs accuracy
        correct_predictions = predicted_classes == self.test_labels
        confidence_bins = np.linspace(0, 1, 11)
        accuracy_by_confidence = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (max_confidences >= confidence_bins[i]) & (max_confidences < confidence_bins[i + 1])
            if np.sum(mask) > 0:
                accuracy = np.mean(correct_predictions[mask])
                accuracy_by_confidence.append(accuracy)
            else:
                accuracy_by_confidence.append(0)
        
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        ax4.plot(bin_centers, accuracy_by_confidence, 'o-', linewidth=2, markersize=8)
        ax4.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, label='Perfect Calibration')
        ax4.set_title('Model Calibration')
        ax4.set_xlabel('Predicted Confidence')
        ax4.set_ylabel('Actual Accuracy')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_ablation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Feature ablation saved to {self.output_dir / 'feature_ablation.png'}")
        
    def plot_representation_analysis(self):
        """Plot representation analysis."""
        print("Creating representation analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Representation Analysis', fontsize=16, fontweight='bold')
        
        label_names = ['Baseline', 'Stress', 'Amusement', 'Meditation']
        
        # 1. PCA visualization of representations
        ax1 = axes[0, 0]
        
        # Get model representations
        self.model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(self.test_data)
            
            # Get representations from different layers
            x = self.model.input_projection(test_tensor)
            input_repr = x.mean(dim=1).numpy()  # Average across time
            
            x = self.model.transformer.layers[0](x)
            layer1_repr = x.mean(dim=1).numpy()
            
            x = self.model.transformer.layers[1](x)
            layer2_repr = x.mean(dim=1).numpy()
            
            # Final representation
            x = x.reshape(x.shape[0], -1)
            final_repr = self.model.output_projection(x).numpy()
        
        # Apply PCA to final representations
        pca = PCA(n_components=2)
        repr_2d = pca.fit_transform(final_repr)
        
        # Plot PCA
        scatter = ax1.scatter(repr_2d[:, 0], repr_2d[:, 1], 
                            c=self.test_labels, cmap='tab10', alpha=0.7, s=100)
        ax1.set_title('PCA of Final Representations')
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        
        # Add legend
        legend1 = ax1.legend(*scatter.legend_elements(), title="Classes")
        ax1.add_artist(legend1)
        
        # 2. t-SNE visualization
        ax2 = axes[0, 1]
        
        # Apply t-SNE to final representations
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(final_repr)-1))
        repr_tsne = tsne.fit_transform(final_repr)
        
        scatter = ax2.scatter(repr_tsne[:, 0], repr_tsne[:, 1], 
                            c=self.test_labels, cmap='tab10', alpha=0.7, s=100)
        ax2.set_title('t-SNE of Final Representations')
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        
        # Add legend
        legend1 = ax2.legend(*scatter.legend_elements(), title="Classes")
        ax2.add_artist(legend1)
        
        # 3. Layer-wise representation similarity
        ax3 = axes[1, 0]
        
        representations = [input_repr, layer1_repr, layer2_repr, final_repr]
        layer_names = ['Input', 'Layer 1', 'Layer 2', 'Output']
        
        # Calculate cosine similarity matrix
        similarity_matrix = np.zeros((len(representations), len(representations)))
        for i in range(len(representations)):
            for j in range(len(representations)):
                # Calculate cosine similarity
                vec1 = representations[i].flatten()
                vec2 = representations[j].flatten()
                sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                similarity_matrix[i, j] = sim
        
        im = ax3.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_title('Layer-wise Representation Similarity')
        ax3.set_xticks(range(len(layer_names)))
        ax3.set_yticks(range(len(layer_names)))
        ax3.set_xticklabels(layer_names)
        ax3.set_yticklabels(layer_names)
        plt.colorbar(im, ax=ax3)
        
        # 4. Class separation analysis
        ax4 = axes[1, 1]
        
        # Calculate class separation metrics
        class_means = []
        class_variances = []
        
        for class_id in range(4):
            class_mask = self.test_labels == class_id
            if np.sum(class_mask) > 0:
                class_repr = final_repr[class_mask]
                class_means.append(np.mean(class_repr, axis=0))
                class_variances.append(np.var(class_repr, axis=0))
            else:
                class_means.append(np.zeros(final_repr.shape[1]))
                class_variances.append(np.zeros(final_repr.shape[1]))
        
        class_means = np.array(class_means)
        class_variances = np.array(class_variances)
        
        # Plot class separation
        x_pos = np.arange(4)
        separation_scores = []
        
        for i in range(4):
            # Calculate separation score (simplified)
            other_means = np.delete(class_means, i, axis=0)
            separation = np.mean(np.linalg.norm(class_means[i] - other_means, axis=1))
            separation_scores.append(separation)
        
        bars = ax4.bar(label_names, separation_scores, alpha=0.7, color='lightblue')
        ax4.set_title('Class Separation Scores')
        ax4.set_ylabel('Separation Score')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars, separation_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'representation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Representation analysis saved to {self.output_dir / 'representation_analysis.png'}")
        
    def calculate_accuracy(self, data, labels):
        """Calculate model accuracy on given data."""
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data)
            predictions = self.model(data_tensor)
            pred_labels = torch.argmax(predictions, dim=1).numpy()
            accuracy = np.mean(pred_labels == labels)
        return accuracy
        
    def create_all_interpretability_plots(self):
        """Create all interpretability plots."""
        print("Creating comprehensive interpretability analysis...")
        
        self.plot_saliency_maps()
        self.plot_feature_ablation()
        self.plot_representation_analysis()
        
        # Create interpretability summary
        self.create_interpretability_summary()
        
        print(f"\n🎉 All interpretability plots saved to {self.output_dir}/")
        print("Generated files:")
        print("- saliency_maps.png")
        print("- feature_ablation.png")
        print("- representation_analysis.png")
        print("- interpretability_summary.txt")
        
    def create_interpretability_summary(self):
        """Create interpretability summary report."""
        print("Creating interpretability summary...")
        
        # Calculate interpretability metrics
        baseline_accuracy = self.calculate_accuracy(self.test_data, self.test_labels)
        
        # Feature importance by ablation
        signal_names = ['ACC_X', 'ACC_Y', 'ACC_Z', 'EDA', 'BVP', 'TEMP']
        feature_importance = []
        
        for i in range(6):
            ablated_data = self.test_data.copy()
            ablated_data[:, :, i] = 0
            ablated_accuracy = self.calculate_accuracy(ablated_data, self.test_labels)
            importance = baseline_accuracy - ablated_accuracy
            feature_importance.append(importance)
        
        most_important_feature = signal_names[np.argmax(feature_importance)]
        least_important_feature = signal_names[np.argmin(feature_importance)]
        
        report = f"""
WESAD Model Interpretability Analysis
====================================

Model Interpretability Summary:
-----------------------------
Baseline Accuracy: {baseline_accuracy:.3f} ({baseline_accuracy*100:.1f}%)

Feature Importance (Ablation Analysis):
-------------------------------------
Most Important Feature: {most_important_feature} (Impact: {np.max(feature_importance):.4f})
Least Important Feature: {least_important_feature} (Impact: {np.min(feature_importance):.4f})

Feature Importance Ranking:
1. {signal_names[np.argsort(feature_importance)[-1]]}: {np.sort(feature_importance)[-1]:.4f}
2. {signal_names[np.argsort(feature_importance)[-2]]}: {np.sort(feature_importance)[-2]:.4f}
3. {signal_names[np.argsort(feature_importance)[-3]]}: {np.sort(feature_importance)[-3]:.4f}
4. {signal_names[np.argsort(feature_importance)[-4]]}: {np.sort(feature_importance)[-4]:.4f}
5. {signal_names[np.argsort(feature_importance)[-5]]}: {np.sort(feature_importance)[-5]:.4f}
6. {signal_names[np.argsort(feature_importance)[-6]]}: {np.sort(feature_importance)[-6]:.4f}

Interpretability Methods Used:
----------------------------
1. Saliency Maps: Gradient-based feature importance
2. Feature Ablation: Remove features and measure impact
3. Representation Analysis: PCA and t-SNE visualization
4. Layer-wise Analysis: Compare representations across layers

Key Insights:
-------------
1. The model relies most heavily on {most_important_feature} for predictions
2. Feature ablation shows which signals are critical for performance
3. Saliency maps reveal temporal patterns in feature importance
4. Representation analysis shows how the model transforms input data

Recommendations for Interpretability:
-----------------------------------
1. Focus on {most_important_feature} for feature engineering
2. Consider removing {least_important_feature} to reduce complexity
3. Use saliency maps to identify important time periods
4. Monitor representation quality across layers

Files Generated:
---------------
- saliency_maps.png: Gradient-based feature importance visualization
- feature_ablation.png: Feature and time step importance analysis
- representation_analysis.png: PCA, t-SNE, and layer-wise analysis
- interpretability_summary.txt: This report

"""
        
        with open(self.output_dir / 'interpretability_summary.txt', 'w') as f:
            f.write(report)
        
        print(f"✓ Interpretability summary saved to {self.output_dir / 'interpretability_summary.txt'}")

def main():
    """Main function to create interpretability plots."""
    print("=== WESAD Model Interpretability Analysis ===")
    
    interpreter = WESADInterpretability()
    interpreter.create_all_interpretability_plots()
    
    print("\n🎉 Interpretability analysis complete!")
    print("Check the 'interpretability_plots' folder for all interpretability visualizations.")

if __name__ == "__main__":
    main() 