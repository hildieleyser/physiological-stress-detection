#!/usr/bin/env python3
"""
Labeled Graphs Slide Presentation for WESAD Dataset Analysis
Clean graph labels and titles, no bullet points or prose text.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Professional styling
plt.style.use('default')
sns.set_theme(style="whitegrid", font_scale=1.1)

# Professional color palette
COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange
    'accent': '#2ca02c',       # Green
    'stress': '#d62728',       # Red
    'neutral': '#7f7f7f',      # Grey
    'light': '#f7f7f7',        # Light grey
    'dark': '#2f2f2f'          # Dark grey
}

# Professional font settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 20,
    'figure.titleweight': 'bold',
    'axes.titleweight': 'bold',
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3
})

class LabeledGraphsSlide:
    def __init__(self, results_path: str = "results"):
        self.results_path = Path(results_path)
        self.output_path = Path("labeled_graphs_slide")
        self.output_path.mkdir(exist_ok=True)
        
        # Load analysis results
        self.load_data()
    
    def load_data(self):
        """Load analysis results."""
        with open(self.results_path / 'cv_summary.json', 'r') as f:
            self.cv_results = json.load(f)
        with open(self.results_path / 'shap_summary.json', 'r') as f:
            self.shap_results = json.load(f)
        with open(self.results_path / 'robustness_summary.json', 'r') as f:
            self.robustness_results = json.load(f)
        
        # Load actual data dimensions
        X = np.load(self.results_path / 'X.npy')
        y = np.load(self.results_path / 'y.npy')
        self.data_shape = X.shape
        self.class_counts = np.bincount(y)
    
    def create_labeled_slide(self):
        """Create slide with properly labeled graphs."""
        print("Creating labeled graphs slide...")
        
        # Create large figure for single slide
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.3)
        
        # Main title
        fig.suptitle('WESAD Dataset: Wearable Stress Detection Analysis', 
                    fontsize=24, fontweight='bold', color=COLORS['dark'], y=0.98)
        
        # 1. Dataset Overview (Top Left)
        ax1 = fig.add_subplot(gs[0, 0:2])
        self.plot_dataset_overview(ax1)
        
        # 2. Model Performance (Top Center)
        ax2 = fig.add_subplot(gs[0, 2:4])
        self.plot_model_performance(ax2)
        
        # 3. Feature Importance (Top Right)
        ax3 = fig.add_subplot(gs[0, 4:6])
        self.plot_feature_importance(ax3)
        
        # 4. Robustness Analysis (Bottom Left)
        ax4 = fig.add_subplot(gs[1, 0:3])
        self.plot_robustness_analysis(ax4)
        
        # 5. Signal Mapping (Bottom Right)
        ax5 = fig.add_subplot(gs[1, 3:6])
        self.plot_signal_mapping(ax5)
        
        # 6. Feature Engineering Flow (Bottom Full Width)
        ax6 = fig.add_subplot(gs[2:, :])
        self.plot_feature_flow(ax6)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'labeled_graphs_slide.png', dpi=300, 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Labeled graphs slide saved to: {self.output_path}")
    
    def plot_dataset_overview(self, ax):
        """Dataset overview with proper labels."""
        # Class distribution pie chart
        class_labels = ['Stress', 'Amusement', 'Resting']
        colors = [COLORS['stress'], COLORS['accent'], COLORS['primary']]
        wedges, texts, autotexts = ax.pie(self.class_counts, labels=class_labels, 
                                         colors=colors, autopct='%1.0f%%', 
                                         startangle=90, explode=(0.05, 0.05, 0.05))
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        ax.set_title('Dataset Composition', fontweight='bold', pad=20)
        
        # Add sample count annotation
        ax.text(0.5, -1.3, f'Total: {self.data_shape[0]:,} samples\n{self.data_shape[1]:,} features', 
               ha='center', va='center', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['light'], alpha=0.8))
    
    def plot_model_performance(self, ax):
        """Model performance with proper labels."""
        models = ['Logistic\nRegression', 'Random\nForest', 'XGBoost']
        accuracies = [self.cv_results['logistic_regression']['mean_accuracy'],
                     self.cv_results['random_forest']['mean_accuracy'],
                     self.cv_results['xgboost']['mean_accuracy']]
        stds = [self.cv_results['logistic_regression']['std_accuracy'],
                self.cv_results['random_forest']['std_accuracy'],
                self.cv_results['xgboost']['std_accuracy']]
        
        # Create bar plot
        bars = ax.bar(models, accuracies, yerr=stds, capsize=5, 
                     color=[COLORS['primary'], COLORS['secondary'], COLORS['accent']], 
                     alpha=0.8, edgecolor='white', linewidth=1.5)
        
        ax.set_title('Model Performance (Cross-Validation)', fontweight='bold', pad=20)
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, acc, std in zip(bars, accuracies, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{acc:.1%}\n±{std:.1%}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=10)
        
        # Highlight best model
        best_idx = np.argmax(accuracies)
        bars[best_idx].set_edgecolor(COLORS['dark'])
        bars[best_idx].set_linewidth(3)
    
    def plot_feature_importance(self, ax):
        """Feature importance with proper labels."""
        # Get top features across all models
        feature_importance_dict = {}
        for model in self.shap_results:
            for feat in self.shap_results[model]['top_features'][:3]:
                feature = feat['feature']
                importance = feat['importance']
                if feature in feature_importance_dict:
                    feature_importance_dict[feature] += importance
                else:
                    feature_importance_dict[feature] = importance
        
        # Get top 5 features
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        feature_names = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        
        # Create meaningful labels
        meaningful_labels = [f'Feature {f.split("_")[-1]}' for f in feature_names]
        
        # Create horizontal bar plot
        bars = ax.barh(meaningful_labels, importances, 
                      color=COLORS['secondary'], alpha=0.8, edgecolor='white', linewidth=1.5)
        
        ax.set_title('Top 5 Features (SHAP Importance)', fontweight='bold', pad=20)
        ax.set_xlabel('Importance Score', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add importance values
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            width = bar.get_width()
            ax.text(width + max(importances)*0.01, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    def plot_robustness_analysis(self, ax):
        """Robustness analysis with proper labels."""
        noise_levels = self.robustness_results['logistic_regression']['noise_levels']
        
        # Plot accuracy vs noise for all models
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
        model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
        
        for i, model in enumerate(self.robustness_results):
            accuracies = self.robustness_results[model]['accuracies']
            ax.plot(noise_levels, accuracies, 'o-', linewidth=2.5, 
                   label=model_names[i], color=colors[i], markersize=6, 
                   markerfacecolor='white', markeredgecolor=colors[i], markeredgewidth=1.5)
        
        ax.set_title('Model Robustness (Noise Sensitivity)', fontweight='bold', pad=20)
        ax.set_xlabel('Noise Level (%)', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add robustness ranking
        robustness_scores = []
        for model in self.robustness_results:
            accuracies = self.robustness_results[model]['accuracies']
            degradation = accuracies[0] - accuracies[-1]
            robustness_scores.append(1 - degradation)
        
        # Add ranking text
        ranking_text = "Robustness Ranking:\n"
        for i, (model, score) in enumerate(zip(model_names, robustness_scores)):
            ranking_text += f"{i+1}. {model}: {score:.3f}\n"
        
        ax.text(0.02, 0.98, ranking_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor=COLORS['light'], alpha=0.8))
    
    def plot_signal_mapping(self, ax):
        """Signal mapping diagram with proper labels."""
        ax.axis('off')
        
        # Create visual signal mapping with labels
        # Draw device outlines
        chest_rect = plt.Rectangle((0.1, 0.6), 0.35, 0.3, fill=False, 
                                 edgecolor=COLORS['primary'], linewidth=3)
        wrist_rect = plt.Rectangle((0.55, 0.6), 0.35, 0.3, fill=False, 
                                 edgecolor=COLORS['secondary'], linewidth=3)
        ax.add_patch(chest_rect)
        ax.add_patch(wrist_rect)
        
        # Add device labels
        ax.text(0.275, 0.95, 'Chest Device (RespiBAN)', ha='center', va='center', 
               fontweight='bold', fontsize=12, color=COLORS['primary'])
        ax.text(0.725, 0.95, 'Wrist Device (E4)', ha='center', va='center', 
               fontweight='bold', fontsize=12, color=COLORS['secondary'])
        
        # Draw signal channels as colored rectangles with labels
        chest_signals = ['ACC', 'ECG', 'EDA', 'EMG', 'RESP']
        chest_colors = [COLORS['primary'], COLORS['accent'], COLORS['stress'], 
                       COLORS['neutral'], COLORS['secondary']]
        
        wrist_signals = ['ACC', 'BVP', 'EDA']
        wrist_colors = [COLORS['secondary'], COLORS['accent'], COLORS['stress']]
        
        # Chest signals
        for i, (signal, color) in enumerate(zip(chest_signals, chest_colors)):
            x = 0.12 + (i % 3) * 0.1
            y = 0.75 - (i // 3) * 0.1
            signal_rect = plt.Rectangle((x, y), 0.08, 0.08, 
                                      facecolor=color, alpha=0.7)
            ax.add_patch(signal_rect)
            ax.text(x + 0.04, y + 0.04, signal, ha='center', va='center', 
                   fontweight='bold', fontsize=8, color='white')
        
        # Wrist signals
        for i, (signal, color) in enumerate(zip(wrist_signals, wrist_colors)):
            x = 0.57 + i * 0.1
            y = 0.75
            signal_rect = plt.Rectangle((x, y), 0.08, 0.08, 
                                      facecolor=color, alpha=0.7)
            ax.add_patch(signal_rect)
            ax.text(x + 0.04, y + 0.04, signal, ha='center', va='center', 
                   fontweight='bold', fontsize=8, color='white')
        
        # Draw connection lines
        for i in range(3):
            x1 = 0.45
            y1 = 0.75 - i * 0.1
            x2 = 0.55
            y2 = 0.75
            ax.plot([x1, x2], [y1, y2], color=COLORS['neutral'], 
                   linewidth=2, alpha=0.5)
        
        ax.set_title('WESAD Signal Mapping', fontweight='bold', pad=20)
    
    def plot_feature_flow(self, ax):
        """Feature engineering flow diagram with proper labels."""
        ax.axis('off')
        
        # Create visual flow diagram with labels
        # Raw signals
        raw_rect = plt.Rectangle((0.05, 0.7), 0.15, 0.2, 
                               facecolor=COLORS['primary'], alpha=0.3)
        ax.add_patch(raw_rect)
        ax.text(0.125, 0.8, 'Raw Signals\n(ACC, ECG, EDA,\nEMG, RESP, TEMP)', 
               ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Processing arrows
        ax.arrow(0.22, 0.8, 0.08, 0, head_width=0.02, head_length=0.01, 
                fc=COLORS['neutral'], ec=COLORS['neutral'])
        
        # Feature extraction
        feature_rect = plt.Rectangle((0.32, 0.7), 0.15, 0.2, 
                                   facecolor=COLORS['secondary'], alpha=0.3)
        ax.add_patch(feature_rect)
        ax.text(0.395, 0.8, 'Feature Extraction\n(Statistical Measures,\nTime Windows)', 
               ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Processing arrows
        ax.arrow(0.49, 0.8, 0.08, 0, head_width=0.02, head_length=0.01, 
                fc=COLORS['neutral'], ec=COLORS['neutral'])
        
        # Model training
        model_rect = plt.Rectangle((0.59, 0.7), 0.15, 0.2, 
                                 facecolor=COLORS['accent'], alpha=0.3)
        ax.add_patch(model_rect)
        ax.text(0.665, 0.8, 'Model Training\n(Cross-Validation,\nHyperparameter Tuning)', 
               ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Processing arrows
        ax.arrow(0.76, 0.8, 0.08, 0, head_width=0.02, head_length=0.01, 
                fc=COLORS['neutral'], ec=COLORS['neutral'])
        
        # Results
        result_rect = plt.Rectangle((0.86, 0.7), 0.1, 0.2, 
                                  facecolor=COLORS['stress'], alpha=0.3)
        ax.add_patch(result_rect)
        ax.text(0.91, 0.8, 'Stress Detection\nResults', 
               ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Draw feature types as small circles with labels
        feature_types = ['Time\nDomain', 'Frequency\nDomain', 'Cross-\nModal', 'Temporal\nPatterns', 'Quality\nMetrics']
        feature_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
                         COLORS['stress'], COLORS['neutral']]
        
        for i, (feature_type, color) in enumerate(zip(feature_types, feature_colors)):
            x = 0.32 + (i % 3) * 0.04
            y = 0.5 - (i // 3) * 0.15
            circle = plt.Circle((x, y), 0.015, facecolor=color, alpha=0.7)
            ax.add_patch(circle)
            ax.text(x, y - 0.03, feature_type, ha='center', va='center', 
                   fontweight='bold', fontsize=8)
        
        ax.set_title('Feature Engineering Pipeline', fontweight='bold', pad=20)

def main():
    """Create labeled graphs slide."""
    visualizer = LabeledGraphsSlide()
    visualizer.create_labeled_slide()
    print("Labeled graphs slide created successfully!")

if __name__ == "__main__":
    main() 