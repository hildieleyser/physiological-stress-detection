#!/usr/bin/env python3
"""
Visual-Only Single Slide Presentation for WESAD Dataset Analysis
No text on slide - all explanations in separate markdown file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Clean visual styling
plt.style.use('default')
sns.set_theme(style="whitegrid", font_scale=1.0)

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

# Minimal font settings for visual-only slide
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'axes.titleweight': 'bold',
    'lines.linewidth': 2,
    'axes.linewidth': 1,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.2
})

class VisualOnlySlide:
    def __init__(self, results_path: str = "results"):
        self.results_path = Path(results_path)
        self.output_path = Path("visual_only_slide")
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
    
    def create_visual_slide(self):
        """Create visual-only slide with no text."""
        print("Creating visual-only slide...")
        
        # Create large figure for single slide
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 6, hspace=0.3, wspace=0.2)
        
        # 1. Dataset Overview (Top Left) - No text
        ax1 = fig.add_subplot(gs[0, 0:2])
        self.plot_dataset_overview(ax1)
        
        # 2. Model Performance (Top Center) - No text
        ax2 = fig.add_subplot(gs[0, 2:4])
        self.plot_model_performance(ax2)
        
        # 3. Feature Importance (Top Right) - No text
        ax3 = fig.add_subplot(gs[0, 4:6])
        self.plot_feature_importance(ax3)
        
        # 4. Robustness Analysis (Bottom Left) - No text
        ax4 = fig.add_subplot(gs[1, 0:3])
        self.plot_robustness_analysis(ax4)
        
        # 5. Signal Mapping (Bottom Right) - No text
        ax5 = fig.add_subplot(gs[1, 3:6])
        self.plot_signal_mapping(ax5)
        
        # 6. Feature Engineering Flow (Bottom Full Width) - No text
        ax6 = fig.add_subplot(gs[2:, :])
        self.plot_feature_flow(ax6)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'visual_only_slide.png', dpi=300, 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Visual-only slide saved to: {self.output_path}")
    
    def plot_dataset_overview(self, ax):
        """Dataset overview with no text labels."""
        # Class distribution pie chart
        colors = [COLORS['stress'], COLORS['accent'], COLORS['primary']]
        wedges, texts, autotexts = ax.pie(self.class_counts, autopct='%1.0f%%', 
                                         colors=colors, startangle=90, 
                                         explode=(0.05, 0.05, 0.05))
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        # Remove all text labels
        for text in texts:
            text.set_text('')
    
    def plot_model_performance(self, ax):
        """Model performance with no text labels."""
        models = list(self.cv_results.keys())
        accuracies = [self.cv_results[model]['mean_accuracy'] for model in models]
        stds = [self.cv_results[model]['std_accuracy'] for model in models]
        
        # Create bar plot
        bars = ax.bar(range(len(models)), accuracies, yerr=stds, capsize=5, 
                     color=[COLORS['primary'], COLORS['secondary'], COLORS['accent']], 
                     alpha=0.8, edgecolor='white', linewidth=1.5)
        
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Highlight best model
        best_idx = np.argmax(accuracies)
        bars[best_idx].set_edgecolor(COLORS['dark'])
        bars[best_idx].set_linewidth(3)
    
    def plot_feature_importance(self, ax):
        """Feature importance with no text labels."""
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
        importances = [f[1] for f in sorted_features]
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(importances)), importances, 
                      color=COLORS['secondary'], alpha=0.8, edgecolor='white', linewidth=1.5)
        
        ax.grid(True, alpha=0.3, axis='x')
        
        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    def plot_robustness_analysis(self, ax):
        """Robustness analysis with no text labels."""
        noise_levels = self.robustness_results['logistic_regression']['noise_levels']
        
        # Plot accuracy vs noise for all models
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
        for i, model in enumerate(self.robustness_results):
            accuracies = self.robustness_results[model]['accuracies']
            ax.plot(noise_levels, accuracies, 'o-', linewidth=2.5, 
                   color=colors[i], markersize=6, markerfacecolor='white', 
                   markeredgecolor=colors[i], markeredgewidth=1.5)
        
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    def plot_signal_mapping(self, ax):
        """Signal mapping diagram with no text."""
        ax.axis('off')
        
        # Create visual signal mapping without text
        # Draw device outlines
        chest_rect = plt.Rectangle((0.1, 0.6), 0.35, 0.3, fill=False, 
                                 edgecolor=COLORS['primary'], linewidth=3)
        wrist_rect = plt.Rectangle((0.55, 0.6), 0.35, 0.3, fill=False, 
                                 edgecolor=COLORS['secondary'], linewidth=3)
        ax.add_patch(chest_rect)
        ax.add_patch(wrist_rect)
        
        # Draw signal channels as colored rectangles
        chest_signals = [COLORS['primary'], COLORS['accent'], COLORS['stress'], 
                        COLORS['neutral'], COLORS['secondary']]
        wrist_signals = [COLORS['secondary'], COLORS['accent'], COLORS['stress']]
        
        # Chest signals
        for i, color in enumerate(chest_signals):
            x = 0.12 + (i % 3) * 0.1
            y = 0.75 - (i // 3) * 0.1
            signal_rect = plt.Rectangle((x, y), 0.08, 0.08, 
                                      facecolor=color, alpha=0.7)
            ax.add_patch(signal_rect)
        
        # Wrist signals
        for i, color in enumerate(wrist_signals):
            x = 0.57 + i * 0.1
            y = 0.75
            signal_rect = plt.Rectangle((x, y), 0.08, 0.08, 
                                      facecolor=color, alpha=0.7)
            ax.add_patch(signal_rect)
        
        # Draw connection lines
        for i in range(3):
            x1 = 0.45
            y1 = 0.75 - i * 0.1
            x2 = 0.55
            y2 = 0.75
            ax.plot([x1, x2], [y1, y2], color=COLORS['neutral'], 
                   linewidth=2, alpha=0.5)
    
    def plot_feature_flow(self, ax):
        """Feature engineering flow diagram with no text."""
        ax.axis('off')
        
        # Create visual flow diagram
        # Raw signals
        raw_rect = plt.Rectangle((0.05, 0.7), 0.15, 0.2, 
                               facecolor=COLORS['primary'], alpha=0.3)
        ax.add_patch(raw_rect)
        
        # Processing arrows
        ax.arrow(0.22, 0.8, 0.08, 0, head_width=0.02, head_length=0.01, 
                fc=COLORS['neutral'], ec=COLORS['neutral'])
        
        # Feature extraction
        feature_rect = plt.Rectangle((0.32, 0.7), 0.15, 0.2, 
                                   facecolor=COLORS['secondary'], alpha=0.3)
        ax.add_patch(feature_rect)
        
        # Processing arrows
        ax.arrow(0.49, 0.8, 0.08, 0, head_width=0.02, head_length=0.01, 
                fc=COLORS['neutral'], ec=COLORS['neutral'])
        
        # Model training
        model_rect = plt.Rectangle((0.59, 0.7), 0.15, 0.2, 
                                 facecolor=COLORS['accent'], alpha=0.3)
        ax.add_patch(model_rect)
        
        # Processing arrows
        ax.arrow(0.76, 0.8, 0.08, 0, head_width=0.02, head_length=0.01, 
                fc=COLORS['neutral'], ec=COLORS['neutral'])
        
        # Results
        result_rect = plt.Rectangle((0.86, 0.7), 0.1, 0.2, 
                                  facecolor=COLORS['stress'], alpha=0.3)
        ax.add_patch(result_rect)
        
        # Draw feature types as small circles
        feature_types = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
                        COLORS['stress'], COLORS['neutral']]
        for i, color in enumerate(feature_types):
            x = 0.32 + (i % 3) * 0.04
            y = 0.6 - (i // 3) * 0.1
            circle = plt.Circle((x, y), 0.015, facecolor=color, alpha=0.7)
            ax.add_patch(circle)

def main():
    """Create visual-only slide."""
    visualizer = VisualOnlySlide()
    visualizer.create_visual_slide()
    print("Visual-only slide created successfully!")

if __name__ == "__main__":
    main() 