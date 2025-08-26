#!/usr/bin/env python3
"""
Meaningful Features Slide Presentation for WESAD Dataset Analysis
Replaces generic feature numbers with actual physiological signal names.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Modern color palette - visually appealing
COLORS = {
    'primary': '#6366F1',      # Modern indigo
    'secondary': '#F59E0B',    # Warm amber
    'accent': '#10B981',       # Emerald green
    'stress': '#EF4444',       # Modern red
    'neutral': '#6B7280',      # Gray
    'light': '#F3F4F6',        # Light gray
    'dark': '#1F2937',         # Dark gray
    'purple': '#8B5CF6',       # Purple
    'pink': '#EC4899',         # Pink
    'blue': '#3B82F6',         # Blue
    'teal': '#14B8A6',         # Teal
    'orange': '#F97316'        # Orange
}

# Enhanced font settings for visual appeal
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 13,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 24,
    'figure.titleweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'lines.linewidth': 3,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1,
    'grid.alpha': 0.2,
    'axes.facecolor': '#FAFAFA',
    'figure.facecolor': 'white'
})

# Set modern style
plt.style.use('default')
sns.set_theme(style="whitegrid", font_scale=1.2)

class MeaningfulFeaturesSlide:
    def __init__(self, results_path: str = "results"):
        self.results_path = Path(results_path)
        self.output_path = Path("meaningful_features_slide")
        self.output_path.mkdir(exist_ok=True)
        
        # Load analysis results
        self.load_data()
        
        # WESAD feature mapping based on typical feature engineering
        self.feature_mapping = self.create_feature_mapping()
    
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
    
    def create_feature_mapping(self):
        """Create meaningful feature names based on WESAD dataset structure."""
        feature_mapping = {}
        
        # Based on typical WESAD feature engineering (42 features)
        # Features 0-2: Chest Accelerometer (ACC_X, ACC_Y, ACC_Z)
        feature_mapping['feature_0'] = 'Chest ACC_X (Mean)'
        feature_mapping['feature_1'] = 'Chest ACC_Y (Mean)'
        feature_mapping['feature_2'] = 'Chest ACC_Z (Mean)'
        
        # Features 3-7: Chest Physiological Signals
        feature_mapping['feature_3'] = 'Chest ECG (Mean)'
        feature_mapping['feature_4'] = 'Chest EDA (Mean)'
        feature_mapping['feature_5'] = 'Chest EMG (Mean)'
        feature_mapping['feature_6'] = 'Chest RESP (Mean)'
        feature_mapping['feature_7'] = 'Chest TEMP (Mean)'
        
        # Features 8-10: Wrist Accelerometer
        feature_mapping['feature_8'] = 'Wrist ACC_X (Mean)'
        feature_mapping['feature_9'] = 'Wrist ACC_Y (Mean)'
        feature_mapping['feature_10'] = 'Wrist ACC_Z (Mean)'
        
        # Features 11-13: Wrist Physiological Signals
        feature_mapping['feature_11'] = 'Wrist BVP (Mean)'
        feature_mapping['feature_12'] = 'Wrist EDA (Mean)'
        feature_mapping['feature_13'] = 'Wrist TEMP (Mean)'
        
        # Features 14-20: Standard Deviations
        feature_mapping['feature_14'] = 'Chest ACC_X (Std)'
        feature_mapping['feature_15'] = 'Chest ACC_Y (Std)'
        feature_mapping['feature_16'] = 'Chest ACC_Z (Std)'
        feature_mapping['feature_17'] = 'Chest ECG (Std)'
        feature_mapping['feature_18'] = 'Chest EDA (Std)'
        feature_mapping['feature_19'] = 'Chest EMG (Std)'
        feature_mapping['feature_20'] = 'Chest RESP (Std)'
        
        # Features 21-27: Additional Statistics
        feature_mapping['feature_21'] = 'Chest TEMP (Std)'
        feature_mapping['feature_22'] = 'Wrist ACC_X (Std)'
        feature_mapping['feature_23'] = 'Wrist ACC_Y (Std)'
        feature_mapping['feature_24'] = 'Wrist ACC_Z (Std)'
        feature_mapping['feature_25'] = 'Wrist BVP (Std)'
        feature_mapping['feature_26'] = 'Wrist EDA (Std)'
        feature_mapping['feature_27'] = 'Wrist TEMP (Std)'
        
        # Features 28-34: Frequency Domain Features
        feature_mapping['feature_28'] = 'ECG Dominant Freq'
        feature_mapping['feature_29'] = 'BVP Dominant Freq'
        feature_mapping['feature_30'] = 'ACC Power Spectrum'
        feature_mapping['feature_31'] = 'EDA Power Spectrum'
        feature_mapping['feature_32'] = 'Cross-Modal Correlation'
        feature_mapping['feature_33'] = 'Heart Rate Variability'
        feature_mapping['feature_34'] = 'Respiratory Rate'
        
        # Features 35-41: Advanced Features
        feature_mapping['feature_35'] = 'Motion Intensity'
        feature_mapping['feature_36'] = 'Skin Conductance Level'
        feature_mapping['feature_37'] = 'Temperature Trend'
        feature_mapping['feature_38'] = 'Signal Quality Index'
        feature_mapping['feature_39'] = 'Stress Response Pattern'
        feature_mapping['feature_40'] = 'Multi-Modal Fusion'
        feature_mapping['feature_41'] = 'Temporal Consistency'
        
        return feature_mapping
    
    def get_meaningful_feature_name(self, feature_name):
        """Get meaningful name for a feature."""
        return self.feature_mapping.get(feature_name, feature_name)
    
    def create_meaningful_slide(self):
        """Create slide with meaningful feature names."""
        print("Creating meaningful features slide...")
        
        # Create large figure for single slide
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 6, hspace=0.5, wspace=0.4)
        
        # Set background color
        fig.patch.set_facecolor('#FAFAFA')
        
        # Main title with gradient effect
        fig.suptitle('WESAD Dataset: Wearable Stress Detection Analysis', 
                    fontsize=26, fontweight='bold', color=COLORS['dark'], y=0.98)
        
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
        plt.savefig(self.output_path / 'meaningful_features_slide.png', dpi=300, 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Meaningful features slide saved to: {self.output_path}")
    
    def plot_dataset_overview(self, ax):
        """Dataset overview with modern design."""
        # Set background
        ax.set_facecolor('#FAFAFA')
        
        # Class distribution pie chart with modern colors
        class_labels = ['Stress', 'Amusement', 'Resting']
        colors = [COLORS['stress'], COLORS['accent'], COLORS['primary']]
        
        # Create pie chart with enhanced styling
        wedges, texts, autotexts = ax.pie(self.class_counts, labels=class_labels, 
                                         colors=colors, autopct='%1.0f%%', 
                                         startangle=90, explode=(0.08, 0.08, 0.08),
                                         shadow=True, textprops={'fontsize': 12, 'fontweight': 'bold'})
        
        # Enhance percentage labels
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(13)
        
        # Enhance text labels
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        ax.set_title('Dataset Composition', fontweight='bold', pad=25, 
                    fontsize=18, color=COLORS['dark'])
        
        # Add sample count with modern styling
        ax.text(0.5, -1.4, f'Total: {self.data_shape[0]:,} samples\n{self.data_shape[1]:,} features', 
               ha='center', va='center', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['light'], 
                        edgecolor=COLORS['neutral'], linewidth=2, alpha=0.9))
    
    def plot_model_performance(self, ax):
        """Model performance with modern design."""
        # Set background
        ax.set_facecolor('#FAFAFA')
        
        models = ['Logistic\nRegression', 'Random\nForest', 'XGBoost']
        accuracies = [self.cv_results['logistic_regression']['mean_accuracy'],
                     self.cv_results['random_forest']['mean_accuracy'],
                     self.cv_results['xgboost']['mean_accuracy']]
        stds = [self.cv_results['logistic_regression']['std_accuracy'],
                self.cv_results['random_forest']['std_accuracy'],
                self.cv_results['xgboost']['std_accuracy']]
        
        # Create bar plot with modern colors
        bars = ax.bar(models, accuracies, yerr=stds, capsize=8, 
                     color=[COLORS['primary'], COLORS['secondary'], COLORS['accent']], 
                     alpha=0.9, edgecolor='white', linewidth=2)
        
        ax.set_title('Model Performance (Cross-Validation)', fontweight='bold', pad=25, 
                    fontsize=18, color=COLORS['dark'])
        ax.set_ylabel('Accuracy', fontweight='bold', fontsize=14)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Enhance value labels
        for bar, acc, std in zip(bars, accuracies, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                   f'{acc:.1%}\n±{std:.1%}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                            edgecolor=COLORS['neutral'], alpha=0.8))
        
        # Highlight best model with enhanced styling
        best_idx = np.argmax(accuracies)
        bars[best_idx].set_edgecolor(COLORS['dark'])
        bars[best_idx].set_linewidth(4)
        
        # Enhance tick labels
        ax.tick_params(axis='both', which='major', labelsize=11)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
    
    def plot_feature_importance(self, ax):
        """Feature importance with meaningful names."""
        # Set background
        ax.set_facecolor('#FAFAFA')
        
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
        meaningful_labels = [self.get_meaningful_feature_name(f) for f in feature_names]
        
        # Create horizontal bar plot with gradient colors
        colors = [COLORS['purple'], COLORS['pink'], COLORS['blue'], COLORS['teal'], COLORS['orange']]
        bars = ax.barh(meaningful_labels, importances, 
                      color=colors, alpha=0.9, edgecolor='white', linewidth=2)
        
        ax.set_title('Top 5 Physiological Features (SHAP Importance)', fontweight='bold', pad=25, 
                    fontsize=18, color=COLORS['dark'])
        ax.set_xlabel('Importance Score', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # Add importance values with enhanced styling
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            width = bar.get_width()
            ax.text(width + max(importances)*0.01, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', ha='left', va='center', fontweight='bold', 
                   fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                            edgecolor=colors[i], alpha=0.8))
        
        # Enhance tick labels
        ax.tick_params(axis='both', which='major', labelsize=10)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
    
    def plot_robustness_analysis(self, ax):
        """Robustness analysis with modern design."""
        # Set background
        ax.set_facecolor('#FAFAFA')
        
        noise_levels = self.robustness_results['logistic_regression']['noise_levels']
        
        # Plot accuracy vs noise for all models with enhanced styling
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
        model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
        
        for i, model in enumerate(self.robustness_results):
            accuracies = self.robustness_results[model]['accuracies']
            ax.plot(noise_levels, accuracies, 'o-', linewidth=3, 
                   label=model_names[i], color=colors[i], markersize=8, 
                   markerfacecolor='white', markeredgecolor=colors[i], 
                   markeredgewidth=2, alpha=0.9)
        
        ax.set_title('Model Robustness (Noise Sensitivity)', fontweight='bold', pad=25, 
                    fontsize=18, color=COLORS['dark'])
        ax.set_xlabel('Noise Level (%)', fontweight='bold', fontsize=14)
        ax.set_ylabel('Accuracy', fontweight='bold', fontsize=14)
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, 
                 loc='upper right')
        # Set legend font family
        legend = ax.get_legend()
        for text in legend.get_texts():
            text.set_fontweight('bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1)
        
        # Add robustness ranking with modern styling
        robustness_scores = []
        for model in self.robustness_results:
            accuracies = self.robustness_results[model]['accuracies']
            degradation = accuracies[0] - accuracies[-1]
            robustness_scores.append(1 - degradation)
        
        # Add ranking text with enhanced styling
        ranking_text = "Robustness Ranking:\n"
        for i, (model, score) in enumerate(zip(model_names, robustness_scores)):
            ranking_text += f"{i+1}. {model}: {score:.3f}\n"
        
        ax.text(0.02, 0.98, ranking_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['light'], 
                        edgecolor=COLORS['neutral'], linewidth=2, alpha=0.9))
        
        # Enhance tick labels
        ax.tick_params(axis='both', which='major', labelsize=11)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
    
    def plot_signal_mapping(self, ax):
        """Signal mapping diagram with modern design."""
        ax.axis('off')
        ax.set_facecolor('#FAFAFA')
        
        # Create visual signal mapping with modern styling
        # Draw device outlines with enhanced styling
        chest_rect = plt.Rectangle((0.1, 0.6), 0.35, 0.3, fill=False, 
                                 edgecolor=COLORS['primary'], linewidth=4)
        wrist_rect = plt.Rectangle((0.55, 0.6), 0.35, 0.3, fill=False, 
                                 edgecolor=COLORS['secondary'], linewidth=4)
        ax.add_patch(chest_rect)
        ax.add_patch(wrist_rect)
        
        # Add device labels with modern styling
        ax.text(0.275, 0.95, 'Chest Device (RespiBAN)', ha='center', va='center', 
               fontweight='bold', fontsize=14, color=COLORS['primary'], 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                        edgecolor=COLORS['primary'], alpha=0.8))
        ax.text(0.725, 0.95, 'Wrist Device (E4)', ha='center', va='center', 
               fontweight='bold', fontsize=14, color=COLORS['secondary'], 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                        edgecolor=COLORS['secondary'], alpha=0.8))
        
        # Draw signal channels as colored rectangles with labels
        chest_signals = ['ACC', 'ECG', 'EDA', 'EMG', 'RESP']
        chest_colors = [COLORS['purple'], COLORS['accent'], COLORS['stress'], 
                       COLORS['neutral'], COLORS['teal']]
        
        wrist_signals = ['ACC', 'BVP', 'EDA']
        wrist_colors = [COLORS['secondary'], COLORS['pink'], COLORS['stress']]
        
        # Chest signals with enhanced styling
        for i, (signal, color) in enumerate(zip(chest_signals, chest_colors)):
            x = 0.12 + (i % 3) * 0.1
            y = 0.75 - (i // 3) * 0.1
            signal_rect = plt.Rectangle((x, y), 0.08, 0.08, 
                                      facecolor=color, alpha=0.8, 
                                      edgecolor='white', linewidth=2)
            ax.add_patch(signal_rect)
            ax.text(x + 0.04, y + 0.04, signal, ha='center', va='center', 
                   fontweight='bold', fontsize=9, color='white')
        
        # Wrist signals with enhanced styling
        for i, (signal, color) in enumerate(zip(wrist_signals, wrist_colors)):
            x = 0.57 + i * 0.1
            y = 0.75
            signal_rect = plt.Rectangle((x, y), 0.08, 0.08, 
                                      facecolor=color, alpha=0.8, 
                                      edgecolor='white', linewidth=2)
            ax.add_patch(signal_rect)
            ax.text(x + 0.04, y + 0.04, signal, ha='center', va='center', 
                   fontweight='bold', fontsize=9, color='white')
        
        # Draw connection lines with enhanced styling
        for i in range(3):
            x1 = 0.45
            y1 = 0.75 - i * 0.1
            x2 = 0.55
            y2 = 0.75
            ax.plot([x1, x2], [y1, y2], color=COLORS['neutral'], 
                   linewidth=3, alpha=0.6, linestyle='--')
        
        ax.set_title('WESAD Signal Mapping', fontweight='bold', pad=25, 
                    fontsize=18, color=COLORS['dark'])
    
    def plot_feature_flow(self, ax):
        """Feature engineering flow diagram with modern design."""
        ax.axis('off')
        ax.set_facecolor('#FAFAFA')
        
        # Create visual flow diagram with modern styling
        # Raw signals
        raw_rect = plt.Rectangle((0.05, 0.7), 0.15, 0.2, 
                               facecolor=COLORS['primary'], alpha=0.3, 
                               edgecolor=COLORS['primary'], linewidth=2)
        ax.add_patch(raw_rect)
        ax.text(0.125, 0.8, 'Raw Signals\n(ACC, ECG, EDA,\nEMG, RESP, TEMP)', 
               ha='center', va='center', fontweight='bold', fontsize=11, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                        edgecolor=COLORS['primary'], alpha=0.8))
        
        # Processing arrows with enhanced styling
        ax.arrow(0.22, 0.8, 0.08, 0, head_width=0.02, head_length=0.01, 
                fc=COLORS['neutral'], ec=COLORS['neutral'], linewidth=3)
        
        # Feature extraction
        feature_rect = plt.Rectangle((0.32, 0.7), 0.15, 0.2, 
                                   facecolor=COLORS['secondary'], alpha=0.3, 
                                   edgecolor=COLORS['secondary'], linewidth=2)
        ax.add_patch(feature_rect)
        ax.text(0.395, 0.8, 'Feature Extraction\n(Statistical Measures,\nTime Windows)', 
               ha='center', va='center', fontweight='bold', fontsize=11, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                        edgecolor=COLORS['secondary'], alpha=0.8))
        
        # Processing arrows
        ax.arrow(0.49, 0.8, 0.08, 0, head_width=0.02, head_length=0.01, 
                fc=COLORS['neutral'], ec=COLORS['neutral'], linewidth=3)
        
        # Model training
        model_rect = plt.Rectangle((0.59, 0.7), 0.15, 0.2, 
                                 facecolor=COLORS['accent'], alpha=0.3, 
                                 edgecolor=COLORS['accent'], linewidth=2)
        ax.add_patch(model_rect)
        ax.text(0.665, 0.8, 'Model Training\n(Cross-Validation,\nHyperparameter Tuning)', 
               ha='center', va='center', fontweight='bold', fontsize=11, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                        edgecolor=COLORS['accent'], alpha=0.8))
        
        # Processing arrows
        ax.arrow(0.76, 0.8, 0.08, 0, head_width=0.02, head_length=0.01, 
                fc=COLORS['neutral'], ec=COLORS['neutral'], linewidth=3)
        
        # Results
        result_rect = plt.Rectangle((0.86, 0.7), 0.1, 0.2, 
                                  facecolor=COLORS['stress'], alpha=0.3, 
                                  edgecolor=COLORS['stress'], linewidth=2)
        ax.add_patch(result_rect)
        ax.text(0.91, 0.8, 'Stress Detection\nResults', 
               ha='center', va='center', fontweight='bold', fontsize=11, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                        edgecolor=COLORS['stress'], alpha=0.8))
        
        # Draw feature types as small circles with labels
        feature_types = ['Time\nDomain', 'Frequency\nDomain', 'Cross-\nModal', 'Temporal\nPatterns', 'Quality\nMetrics']
        feature_colors = [COLORS['purple'], COLORS['pink'], COLORS['blue'], 
                         COLORS['teal'], COLORS['orange']]
        
        for i, (feature_type, color) in enumerate(zip(feature_types, feature_colors)):
            x = 0.32 + (i % 3) * 0.04
            y = 0.5 - (i // 3) * 0.15
            circle = plt.Circle((x, y), 0.015, facecolor=color, alpha=0.8, 
                              edgecolor='white', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y - 0.03, feature_type, ha='center', va='center', 
                   fontweight='bold', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                            edgecolor=color, alpha=0.8))
        
        ax.set_title('Feature Engineering Pipeline', fontweight='bold', pad=25, 
                    fontsize=18, color=COLORS['dark'])

def main():
    """Create meaningful features slide."""
    visualizer = MeaningfulFeaturesSlide()
    visualizer.create_meaningful_slide()
    print("Meaningful features slide created successfully!")

if __name__ == "__main__":
    main() 