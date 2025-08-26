#!/usr/bin/env python3
"""
Final Presentation Slide with Specific Color Palette and Lato Font
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Specific color palette from the image
COLORS = {
    'dark_purple': '#190B28',      # Dark Purple
    'wenge': '#685762',            # Wenge
    'battleship_gray': '#9B9987',  # Battleship Gray
    'cherry_blossom': '#EFA9AE',   # Cherry Blossom Pink
    'blush': '#E55381',            # Blush
    'white': '#FFFFFF',            # White
    'light_gray': '#F8F9FA'        # Light gray for backgrounds
}

# Enhanced font settings with Lato
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Lato', 'Arial', 'DejaVu Sans'],
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
    'axes.facecolor': COLORS['light_gray'],
    'figure.facecolor': COLORS['white']
})

# Set modern style
plt.style.use('default')
sns.set_theme(style="whitegrid", font_scale=1.2)

def create_final_slide():
    """Create final presentation slide with specific colors and Lato font."""
    print("Creating final presentation slide...")
    
    # Load data
    results_path = Path("results")
    output_path = Path("final_slide")
    output_path.mkdir(exist_ok=True)
    
    with open(results_path / 'cv_summary.json', 'r') as f:
        cv_results = json.load(f)
    with open(results_path / 'shap_summary.json', 'r') as f:
        shap_results = json.load(f)
    with open(results_path / 'robustness_summary.json', 'r') as f:
        robustness_results = json.load(f)
    
    X = np.load(results_path / 'X.npy')
    y = np.load(results_path / 'y.npy')
    data_shape = X.shape
    class_counts = np.bincount(y)
    
    # Feature mapping
    feature_mapping = {
        'feature_25': 'Wrist BVP (Std)',
        'feature_32': 'Cross-Modal Correlation',
        'feature_9': 'Wrist ACC_Y (Mean)',
        'feature_41': 'Temporal Consistency',
        'feature_33': 'Heart Rate Variability'
    }
    
    # Create large figure for single slide
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 6, hspace=0.5, wspace=0.4)
    
    # Set background color
    fig.patch.set_facecolor(COLORS['white'])
    
    # Main title with Lato font
    fig.suptitle('WESAD Dataset: Wearable Stress Detection Analysis', 
                fontsize=26, fontweight='bold', color=COLORS['dark_purple'], y=0.98,
                fontfamily='Lato')
    
    # 1. Dataset Overview (Top Left)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.set_facecolor(COLORS['light_gray'])
    
    class_labels = ['Stress', 'Amusement', 'Resting']
    colors = [COLORS['blush'], COLORS['cherry_blossom'], COLORS['dark_purple']]
    
    wedges, texts, autotexts = ax1.pie(class_counts, labels=class_labels, 
                                     colors=colors, autopct='%1.0f%%', 
                                     startangle=90, explode=(0.08, 0.08, 0.08),
                                     shadow=True, textprops={'fontsize': 12, 'fontweight': 'bold', 'fontfamily': 'Lato'})
    
    for autotext in autotexts:
        autotext.set_color(COLORS['white'])
        autotext.set_fontweight('bold')
        autotext.set_fontsize(13)
        autotext.set_fontfamily('Lato')
    
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
        text.set_fontfamily('Lato')
    
    ax1.set_title('Dataset Composition', fontweight='bold', pad=25, 
                fontsize=18, fontfamily='Lato', color=COLORS['dark_purple'])
    
    ax1.text(0.5, -1.4, f'Total: {data_shape[0]:,} samples\n{data_shape[1]:,} features', 
           ha='center', va='center', fontweight='bold', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['white'], 
                    edgecolor=COLORS['wenge'], linewidth=2, alpha=0.9),
           fontfamily='Lato')
    
    # 2. Model Performance (Top Center)
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.set_facecolor(COLORS['light_gray'])
    
    models = ['Logistic\nRegression', 'Random\nForest', 'XGBoost']
    accuracies = [cv_results['logistic_regression']['mean_accuracy'],
                 cv_results['random_forest']['mean_accuracy'],
                 cv_results['xgboost']['mean_accuracy']]
    stds = [cv_results['logistic_regression']['std_accuracy'],
            cv_results['random_forest']['std_accuracy'],
            cv_results['xgboost']['std_accuracy']]
    
    bars = ax2.bar(models, accuracies, yerr=stds, capsize=8, 
                 color=[COLORS['dark_purple'], COLORS['wenge'], COLORS['battleship_gray']], 
                 alpha=0.9, edgecolor=COLORS['white'], linewidth=2)
    
    ax2.set_title('Model Performance (Cross-Validation)', fontweight='bold', pad=25, 
                fontsize=18, fontfamily='Lato', color=COLORS['dark_purple'])
    ax2.set_ylabel('Accuracy', fontweight='bold', fontsize=14, fontfamily='Lato')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    for bar, acc, std in zip(bars, accuracies, stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.03,
               f'{acc:.1%}\n±{std:.1%}', ha='center', va='bottom', 
               fontweight='bold', fontsize=11, fontfamily='Lato',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['white'], 
                        edgecolor=COLORS['wenge'], alpha=0.8))
    
    best_idx = np.argmax(accuracies)
    bars[best_idx].set_edgecolor(COLORS['blush'])
    bars[best_idx].set_linewidth(4)
    
    ax2.tick_params(axis='both', which='major', labelsize=11)
    for label in ax2.get_xticklabels():
        label.set_fontfamily('Lato')
        label.set_fontweight('bold')
    
    # 3. Feature Importance (Top Right)
    ax3 = fig.add_subplot(gs[0, 4:6])
    ax3.set_facecolor(COLORS['light_gray'])
    
    # Get top features
    feature_importance_dict = {}
    for model in shap_results:
        for feat in shap_results[model]['top_features'][:3]:
            feature = feat['feature']
            importance = feat['importance']
            if feature in feature_importance_dict:
                feature_importance_dict[feature] += importance
            else:
                feature_importance_dict[feature] = importance
    
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    feature_names = [f[0] for f in sorted_features]
    importances = [f[1] for f in sorted_features]
    
    meaningful_labels = [feature_mapping.get(f, f'Feature {f.split("_")[-1]}') for f in feature_names]
    
    colors = [COLORS['blush'], COLORS['cherry_blossom'], COLORS['dark_purple'], 
             COLORS['wenge'], COLORS['battleship_gray']]
    bars = ax3.barh(meaningful_labels, importances, 
                  color=colors, alpha=0.9, edgecolor=COLORS['white'], linewidth=2)
    
    ax3.set_title('Top 5 Physiological Features (SHAP Importance)', fontweight='bold', pad=25, 
                fontsize=18, fontfamily='Lato', color=COLORS['dark_purple'])
    ax3.set_xlabel('Importance Score', fontweight='bold', fontsize=14, fontfamily='Lato')
    ax3.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    for i, (bar, importance) in enumerate(zip(bars, importances)):
        width = bar.get_width()
        ax3.text(width + max(importances)*0.01, bar.get_y() + bar.get_height()/2, 
               f'{importance:.3f}', ha='left', va='center', fontweight='bold', 
               fontsize=11, fontfamily='Lato',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['white'], 
                        edgecolor=colors[i], alpha=0.8))
    
    ax3.tick_params(axis='both', which='major', labelsize=10)
    for label in ax3.get_yticklabels():
        label.set_fontfamily('Lato')
        label.set_fontweight('bold')
    
    # 4. Robustness Analysis (Bottom Left)
    ax4 = fig.add_subplot(gs[1, 0:3])
    ax4.set_facecolor(COLORS['light_gray'])
    
    noise_levels = robustness_results['logistic_regression']['noise_levels']
    colors = [COLORS['dark_purple'], COLORS['wenge'], COLORS['battleship_gray']]
    model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
    
    for i, model in enumerate(robustness_results):
        accuracies = robustness_results[model]['accuracies']
        ax4.plot(noise_levels, accuracies, 'o-', linewidth=3, 
               label=model_names[i], color=colors[i], markersize=8, 
               markerfacecolor=COLORS['white'], markeredgecolor=colors[i], 
               markeredgewidth=2, alpha=0.9)
    
    ax4.set_title('Model Robustness (Noise Sensitivity)', fontweight='bold', pad=25, 
                fontsize=18, fontfamily='Lato', color=COLORS['dark_purple'])
    ax4.set_xlabel('Noise Level (%)', fontweight='bold', fontsize=14, fontfamily='Lato')
    ax4.set_ylabel('Accuracy', fontweight='bold', fontsize=14, fontfamily='Lato')
    ax4.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, 
             loc='upper right')
    # Set legend font family
    legend = ax4.get_legend()
    for text in legend.get_texts():
        text.set_fontfamily('Lato')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_ylim(0, 1)
    
    robustness_scores = []
    for model in robustness_results:
        accuracies = robustness_results[model]['accuracies']
        degradation = accuracies[0] - accuracies[-1]
        robustness_scores.append(1 - degradation)
    
    ranking_text = "Robustness Ranking:\n"
    for i, (model, score) in enumerate(zip(model_names, robustness_scores)):
        ranking_text += f"{i+1}. {model}: {score:.3f}\n"
    
    ax4.text(0.02, 0.98, ranking_text, transform=ax4.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='Lato', fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['white'], 
                    edgecolor=COLORS['wenge'], linewidth=2, alpha=0.9))
    
    ax4.tick_params(axis='both', which='major', labelsize=11)
    for label in ax4.get_xticklabels() + ax4.get_yticklabels():
        label.set_fontfamily('Lato')
    
    # 5. Signal Mapping (Bottom Right)
    ax5 = fig.add_subplot(gs[1, 3:6])
    ax5.axis('off')
    ax5.set_facecolor(COLORS['light_gray'])
    
    chest_rect = plt.Rectangle((0.1, 0.6), 0.35, 0.3, fill=False, 
                             edgecolor=COLORS['dark_purple'], linewidth=4)
    wrist_rect = plt.Rectangle((0.55, 0.6), 0.35, 0.3, fill=False, 
                             edgecolor=COLORS['wenge'], linewidth=4)
    ax5.add_patch(chest_rect)
    ax5.add_patch(wrist_rect)
    
    ax5.text(0.275, 0.95, 'Chest Device (RespiBAN)', ha='center', va='center', 
           fontweight='bold', fontsize=14, color=COLORS['dark_purple'], 
           fontfamily='Lato',
           bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['white'], 
                    edgecolor=COLORS['dark_purple'], alpha=0.8))
    ax5.text(0.725, 0.95, 'Wrist Device (E4)', ha='center', va='center', 
           fontweight='bold', fontsize=14, color=COLORS['wenge'], 
           fontfamily='Lato',
           bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['white'], 
                    edgecolor=COLORS['wenge'], alpha=0.8))
    
    chest_signals = ['ACC', 'ECG', 'EDA', 'EMG', 'RESP']
    chest_colors = [COLORS['blush'], COLORS['cherry_blossom'], COLORS['dark_purple'], 
                   COLORS['wenge'], COLORS['battleship_gray']]
    
    wrist_signals = ['ACC', 'BVP', 'EDA']
    wrist_colors = [COLORS['wenge'], COLORS['cherry_blossom'], COLORS['dark_purple']]
    
    for i, (signal, color) in enumerate(zip(chest_signals, chest_colors)):
        x = 0.12 + (i % 3) * 0.1
        y = 0.75 - (i // 3) * 0.1
        signal_rect = plt.Rectangle((x, y), 0.08, 0.08, 
                                  facecolor=color, alpha=0.8, 
                                  edgecolor=COLORS['white'], linewidth=2)
        ax5.add_patch(signal_rect)
        ax5.text(x + 0.04, y + 0.04, signal, ha='center', va='center', 
               fontweight='bold', fontsize=9, color=COLORS['white'], 
               fontfamily='Lato')
    
    for i, (signal, color) in enumerate(zip(wrist_signals, wrist_colors)):
        x = 0.57 + i * 0.1
        y = 0.75
        signal_rect = plt.Rectangle((x, y), 0.08, 0.08, 
                                  facecolor=color, alpha=0.8, 
                                  edgecolor=COLORS['white'], linewidth=2)
        ax5.add_patch(signal_rect)
        ax5.text(x + 0.04, y + 0.04, signal, ha='center', va='center', 
               fontweight='bold', fontsize=9, color=COLORS['white'], 
               fontfamily='Lato')
    
    for i in range(3):
        x1 = 0.45
        y1 = 0.75 - i * 0.1
        x2 = 0.55
        y2 = 0.75
        ax5.plot([x1, x2], [y1, y2], color=COLORS['battleship_gray'], 
               linewidth=3, alpha=0.6, linestyle='--')
    
    ax5.set_title('WESAD Signal Mapping', fontweight='bold', pad=25, 
                fontsize=18, fontfamily='Lato', color=COLORS['dark_purple'])
    
    # 6. Feature Engineering Flow (Bottom Full Width)
    ax6 = fig.add_subplot(gs[2:, :])
    ax6.axis('off')
    ax6.set_facecolor(COLORS['light_gray'])
    
    raw_rect = plt.Rectangle((0.05, 0.7), 0.15, 0.2, 
                           facecolor=COLORS['dark_purple'], alpha=0.3, 
                           edgecolor=COLORS['dark_purple'], linewidth=2)
    ax6.add_patch(raw_rect)
    ax6.text(0.125, 0.8, 'Raw Signals\n(ACC, ECG, EDA,\nEMG, RESP, TEMP)', 
           ha='center', va='center', fontweight='bold', fontsize=11, 
           fontfamily='Lato',
           bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['white'], 
                    edgecolor=COLORS['dark_purple'], alpha=0.8))
    
    ax6.arrow(0.22, 0.8, 0.08, 0, head_width=0.02, head_length=0.01, 
            fc=COLORS['wenge'], ec=COLORS['wenge'], linewidth=3)
    
    feature_rect = plt.Rectangle((0.32, 0.7), 0.15, 0.2, 
                               facecolor=COLORS['wenge'], alpha=0.3, 
                               edgecolor=COLORS['wenge'], linewidth=2)
    ax6.add_patch(feature_rect)
    ax6.text(0.395, 0.8, 'Feature Extraction\n(Statistical Measures,\nTime Windows)', 
           ha='center', va='center', fontweight='bold', fontsize=11, 
           fontfamily='Lato',
           bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['white'], 
                    edgecolor=COLORS['wenge'], alpha=0.8))
    
    ax6.arrow(0.49, 0.8, 0.08, 0, head_width=0.02, head_length=0.01, 
            fc=COLORS['wenge'], ec=COLORS['wenge'], linewidth=3)
    
    model_rect = plt.Rectangle((0.59, 0.7), 0.15, 0.2, 
                             facecolor=COLORS['battleship_gray'], alpha=0.3, 
                             edgecolor=COLORS['battleship_gray'], linewidth=2)
    ax6.add_patch(model_rect)
    ax6.text(0.665, 0.8, 'Model Training\n(Cross-Validation,\nHyperparameter Tuning)', 
           ha='center', va='center', fontweight='bold', fontsize=11, 
           fontfamily='Lato',
           bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['white'], 
                    edgecolor=COLORS['battleship_gray'], alpha=0.8))
    
    ax6.arrow(0.76, 0.8, 0.08, 0, head_width=0.02, head_length=0.01, 
            fc=COLORS['wenge'], ec=COLORS['wenge'], linewidth=3)
    
    result_rect = plt.Rectangle((0.86, 0.7), 0.1, 0.2, 
                              facecolor=COLORS['blush'], alpha=0.3, 
                              edgecolor=COLORS['blush'], linewidth=2)
    ax6.add_patch(result_rect)
    ax6.text(0.91, 0.8, 'Stress Detection\nResults', 
           ha='center', va='center', fontweight='bold', fontsize=11, 
           fontfamily='Lato',
           bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['white'], 
                    edgecolor=COLORS['blush'], alpha=0.8))
    
    feature_types = ['Time\nDomain', 'Frequency\nDomain', 'Cross-\nModal', 'Temporal\nPatterns', 'Quality\nMetrics']
    feature_colors = [COLORS['blush'], COLORS['cherry_blossom'], COLORS['dark_purple'], 
                     COLORS['wenge'], COLORS['battleship_gray']]
    
    for i, (feature_type, color) in enumerate(zip(feature_types, feature_colors)):
        x = 0.32 + (i % 3) * 0.04
        y = 0.5 - (i // 3) * 0.15
        circle = plt.Circle((x, y), 0.015, facecolor=color, alpha=0.8, 
                          edgecolor=COLORS['white'], linewidth=2)
        ax6.add_patch(circle)
        ax6.text(x, y - 0.03, feature_type, ha='center', va='center', 
               fontweight='bold', fontsize=8, fontfamily='Lato',
               bbox=dict(boxstyle="round,pad=0.2", facecolor=COLORS['white'], 
                        edgecolor=color, alpha=0.8))
    
    ax6.set_title('Feature Engineering Pipeline', fontweight='bold', pad=25, 
                fontsize=18, fontfamily='Lato', color=COLORS['dark_purple'])
    
    plt.tight_layout()
    plt.savefig(output_path / 'final_slide.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Final presentation slide saved to: {output_path}")

if __name__ == "__main__":
    create_final_slide() 