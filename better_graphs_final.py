#!/usr/bin/env python3
"""
Better Graphs Final - Recreating the Better Version with Proper Fonts
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

# Enhanced font settings - using Helvetica without bold
plt.rcParams.update({
    'font.family': 'Helvetica',
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 28,
    'figure.titleweight': 'normal',
    'axes.titleweight': 'normal',
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
sns.set_theme(style="whitegrid", font_scale=1.3)

def create_better_graphs():
    """Create better graphs with proper fonts and spacing."""
    print("Creating better graphs with proper fonts...")
    
    # Load data
    results_path = Path("results")
    output_path = Path("better_graphs_final")
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
        'feature_25': 'Heart Rate Variability\n(ECG/BVP Frequency)',
        'feature_32': 'Cross-Modal Correlation\n(Chest-Wrist Sync)',
        'feature_9': 'Wrist Movement\n(ACC_Y Statistics)',
        'feature_41': 'Signal Quality\n(Artifact Detection)',
        'feature_30': 'Breathing Pattern\n(RESP Statistics)',
        'feature_39': 'Temperature Stability\n(TEMP Variance)'
    }
    
    # 1. Dataset Overview (Top Left)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor(COLORS['light_gray'])
    
    class_labels = ['Stress', 'Amusement', 'Resting']
    colors = [COLORS['blush'], COLORS['cherry_blossom'], COLORS['dark_purple']]
    
    wedges, texts, autotexts = ax.pie(class_counts, labels=class_labels, 
                                     colors=colors, autopct='%1.0f%%', 
                                     startangle=90, explode=(0.08, 0.08, 0.08),
                                     shadow=True, textprops={'fontsize': 14, 'fontweight': 'normal'})
    
    for autotext in autotexts:
        autotext.set_color(COLORS['white'])
        autotext.set_fontweight('normal')
        autotext.set_fontsize(15)
    
    for text in texts:
        text.set_fontsize(14)
        text.set_fontweight('normal')
    
    ax.set_title('Dataset Composition', fontweight='normal', pad=30, 
                fontsize=20, color=COLORS['dark_purple'])
    
    ax.text(0.5, -1.5, f'Total: {data_shape[0]:,} samples\n{data_shape[1]:,} features', 
           ha='center', va='center', fontweight='normal', fontsize=14,
           bbox=dict(boxstyle="round,pad=0.6", facecolor=COLORS['white'], 
                    edgecolor=COLORS['wenge'], linewidth=2, alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path / '01_dataset_overview.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # 2. Model Performance (Top Center)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor(COLORS['light_gray'])
    
    models = ['Logistic\nRegression', 'Random\nForest', 'XGBoost']
    accuracies = [cv_results['logistic_regression']['mean_accuracy'],
                 cv_results['random_forest']['mean_accuracy'],
                 cv_results['xgboost']['mean_accuracy']]
    stds = [cv_results['logistic_regression']['std_accuracy'],
            cv_results['random_forest']['std_accuracy'],
            cv_results['xgboost']['std_accuracy']]
    
    bars = ax.bar(models, accuracies, yerr=stds, capsize=10, 
                 color=[COLORS['dark_purple'], COLORS['wenge'], COLORS['battleship_gray']], 
                 alpha=0.9, edgecolor=COLORS['white'], linewidth=2)
    
    ax.set_title('Model Performance (Cross-Validation)', fontweight='normal', pad=30, 
                fontsize=20, color=COLORS['dark_purple'])
    ax.set_ylabel('Accuracy', fontweight='normal', fontsize=16)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    for bar, acc, std in zip(bars, accuracies, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.04,
               f'{acc:.1%}\n±{std:.1%}', ha='center', va='bottom', 
               fontweight='normal', fontsize=13,
               bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS['white'], 
                        edgecolor=COLORS['wenge'], alpha=0.8))
    
    best_idx = np.argmax(accuracies)
    bars[best_idx].set_edgecolor(COLORS['blush'])
    bars[best_idx].set_linewidth(4)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels():
        label.set_fontweight('normal')
    
    plt.tight_layout()
    plt.savefig(output_path / '02_model_performance.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # 3. Feature Importance (Top Right)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor(COLORS['light_gray'])
    
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
    bars = ax.barh(meaningful_labels, importances, 
                  color=colors, alpha=0.9, edgecolor=COLORS['white'], linewidth=2)
    
    ax.set_title('Top 5 Physiological Features (SHAP Importance)', fontweight='normal', pad=30, 
                fontsize=20, color=COLORS['dark_purple'])
    ax.set_xlabel('Importance Score', fontweight='normal', fontsize=16)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    for i, (bar, importance) in enumerate(zip(bars, importances)):
        width = bar.get_width()
        ax.text(width + max(importances)*0.01, bar.get_y() + bar.get_height()/2, 
               f'{importance:.3f}', ha='left', va='center', fontweight='normal', 
               fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['white'], 
                        edgecolor=colors[i], alpha=0.8))
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    for label in ax.get_yticklabels():
        label.set_fontweight('normal')
    
    plt.tight_layout()
    plt.savefig(output_path / '03_feature_importance.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # 4. Robustness Analysis (Bottom Left)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor(COLORS['light_gray'])
    
    noise_levels = robustness_results['logistic_regression']['noise_levels']
    colors = [COLORS['dark_purple'], COLORS['wenge'], COLORS['battleship_gray']]
    model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
    
    for i, model in enumerate(robustness_results):
        accuracies = robustness_results[model]['accuracies']
        ax.plot(noise_levels, accuracies, 'o-', linewidth=3, 
               label=model_names[i], color=colors[i], markersize=10, 
               markerfacecolor=COLORS['white'], markeredgecolor=colors[i], 
               markeredgewidth=2, alpha=0.9)
    
    ax.set_title('Model Robustness (Noise Sensitivity)', fontweight='normal', pad=30, 
                fontsize=20, color=COLORS['dark_purple'])
    ax.set_xlabel('Noise Level (%)', fontweight='normal', fontsize=16)
    ax.set_ylabel('Accuracy', fontweight='normal', fontsize=16)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=13, 
             loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1)
    
    robustness_scores = []
    for model in robustness_results:
        accuracies = robustness_results[model]['accuracies']
        degradation = accuracies[0] - accuracies[-1]
        robustness_scores.append(1 - degradation)
    
    ranking_text = "Robustness Ranking:\n"
    for i, (model, score) in enumerate(zip(model_names, robustness_scores)):
        ranking_text += f"{i+1}. {model}: {score:.3f}\n"
    
    ax.text(0.02, 0.98, ranking_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontweight='normal',
           bbox=dict(boxstyle="round,pad=0.6", facecolor=COLORS['white'], 
                    edgecolor=COLORS['wenge'], linewidth=2, alpha=0.9))
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('normal')
    
    plt.tight_layout()
    plt.savefig(output_path / '04_robustness_analysis.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # 5. WESAD Device Architecture (Bottom Right)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    ax.set_facecolor(COLORS['light_gray'])
    
    # Title with clear explanation
    ax.set_title('WESAD Multi-Device Architecture\n(Chest + Wrist Sensors)', fontweight='normal', pad=30, 
                fontsize=20, color=COLORS['dark_purple'])
    
    # Device boxes with clear labels
    chest_rect = plt.Rectangle((0.1, 0.5), 0.35, 0.35, fill=False, 
                             edgecolor=COLORS['dark_purple'], linewidth=4)
    wrist_rect = plt.Rectangle((0.55, 0.5), 0.35, 0.35, fill=False, 
                             edgecolor=COLORS['wenge'], linewidth=4)
    ax.add_patch(chest_rect)
    ax.add_patch(wrist_rect)
    
    ax.text(0.275, 0.9, 'Chest Device (RespiBAN)\n700 Hz Sampling', ha='center', va='center', 
           fontweight='normal', fontsize=14, color=COLORS['dark_purple'], 
           bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS['white'], 
                    edgecolor=COLORS['dark_purple'], alpha=0.8))
    ax.text(0.725, 0.9, 'Wrist Device (E4)\n4-64 Hz Sampling', ha='center', va='center', 
           fontweight='normal', fontsize=14, color=COLORS['wenge'], 
           bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS['white'], 
                    edgecolor=COLORS['wenge'], alpha=0.8))
    
    # Signal types with clear labels
    chest_signals = ['ACC\n(Motion)', 'ECG\n(Heart)', 'EDA\n(Skin)', 'EMG\n(Muscle)', 'RESP\n(Breathing)']
    chest_colors = [COLORS['blush'], COLORS['cherry_blossom'], COLORS['dark_purple'], 
                   COLORS['wenge'], COLORS['battleship_gray']]
    
    wrist_signals = ['ACC\n(Motion)', 'BVP\n(Blood)', 'EDA\n(Skin)']
    wrist_colors = [COLORS['wenge'], COLORS['cherry_blossom'], COLORS['dark_purple']]
    
    # Chest signals
    for i, (signal, color) in enumerate(zip(chest_signals, chest_colors)):
        x = 0.12 + (i % 3) * 0.1
        y = 0.65 - (i // 3) * 0.12
        signal_rect = plt.Rectangle((x, y), 0.08, 0.1, 
                                  facecolor=color, alpha=0.8, 
                                  edgecolor=COLORS['white'], linewidth=2)
        ax.add_patch(signal_rect)
        ax.text(x + 0.04, y + 0.05, signal, ha='center', va='center', 
               fontweight='normal', fontsize=9, color=COLORS['white'])
    
    # Wrist signals
    for i, (signal, color) in enumerate(zip(wrist_signals, wrist_colors)):
        x = 0.57 + i * 0.1
        y = 0.65
        signal_rect = plt.Rectangle((x, y), 0.08, 0.1, 
                                  facecolor=color, alpha=0.8, 
                                  edgecolor=COLORS['white'], linewidth=2)
        ax.add_patch(signal_rect)
        ax.text(x + 0.04, y + 0.05, signal, ha='center', va='center', 
               fontweight='normal', fontsize=9, color=COLORS['white'])
    
    # Connection arrows showing synchronization
    for i in range(3):
        x1 = 0.45
        y1 = 0.65 - i * 0.12
        x2 = 0.55
        y2 = 0.65
        ax.plot([x1, x2], [y1, y2], color=COLORS['battleship_gray'], 
               linewidth=3, alpha=0.6, linestyle='--')
    
    # Explanation text
    ax.text(0.5, 0.35, 'Synchronized via Double-Tap Gesture\nMulti-modal stress detection across body locations', 
           ha='center', va='center', fontweight='normal', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.6", facecolor=COLORS['white'], 
                    edgecolor=COLORS['battleship_gray'], linewidth=2, alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path / '05_device_architecture.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # 6. Feature Engineering Pipeline (Bottom Full Width)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')
    ax.set_facecolor(COLORS['light_gray'])
    
    raw_rect = plt.Rectangle((0.05, 0.7), 0.15, 0.2, 
                           facecolor=COLORS['dark_purple'], alpha=0.3, 
                           edgecolor=COLORS['dark_purple'], linewidth=2)
    ax.add_patch(raw_rect)
    ax.text(0.125, 0.8, 'Raw Signals\n(ACC, ECG, EDA,\nEMG, RESP, TEMP)', 
           ha='center', va='center', fontweight='normal', fontsize=13, 
           bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS['white'], 
                    edgecolor=COLORS['dark_purple'], alpha=0.8))
    
    ax.arrow(0.22, 0.8, 0.08, 0, head_width=0.02, head_length=0.01, 
            fc=COLORS['wenge'], ec=COLORS['wenge'], linewidth=3)
    
    feature_rect = plt.Rectangle((0.32, 0.7), 0.15, 0.2, 
                               facecolor=COLORS['wenge'], alpha=0.3, 
                               edgecolor=COLORS['wenge'], linewidth=2)
    ax.add_patch(feature_rect)
    ax.text(0.395, 0.8, 'Feature Extraction\n(Statistical Measures,\nTime Windows)', 
           ha='center', va='center', fontweight='normal', fontsize=13, 
           bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS['white'], 
                    edgecolor=COLORS['wenge'], alpha=0.8))
    
    ax.arrow(0.49, 0.8, 0.08, 0, head_width=0.02, head_length=0.01, 
            fc=COLORS['wenge'], ec=COLORS['wenge'], linewidth=3)
    
    model_rect = plt.Rectangle((0.59, 0.7), 0.15, 0.2, 
                             facecolor=COLORS['battleship_gray'], alpha=0.3, 
                             edgecolor=COLORS['battleship_gray'], linewidth=2)
    ax.add_patch(model_rect)
    ax.text(0.665, 0.8, 'Model Training\n(Cross-Validation,\nHyperparameter Tuning)', 
           ha='center', va='center', fontweight='normal', fontsize=13, 
           bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS['white'], 
                    edgecolor=COLORS['battleship_gray'], alpha=0.8))
    
    ax.arrow(0.76, 0.8, 0.08, 0, head_width=0.02, head_length=0.01, 
            fc=COLORS['wenge'], ec=COLORS['wenge'], linewidth=3)
    
    result_rect = plt.Rectangle((0.86, 0.7), 0.1, 0.2, 
                              facecolor=COLORS['blush'], alpha=0.3, 
                              edgecolor=COLORS['blush'], linewidth=2)
    ax.add_patch(result_rect)
    ax.text(0.91, 0.8, 'Stress Detection\nResults', 
           ha='center', va='center', fontweight='normal', fontsize=13, 
           bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS['white'], 
                    edgecolor=COLORS['blush'], alpha=0.8))
    
    feature_types = ['Time\nDomain', 'Frequency\nDomain', 'Cross-\nModal', 'Temporal\nPatterns', 'Quality\nMetrics']
    feature_colors = [COLORS['blush'], COLORS['cherry_blossom'], COLORS['dark_purple'], 
                     COLORS['wenge'], COLORS['battleship_gray']]
    
    for i, (feature_type, color) in enumerate(zip(feature_types, feature_colors)):
        x = 0.32 + (i % 3) * 0.04
        y = 0.5 - (i // 3) * 0.15
        circle = plt.Circle((x, y), 0.015, facecolor=color, alpha=0.8, 
                          edgecolor=COLORS['white'], linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y - 0.03, feature_type, ha='center', va='center', 
               fontweight='normal', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['white'], 
                        edgecolor=color, alpha=0.8))
    
    ax.set_title('Feature Engineering Pipeline', fontweight='normal', pad=30, 
                fontsize=20, color=COLORS['dark_purple'])
    
    plt.tight_layout()
    plt.savefig(output_path / '06_feature_pipeline.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Better graphs saved to: {output_path}")
    print("Files created:")
    print("  01_dataset_overview.png - Dataset composition")
    print("  02_model_performance.png - Model performance comparison")
    print("  03_feature_importance.png - Feature importance")
    print("  04_robustness_analysis.png - Robustness analysis")
    print("  05_device_architecture.png - WESAD device architecture")
    print("  06_feature_pipeline.png - Feature engineering pipeline")

if __name__ == "__main__":
    create_better_graphs() 