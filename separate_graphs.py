#!/usr/bin/env python3
"""
Separate Graphs for Presentation Slide Assembly
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

# Updated font settings - specifically using Helvetica
plt.rcParams.update({
    'font.family': 'Helvetica',
    'font.size': 16,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 32,
    'figure.titleweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'lines.linewidth': 3,
    'axes.linewidth': 2,
    'grid.linewidth': 1.5,
    'grid.alpha': 0.3,
    'axes.facecolor': COLORS['light_gray'],
    'figure.facecolor': COLORS['white']
})

# Set modern style
plt.style.use('default')
sns.set_theme(style="whitegrid", font_scale=1.4)

def create_separate_graphs():
    """Create separate PNG files for each graph."""
    print("Creating separate graph files...")
    
    # Load data
    results_path = Path("results")
    output_path = Path("separate_graphs")
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
        'feature_39': 'Temperature Stability\n(TEMP Variance)',
        'feature_26': 'Skin Conductance\n(EDA Response)',
        'feature_23': 'Chest Movement\n(ACC Statistics)',
        'feature_34': 'Frequency Domain\n(FFT Features)',
        'feature_35': 'Temporal Patterns\n(Trend Analysis)'
    }
    
    # 1. Dataset Overview
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor(COLORS['light_gray'])
    
    class_labels = ['Stress', 'Amusement', 'Resting']
    colors = [COLORS['blush'], COLORS['cherry_blossom'], COLORS['dark_purple']]
    
    wedges, texts, autotexts = ax.pie(class_counts, labels=class_labels, 
                                     colors=colors, autopct='%1.0f%%', 
                                     startangle=90, explode=(0.08, 0.08, 0.08),
                                     shadow=True, textprops={'fontsize': 18, 'fontweight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color(COLORS['white'])
        autotext.set_fontweight('bold')
        autotext.set_fontsize(20)
    
    for text in texts:
        text.set_fontsize(18)
        text.set_fontweight('bold')
    
    ax.set_title('Dataset Composition', fontweight='bold', pad=40, 
                fontsize=28, color=COLORS['dark_purple'])
    
    ax.text(0.5, -1.8, f'Total: {data_shape[0]:,} samples\n{data_shape[1]:,} features', 
           ha='center', va='center', fontweight='bold', fontsize=18,
           bbox=dict(boxstyle="round,pad=0.8", facecolor=COLORS['white'], 
                    edgecolor=COLORS['wenge'], linewidth=3, alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path / '01_dataset_overview.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # 2. Model Performance
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor(COLORS['light_gray'])
    
    models = ['Logistic\nRegression', 'Random\nForest', 'XGBoost']
    accuracies = [cv_results['logistic_regression']['mean_accuracy'],
                 cv_results['random_forest']['mean_accuracy'],
                 cv_results['xgboost']['mean_accuracy']]
    stds = [cv_results['logistic_regression']['std_accuracy'],
            cv_results['random_forest']['std_accuracy'],
            cv_results['xgboost']['std_accuracy']]
    
    bars = ax.bar(models, accuracies, yerr=stds, capsize=15, 
                 color=[COLORS['dark_purple'], COLORS['wenge'], COLORS['battleship_gray']], 
                 alpha=0.9, edgecolor=COLORS['white'], linewidth=3)
    
    ax.set_title('Model Performance (Cross-Validation)', fontweight='bold', pad=40, 
                fontsize=28, color=COLORS['dark_purple'])
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=22)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    for bar, acc, std in zip(bars, accuracies, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
               f'{acc:.1%}\n±{std:.1%}', ha='center', va='bottom', 
               fontweight='bold', fontsize=16,
               bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['white'], 
                        edgecolor=COLORS['wenge'], alpha=0.8))
    
    best_idx = np.argmax(accuracies)
    bars[best_idx].set_edgecolor(COLORS['blush'])
    bars[best_idx].set_linewidth(5)
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(output_path / '02_model_performance.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # 3. Feature Importance
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
                  color=colors, alpha=0.9, edgecolor=COLORS['white'], linewidth=3)
    
    ax.set_title('Top 5 Physiological Features (SHAP Importance)', fontweight='bold', pad=40, 
                fontsize=28, color=COLORS['dark_purple'])
    ax.set_xlabel('Importance Score', fontweight='bold', fontsize=22)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    for i, (bar, importance) in enumerate(zip(bars, importances)):
        width = bar.get_width()
        ax.text(width + max(importances)*0.01, bar.get_y() + bar.get_height()/2, 
               f'{importance:.3f}', ha='left', va='center', fontweight='bold', 
               fontsize=16,
               bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS['white'], 
                        edgecolor=colors[i], alpha=0.8))
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(output_path / '03_feature_importance.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # 4. Robustness Analysis
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor(COLORS['light_gray'])
    
    noise_levels = robustness_results['logistic_regression']['noise_levels']
    colors = [COLORS['dark_purple'], COLORS['wenge'], COLORS['battleship_gray']]
    model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
    
    for i, model in enumerate(robustness_results):
        accuracies = robustness_results[model]['accuracies']
        ax.plot(noise_levels, accuracies, 'o-', linewidth=4, 
               label=model_names[i], color=colors[i], markersize=12, 
               markerfacecolor=COLORS['white'], markeredgecolor=colors[i], 
               markeredgewidth=3, alpha=0.9)
    
    ax.set_title('Model Robustness (Noise Sensitivity)', fontweight='bold', pad=40, 
                fontsize=28, color=COLORS['dark_purple'])
    ax.set_xlabel('Noise Level (%)', fontweight='bold', fontsize=22)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=22)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=16, 
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
    
    ax.text(0.02, 0.98, ranking_text, transform=ax.transAxes, fontsize=14,
           verticalalignment='top', fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.8", facecolor=COLORS['white'], 
                    edgecolor=COLORS['wenge'], linewidth=3, alpha=0.9))
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(output_path / '04_robustness_analysis.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # 5. WESAD Device Architecture
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    ax.set_facecolor(COLORS['light_gray'])
    
    # Title with clear explanation
    ax.set_title('WESAD Multi-Device Architecture\n(Chest + Wrist Sensors)', fontweight='bold', pad=40, 
                fontsize=28, color=COLORS['dark_purple'])
    
    # Device boxes with clear labels
    chest_rect = plt.Rectangle((0.1, 0.4), 0.35, 0.4, fill=False, 
                             edgecolor=COLORS['dark_purple'], linewidth=5)
    wrist_rect = plt.Rectangle((0.55, 0.4), 0.35, 0.4, fill=False, 
                             edgecolor=COLORS['wenge'], linewidth=5)
    ax.add_patch(chest_rect)
    ax.add_patch(wrist_rect)
    
    ax.text(0.275, 0.85, 'Chest Device (RespiBAN)\n700 Hz Sampling', ha='center', va='center', 
           fontweight='bold', fontsize=18, color=COLORS['dark_purple'], 
           bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['white'], 
                    edgecolor=COLORS['dark_purple'], alpha=0.8))
    ax.text(0.725, 0.85, 'Wrist Device (E4)\n4-64 Hz Sampling', ha='center', va='center', 
           fontweight='bold', fontsize=18, color=COLORS['wenge'], 
           bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['white'], 
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
        y = 0.55 - (i // 3) * 0.15
        signal_rect = plt.Rectangle((x, y), 0.08, 0.12, 
                                  facecolor=color, alpha=0.8, 
                                  edgecolor=COLORS['white'], linewidth=3)
        ax.add_patch(signal_rect)
        ax.text(x + 0.04, y + 0.06, signal, ha='center', va='center', 
               fontweight='bold', fontsize=12, color=COLORS['white'])
    
    # Wrist signals
    for i, (signal, color) in enumerate(zip(wrist_signals, wrist_colors)):
        x = 0.57 + i * 0.1
        y = 0.55
        signal_rect = plt.Rectangle((x, y), 0.08, 0.12, 
                                  facecolor=color, alpha=0.8, 
                                  edgecolor=COLORS['white'], linewidth=3)
        ax.add_patch(signal_rect)
        ax.text(x + 0.04, y + 0.06, signal, ha='center', va='center', 
               fontweight='bold', fontsize=12, color=COLORS['white'])
    
    # Connection arrows showing synchronization
    for i in range(3):
        x1 = 0.45
        y1 = 0.55 - i * 0.15
        x2 = 0.55
        y2 = 0.55
        ax.plot([x1, x2], [y1, y2], color=COLORS['battleship_gray'], 
               linewidth=4, alpha=0.6, linestyle='--')
    
    # Explanation text
    ax.text(0.5, 0.25, 'Synchronized via Double-Tap Gesture\nMulti-modal stress detection across body locations', 
           ha='center', va='center', fontweight='bold', fontsize=16,
           bbox=dict(boxstyle="round,pad=0.8", facecolor=COLORS['white'], 
                    edgecolor=COLORS['battleship_gray'], linewidth=3, alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path / '05_device_architecture.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # 6. Feature Engineering Pipeline
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')
    ax.set_facecolor(COLORS['light_gray'])
    
    raw_rect = plt.Rectangle((0.05, 0.6), 0.15, 0.25, 
                           facecolor=COLORS['dark_purple'], alpha=0.3, 
                           edgecolor=COLORS['dark_purple'], linewidth=3)
    ax.add_patch(raw_rect)
    ax.text(0.125, 0.725, 'Raw Signals\n(ACC, ECG, EDA,\nEMG, RESP, TEMP)', 
           ha='center', va='center', fontweight='bold', fontsize=16, 
           bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['white'], 
                    edgecolor=COLORS['dark_purple'], alpha=0.8))
    
    ax.arrow(0.22, 0.725, 0.08, 0, head_width=0.03, head_length=0.015, 
            fc=COLORS['wenge'], ec=COLORS['wenge'], linewidth=4)
    
    feature_rect = plt.Rectangle((0.32, 0.6), 0.15, 0.25, 
                               facecolor=COLORS['wenge'], alpha=0.3, 
                               edgecolor=COLORS['wenge'], linewidth=3)
    ax.add_patch(feature_rect)
    ax.text(0.395, 0.725, 'Feature Extraction\n(Statistical Measures,\nTime Windows)', 
           ha='center', va='center', fontweight='bold', fontsize=16, 
           bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['white'], 
                    edgecolor=COLORS['wenge'], alpha=0.8))
    
    ax.arrow(0.49, 0.725, 0.08, 0, head_width=0.03, head_length=0.015, 
            fc=COLORS['wenge'], ec=COLORS['wenge'], linewidth=4)
    
    model_rect = plt.Rectangle((0.59, 0.6), 0.15, 0.25, 
                             facecolor=COLORS['battleship_gray'], alpha=0.3, 
                             edgecolor=COLORS['battleship_gray'], linewidth=3)
    ax.add_patch(model_rect)
    ax.text(0.665, 0.725, 'Model Training\n(Cross-Validation,\nHyperparameter Tuning)', 
           ha='center', va='center', fontweight='bold', fontsize=16, 
           bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['white'], 
                    edgecolor=COLORS['battleship_gray'], alpha=0.8))
    
    ax.arrow(0.76, 0.725, 0.08, 0, head_width=0.03, head_length=0.015, 
            fc=COLORS['wenge'], ec=COLORS['wenge'], linewidth=4)
    
    result_rect = plt.Rectangle((0.86, 0.6), 0.1, 0.25, 
                              facecolor=COLORS['blush'], alpha=0.3, 
                              edgecolor=COLORS['blush'], linewidth=3)
    ax.add_patch(result_rect)
    ax.text(0.91, 0.725, 'Stress Detection\nResults', 
           ha='center', va='center', fontweight='bold', fontsize=16, 
           bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['white'], 
                    edgecolor=COLORS['blush'], alpha=0.8))
    
    feature_types = ['Time\nDomain', 'Frequency\nDomain', 'Cross-\nModal', 'Temporal\nPatterns', 'Quality\nMetrics']
    feature_colors = [COLORS['blush'], COLORS['cherry_blossom'], COLORS['dark_purple'], 
                     COLORS['wenge'], COLORS['battleship_gray']]
    
    for i, (feature_type, color) in enumerate(zip(feature_types, feature_colors)):
        x = 0.32 + (i % 3) * 0.04
        y = 0.35 - (i // 3) * 0.2
        circle = plt.Circle((x, y), 0.02, facecolor=color, alpha=0.8, 
                          edgecolor=COLORS['white'], linewidth=3)
        ax.add_patch(circle)
        ax.text(x, y - 0.04, feature_type, ha='center', va='center', 
               fontweight='bold', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS['white'], 
                        edgecolor=color, alpha=0.8))
    
    ax.set_title('Feature Engineering Pipeline', fontweight='bold', pad=40, 
                fontsize=28, color=COLORS['dark_purple'])
    
    plt.tight_layout()
    plt.savefig(output_path / '06_feature_pipeline.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"All separate graphs saved to: {output_path}")
    print("Files created:")
    print("  01_dataset_overview.png")
    print("  02_model_performance.png") 
    print("  03_feature_importance.png")
    print("  04_robustness_analysis.png")
    print("  05_device_architecture.png")
    print("  06_feature_pipeline.png")

if __name__ == "__main__":
    create_separate_graphs() 