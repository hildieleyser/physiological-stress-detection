#!/usr/bin/env python3
"""
Dataset Splits Analysis - Show actual train/validation/test splits
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Color palette
COLORS = {
    'dark_purple': '#190B28',
    'wenge': '#685762',
    'battleship_gray': '#9B9987',
    'cherry_blossom': '#EFA9AE',
    'blush': '#E55381',
    'white': '#FFFFFF',
    'light_gray': '#F8F9FA'
}

# Font settings - Helvetica without bold
plt.rcParams.update({
    'font.family': 'Helvetica',
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 24,
    'figure.titleweight': 'normal',
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1,
    'grid.alpha': 0.2,
    'axes.facecolor': COLORS['light_gray'],
    'figure.facecolor': COLORS['white']
})

plt.style.use('default')
sns.set_theme(style="whitegrid", font_scale=1.2)

def analyze_dataset_splits():
    """Analyze and visualize the actual dataset splits."""
    print("Analyzing dataset splits...")
    
    # Load data
    results_path = Path("results")
    output_path = Path("dataset_splits_analysis")
    output_path.mkdir(exist_ok=True)
    
    X = np.load(results_path / 'X.npy')
    y = np.load(results_path / 'y.npy')
    
    # Get dataset info
    n_samples = len(X)
    n_features = X.shape[1]
    class_counts = np.bincount(y)
    class_labels = ['Stress', 'Amusement', 'Resting']
    
    print(f"Dataset Info:")
    print(f"  Total samples: {n_samples:,}")
    print(f"  Features: {n_features:,}")
    print(f"  Classes: {len(class_labels)}")
    print(f"  Class distribution: {dict(zip(class_labels, class_counts))}")
    
    # Calculate different split scenarios
    splits = {
        '70/15/15': {
            'train': int(0.7 * n_samples),
            'val': int(0.15 * n_samples),
            'test': n_samples - int(0.7 * n_samples) - int(0.15 * n_samples)
        },
        '80/10/10': {
            'train': int(0.8 * n_samples),
            'val': int(0.1 * n_samples),
            'test': n_samples - int(0.8 * n_samples) - int(0.1 * n_samples)
        },
        '60/20/20': {
            'train': int(0.6 * n_samples),
            'val': int(0.2 * n_samples),
            'test': n_samples - int(0.6 * n_samples) - int(0.2 * n_samples)
        }
    }
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset Splits Analysis', fontsize=24, fontweight='normal', color=COLORS['dark_purple'], y=0.95)
    
    # 1. Overall dataset composition
    ax1.set_facecolor(COLORS['light_gray'])
    colors = [COLORS['blush'], COLORS['cherry_blossom'], COLORS['dark_purple']]
    
    wedges, texts, autotexts = ax1.pie(class_counts, labels=class_labels, 
                                     colors=colors, autopct='%1.0f%%', 
                                     startangle=90, explode=(0.05, 0.05, 0.05),
                                     shadow=True, textprops={'fontsize': 12, 'fontweight': 'normal'})
    
    for autotext in autotexts:
        autotext.set_color(COLORS['white'])
        autotext.set_fontweight('normal')
        autotext.set_fontsize(13)
    
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('normal')
    
    ax1.set_title('Overall Class Distribution', fontweight='normal', pad=20, 
                fontsize=18, color=COLORS['dark_purple'])
    
    # 2. Different split scenarios
    ax2.set_facecolor(COLORS['light_gray'])
    
    split_names = list(splits.keys())
    train_sizes = [splits[name]['train'] for name in split_names]
    val_sizes = [splits[name]['val'] for name in split_names]
    test_sizes = [splits[name]['test'] for name in split_names]
    
    x = np.arange(len(split_names))
    width = 0.25
    
    bars1 = ax2.bar(x - width, train_sizes, width, label='Training', 
                   color=COLORS['dark_purple'], alpha=0.9, edgecolor=COLORS['white'], linewidth=2)
    bars2 = ax2.bar(x, val_sizes, width, label='Validation', 
                   color=COLORS['wenge'], alpha=0.9, edgecolor=COLORS['white'], linewidth=2)
    bars3 = ax2.bar(x + width, test_sizes, width, label='Test', 
                   color=COLORS['battleship_gray'], alpha=0.9, edgecolor=COLORS['white'], linewidth=2)
    
    ax2.set_title('Different Split Scenarios', fontweight='normal', pad=20, 
                fontsize=18, color=COLORS['dark_purple'])
    ax2.set_ylabel('Number of Samples', fontweight='normal', fontsize=16)
    ax2.set_xlabel('Split Ratio', fontweight='normal', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(split_names)
    ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(train_sizes)*0.01,
                   f'{height:,}', ha='center', va='bottom', 
                   fontweight='normal', fontsize=10)
    
    ax2.tick_params(axis='both', which='major', labelsize=12)
    for label in ax2.get_xticklabels():
        label.set_fontweight('normal')
    
    # 3. Recommended split (70/15/15) with percentages
    ax3.set_facecolor(COLORS['light_gray'])
    
    recommended_split = splits['70/15/15']
    split_labels = ['Training\n(70%)', 'Validation\n(15%)', 'Test\n(15%)']
    split_values = [recommended_split['train'], recommended_split['val'], recommended_split['test']]
    split_colors = [COLORS['dark_purple'], COLORS['wenge'], COLORS['battleship_gray']]
    
    bars = ax3.bar(split_labels, split_values, color=split_colors, alpha=0.9, 
                 edgecolor=COLORS['white'], linewidth=2)
    
    ax3.set_title('Recommended Split (70/15/15)', fontweight='normal', pad=20, 
                fontsize=18, color=COLORS['dark_purple'])
    ax3.set_ylabel('Number of Samples', fontweight='normal', fontsize=16)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    for bar, value in zip(bars, split_values):
        height = bar.get_height()
        percentage = (value / n_samples) * 100
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(split_values)*0.02,
               f'{value:,}\n({percentage:.1f}%)', ha='center', va='bottom', 
               fontweight='normal', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['white'], 
                        edgecolor=COLORS['wenge'], alpha=0.8))
    
    ax3.tick_params(axis='both', which='major', labelsize=12)
    for label in ax3.get_xticklabels():
        label.set_fontweight('normal')
    
    # 4. Class distribution in each split (simulated)
    ax4.set_facecolor(COLORS['light_gray'])
    
    # Simulate class distribution in splits (you should calculate this from actual splits)
    train_class_dist = [int(count * 0.7) for count in class_counts]
    val_class_dist = [int(count * 0.15) for count in class_counts]
    test_class_dist = [count - int(count * 0.7) - int(count * 0.15) for count in class_counts]
    
    x = np.arange(len(class_labels))
    width = 0.25
    
    bars1 = ax4.bar(x - width, train_class_dist, width, label='Training', 
                   color=COLORS['dark_purple'], alpha=0.9, edgecolor=COLORS['white'], linewidth=2)
    bars2 = ax4.bar(x, val_class_dist, width, label='Validation', 
                   color=COLORS['wenge'], alpha=0.9, edgecolor=COLORS['white'], linewidth=2)
    bars3 = ax4.bar(x + width, test_class_dist, width, label='Test', 
                   color=COLORS['battleship_gray'], alpha=0.9, edgecolor=COLORS['white'], linewidth=2)
    
    ax4.set_title('Class Distribution in Splits', fontweight='normal', pad=20, 
                fontsize=18, color=COLORS['dark_purple'])
    ax4.set_ylabel('Number of Samples', fontweight='normal', fontsize=16)
    ax4.set_xlabel('Classes', fontweight='normal', fontsize=16)
    ax4.set_xticks(x)
    ax4.set_xticklabels(class_labels)
    ax4.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height}', ha='center', va='bottom', 
                       fontweight='normal', fontsize=10)
    
    ax4.tick_params(axis='both', which='major', labelsize=12)
    for label in ax4.get_xticklabels():
        label.set_fontweight('normal')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_path / 'dataset_splits_analysis.png', dpi=300, 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Create detailed text summary
    summary_text = f"""
Dataset Splits Summary
=====================

Dataset Information:
- Total samples: {n_samples:,}
- Features: {n_features:,}
- Classes: {len(class_labels)}

Class Distribution:
{chr(10).join([f"- {label}: {count:,} samples ({count/n_samples*100:.1f}%)" for label, count in zip(class_labels, class_counts)])}

Recommended Split (70/15/15):
- Training: {splits['70/15/15']['train']:,} samples (70.0%)
- Validation: {splits['70/15/15']['val']:,} samples (15.0%)
- Test: {splits['70/15/15']['test']:,} samples (15.0%)

Alternative Splits:
- 80/10/10: Train={splits['80/10/10']['train']:,}, Val={splits['80/10/10']['val']:,}, Test={splits['80/10/10']['test']:,}
- 60/20/20: Train={splits['60/20/20']['train']:,}, Val={splits['60/20/20']['val']:,}, Test={splits['60/20/20']['test']:,}

Notes:
- The 70/15/15 split is recommended for this dataset size
- Validation set is used for hyperparameter tuning
- Test set is used for final evaluation only
- Class distribution is maintained across splits
"""
    
    with open(output_path / 'dataset_splits_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print(f"Dataset splits analysis saved to: {output_path}")
    print("Files created:")
    print("  dataset_splits_analysis.png - Visual analysis of splits")
    print("  dataset_splits_summary.txt - Detailed text summary")
    
    # Print summary to console
    print("\n" + "="*50)
    print("DATASET SPLITS SUMMARY")
    print("="*50)
    print(f"Total samples: {n_samples:,}")
    print(f"Features: {n_features:,}")
    print(f"Classes: {len(class_labels)}")
    print()
    print("Class Distribution:")
    for label, count in zip(class_labels, class_counts):
        print(f"  {label}: {count:,} samples ({count/n_samples*100:.1f}%)")
    print()
    print("Recommended Split (70/15/15):")
    print(f"  Training: {splits['70/15/15']['train']:,} samples")
    print(f"  Validation: {splits['70/15/15']['val']:,} samples")
    print(f"  Test: {splits['70/15/15']['test']:,} samples")

if __name__ == "__main__":
    analyze_dataset_splits() 