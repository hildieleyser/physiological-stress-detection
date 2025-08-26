import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_wesad_data(subject='S2', data_path="WESAD"):
    """Load WESAD data for a specific subject"""
    subject_path = Path(data_path) / subject
    pkl_file = subject_path / f"{subject}.pkl"
    
    if pkl_file.exists():
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data
    else:
        print(f"No data found for {subject}")
        return None

def analyze_dataset_structure(data):
    """Analyze the complete dataset structure"""
    print("=== WESAD Dataset Structure Analysis ===")
    print(f"Data type: {type(data)}")
    print(f"Keys: {list(data.keys())}")
    
    if 'signal' in data:
        signals = data['signal']
        print(f"\nSignal types:")
        for signal_type, signal_dict in signals.items():
            print(f"  {signal_type.upper()}: {list(signal_dict.keys())}")
    
    if 'label' in data:
        labels = data['label']
        print(f"\nLabel information:")
        print(f"  Total samples: {len(labels):,}")
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_names = {0: 'Baseline', 1: 'Stress', 2: 'Amusement', 3: 'Meditation'}
        
        print(f"  Label distribution:")
        for label, count in zip(unique_labels, counts):
            name = label_names.get(label, f'Label {label}')
            percentage = (count / len(labels)) * 100
            print(f"    {name}: {count:,} samples ({percentage:.1f}%)")

def analyze_signal_quality(data):
    """Analyze signal quality using full dataset"""
    print("\n=== Signal Quality Analysis (Full Dataset) ===")
    
    if 'signal' not in data:
        return
    
    signals = data['signal']
    
    for signal_type in ['chest', 'wrist']:
        if signal_type in signals:
            print(f"\n{signal_type.upper()} SIGNALS:")
            print("=" * 50)
            
            for signal_name, signal_data in signals[signal_type].items():
                if isinstance(signal_data, np.ndarray):
                    print(f"\n{signal_name}:")
                    print(f"  Shape: {signal_data.shape}")
                    print(f"  Total samples: {len(signal_data):,}")
                    
                    # Handle multi-dimensional signals
                    if len(signal_data.shape) > 1:
                        for dim in range(signal_data.shape[1]):
                            dim_data = signal_data[:, dim]
                            nan_count = np.isnan(dim_data).sum()
                            inf_count = np.isinf(dim_data).sum()
                            
                            print(f"    Dimension {dim}:")
                            print(f"      NaN: {nan_count} ({nan_count/len(dim_data)*100:.3f}%)")
                            print(f"      Inf: {inf_count} ({inf_count/len(dim_data)*100:.3f}%)")
                            print(f"      Range: [{np.min(dim_data):.4f}, {np.max(dim_data):.4f}]")
                            print(f"      Mean: {np.mean(dim_data):.4f}")
                            print(f"      Std: {np.std(dim_data):.4f}")
                    else:
                        nan_count = np.isnan(signal_data).sum()
                        inf_count = np.isinf(signal_data).sum()
                        
                        print(f"  NaN: {nan_count} ({nan_count/len(signal_data)*100:.3f}%)")
                        print(f"  Inf: {inf_count} ({inf_count/len(signal_data)*100:.3f}%)")
                        print(f"  Range: [{np.min(signal_data):.4f}, {np.max(signal_data):.4f}]")
                        print(f"  Mean: {np.mean(signal_data):.4f}")
                        print(f"  Std: {np.std(signal_data):.4f}")

def create_full_dataset_plots(data, subject='S2'):
    """Create plots using the full dataset"""
    print(f"\n=== Creating full dataset plots for {subject} ===")
    
    if 'signal' not in data:
        return
    
    signals = data['signal']
    
    # Create comprehensive signal plots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle(f'WESAD Dataset - {subject} - Full Dataset Analysis', fontsize=16)
    
    # Plot 1: ECG signal (full dataset)
    if 'chest' in signals and 'ECG' in signals['chest']:
        ecg = signals['chest']['ECG'].flatten()
        axes[0, 0].plot(ecg, linewidth=0.2, alpha=0.7)
        axes[0, 0].set_title('ECG Signal (Chest) - Full Dataset')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_xlabel('Sample')
    
    # Plot 2: BVP signal (full dataset)
    if 'wrist' in signals and 'BVP' in signals['wrist']:
        bvp = signals['wrist']['BVP'].flatten()
        axes[0, 1].plot(bvp, linewidth=0.2, alpha=0.7)
        axes[0, 1].set_title('BVP Signal (Wrist) - Full Dataset')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].set_xlabel('Sample')
    
    # Plot 3: Respiration signal
    if 'chest' in signals and 'Resp' in signals['chest']:
        resp = signals['chest']['Resp'].flatten()
        axes[1, 0].plot(resp, linewidth=0.2, alpha=0.7)
        axes[1, 0].set_title('Respiration Signal - Full Dataset')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].set_xlabel('Sample')
    
    # Plot 4: EDA signal
    if 'chest' in signals and 'EDA' in signals['chest']:
        eda = signals['chest']['EDA'].flatten()
        axes[1, 1].plot(eda, linewidth=0.2, alpha=0.7)
        axes[1, 1].set_title('EDA Signal - Full Dataset')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].set_xlabel('Sample')
    
    # Plot 5: Temperature signal
    if 'chest' in signals and 'Temp' in signals['chest']:
        temp = signals['chest']['Temp'].flatten()
        axes[2, 0].plot(temp, linewidth=0.2, alpha=0.7)
        axes[2, 0].set_title('Temperature Signal - Full Dataset')
        axes[2, 0].set_ylabel('Temperature')
        axes[2, 0].set_xlabel('Sample')
    
    # Plot 6: Label distribution
    if 'label' in data:
        labels = data['label']
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_names = {0: 'Baseline', 1: 'Stress', 2: 'Amusement', 3: 'Meditation'}
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
        bars = axes[2, 1].bar(unique_labels, counts, color=colors[:len(unique_labels)])
        axes[2, 1].set_title('Label Distribution')
        axes[2, 1].set_xlabel('Label')
        axes[2, 1].set_ylabel('Count')
        
        # Add labels on bars
        for bar, label in zip(bars, unique_labels):
            name = label_names.get(label, f'Label {label}')
            height = bar.get_height()
            axes[2, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{name}\n({int(height):,})',
                           ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'wesad_full_dataset_{subject}.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_correlations_full(data, subject='S2'):
    """Analyze correlations using full dataset"""
    print(f"\n=== Full Dataset Correlation Analysis for {subject} ===")
    
    if 'signal' not in data:
        return
    
    signals = data['signal']
    
    # Collect all 1D signals
    signal_data = {}
    
    for signal_type in ['chest', 'wrist']:
        if signal_type in signals:
            for signal_name, signal_array in signals[signal_type].items():
                if isinstance(signal_array, np.ndarray) and len(signal_array.shape) == 1:
                    signal_data[f'{signal_type}_{signal_name}'] = signal_array
                elif isinstance(signal_array, np.ndarray) and len(signal_array.shape) > 1:
                    # For multi-dimensional signals, use the first dimension
                    signal_data[f'{signal_type}_{signal_name}'] = signal_array[:, 0]
    
    if len(signal_data) < 2:
        print("Not enough signals for correlation analysis")
        return
    
    print(f"Analyzing correlations between {len(signal_data)} signals:")
    for name, signal in signal_data.items():
        print(f"  - {name}: {len(signal):,} samples")
    
    # Find the minimum length among all signals
    min_length = min(len(signal) for signal in signal_data.values())
    print(f"\nUsing minimum common length: {min_length:,} samples")
    
    # Truncate all signals to the minimum length
    signal_df = pd.DataFrame({
        name: signal[:min_length] for name, signal in signal_data.items()
    })
    
    # Calculate correlations
    corr_matrix = signal_df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
               square=True, fmt='.3f')
    plt.title(f'Signal Correlations - {subject} (Full Dataset)')
    plt.tight_layout()
    plt.savefig(f'wesad_full_correlations_{subject}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print top correlations
    print("\nTop correlations:")
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if not np.isnan(corr_val):
                corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for i, (signal1, signal2, corr_val) in enumerate(corr_pairs[:10]):
        print(f"{i+1}. {signal1} ↔ {signal2}: {corr_val:.3f}")
    
    # Print signal length information
    print(f"\nSignal length information:")
    for name, signal in signal_data.items():
        print(f"  {name}: {len(signal):,} samples")

def main():
    """Main analysis function"""
    print("=== WESAD Dataset Full Analysis ===")
    
    # Load data for S2
    data = load_wesad_data('S2')
    if data is None:
        print("Could not load data")
        return
    
    # Analyze dataset structure
    analyze_dataset_structure(data)
    
    # Analyze signal quality
    analyze_signal_quality(data)
    
    # Create full dataset plots
    create_full_dataset_plots(data, 'S2')
    
    # Analyze correlations
    analyze_correlations_full(data, 'S2')
    
    print("\n=== Full Analysis Complete ===")
    print("Generated plots:")
    print("- wesad_full_dataset_S2.png")
    print("- wesad_full_correlations_S2.png")

if __name__ == "__main__":
    main() 