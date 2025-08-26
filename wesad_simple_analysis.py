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

def analyze_data_structure(data):
    """Analyze the structure of WESAD data"""
    print("=== Data Structure Analysis ===")
    print(f"Data type: {type(data)}")
    print(f"Keys: {list(data.keys())}")
    
    if 'signal' in data:
        print("\nSignal types:")
        for signal_type, signals in data['signal'].items():
            print(f"  {signal_type}: {list(signals.keys())}")
    
    if 'label' in data:
        labels = data['label']
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\nLabel distribution:")
        label_names = {0: 'Baseline', 1: 'Stress', 2: 'Amusement', 3: 'Meditation'}
        for label, count in zip(unique_labels, counts):
            name = label_names.get(label, f'Label {label}')
            print(f"  {name}: {count} samples")

def check_data_quality(data):
    """Check for missing values and data quality issues"""
    print("\n=== Data Quality Check ===")
    
    if 'signal' not in data:
        return
    
    for signal_type, signals in data['signal'].items():
        print(f"\n{signal_type.upper()} signals:")
        for signal_name, signal_data in signals.items():
            if isinstance(signal_data, np.ndarray):
                nan_count = np.isnan(signal_data).sum()
                inf_count = np.isinf(signal_data).sum()
                total_samples = len(signal_data)
                
                print(f"  {signal_name}:")
                print(f"    Shape: {signal_data.shape}")
                print(f"    NaN count: {nan_count} ({nan_count/total_samples*100:.2f}%)")
                print(f"    Inf count: {inf_count}")
                print(f"    Range: [{np.nanmin(signal_data):.4f}, {np.nanmax(signal_data):.4f}]")
                print(f"    Mean: {np.nanmean(signal_data):.4f}")
                print(f"    Std: {np.nanstd(signal_data):.4f}")

def create_basic_plots(data, subject='S2'):
    """Create basic visualization plots"""
    print(f"\n=== Creating plots for {subject} ===")
    
    if 'signal' not in data:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'WESAD Dataset - {subject}', fontsize=16)
    
    signals = data['signal']
    
    # Plot 1: ECG signal
    if 'chest' in signals and 'ECG' in signals['chest']:
        ecg = signals['chest']['ECG']
        axes[0, 0].plot(ecg[:5000], linewidth=0.5)
        axes[0, 0].set_title('ECG Signal (Chest)')
        axes[0, 0].set_ylabel('Amplitude')
    
    # Plot 2: BVP signal
    if 'wrist' in signals and 'BVP' in signals['wrist']:
        bvp = signals['wrist']['BVP']
        axes[0, 1].plot(bvp[:5000], linewidth=0.5)
        axes[0, 1].set_title('BVP Signal (Wrist)')
        axes[0, 1].set_ylabel('Amplitude')
    
    # Plot 3: Respiration
    if 'chest' in signals and 'Resp' in signals['chest']:
        resp = signals['chest']['Resp']
        axes[0, 2].plot(resp[:5000], linewidth=0.5)
        axes[0, 2].set_title('Respiration Signal')
        axes[0, 2].set_ylabel('Amplitude')
    
    # Plot 4: EDA
    if 'chest' in signals and 'EDA' in signals['chest']:
        eda = signals['chest']['EDA']
        axes[1, 0].plot(eda[:5000], linewidth=0.5)
        axes[1, 0].set_title('EDA Signal')
        axes[1, 0].set_ylabel('Amplitude')
    
    # Plot 5: Temperature
    if 'chest' in signals and 'Temp' in signals['chest']:
        temp = signals['chest']['Temp']
        axes[1, 1].plot(temp[:5000], linewidth=0.5)
        axes[1, 1].set_title('Temperature Signal')
        axes[1, 1].set_ylabel('Temperature')
    
    # Plot 6: Label distribution
    if 'label' in data:
        labels = data['label']
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_names = {0: 'Baseline', 1: 'Stress', 2: 'Amusement', 3: 'Meditation'}
        
        colors = ['blue', 'red', 'green', 'orange']
        bars = axes[1, 2].bar(unique_labels, counts, color=colors[:len(unique_labels)])
        axes[1, 2].set_title('Label Distribution')
        axes[1, 2].set_xlabel('Label')
        axes[1, 2].set_ylabel('Count')
        
        # Add labels on bars
        for bar, label in zip(bars, unique_labels):
            name = label_names.get(label, f'Label {label}')
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{name}\n({int(height)})',
                           ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'wesad_basic_plots_{subject}.png', dpi=300, bbox_inches='tight')
    plt.show()

def correlation_analysis(data, subject='S2'):
    """Analyze correlations between signals"""
    print(f"\n=== Correlation Analysis for {subject} ===")
    
    if 'signal' not in data:
        return
    
    # Collect all 1D signals
    signal_data = {}
    signals = data['signal']
    
    for signal_type in ['chest', 'wrist']:
        if signal_type in signals:
            for signal_name, signal_array in signals[signal_type].items():
                if isinstance(signal_array, np.ndarray) and len(signal_array.shape) == 1:
                    signal_data[f'{signal_type}_{signal_name}'] = signal_array
    
    if len(signal_data) < 2:
        print("Not enough signals for correlation analysis")
        return
    
    # Sample data for correlation (first 10000 samples)
    sample_size = min(10000, min(len(signal) for signal in signal_data.values()))
    signal_df = pd.DataFrame({
        name: signal[:sample_size] for name, signal in signal_data.items()
    })
    
    # Calculate correlations
    corr_matrix = signal_df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
               square=True, fmt='.3f')
    plt.title(f'Signal Correlations - {subject}')
    plt.tight_layout()
    plt.savefig(f'wesad_correlations_{subject}.png', dpi=300, bbox_inches='tight')
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
    for i, (signal1, signal2, corr_val) in enumerate(corr_pairs[:5]):
        print(f"{i+1}. {signal1} ↔ {signal2}: {corr_val:.3f}")

def main():
    """Main analysis function"""
    print("=== WESAD Dataset Analysis ===")
    
    # Load data for S2
    data = load_wesad_data('S2')
    if data is None:
        print("Could not load data")
        return
    
    # Analyze data structure
    analyze_data_structure(data)
    
    # Check data quality
    check_data_quality(data)
    
    # Create plots
    create_basic_plots(data, 'S2')
    
    # Correlation analysis
    correlation_analysis(data, 'S2')
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main() 