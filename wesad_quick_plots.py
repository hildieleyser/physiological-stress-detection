import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def quick_plots():
    """Generate quick plots using minimal sampling"""
    print("=== Quick WESAD Plots ===")
    
    # Load just one subject for quick analysis
    data = load_subject_data('S2')
    if not data:
        print("Could not load data")
        return
    
    # Create quick overview plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('WESAD Dataset Quick Overview - S2', fontsize=16)
    
    signals = data.get('signal', {})
    
    # Plot 1: ECG (sample first 5000 points)
    if 'chest' in signals and 'ECG' in signals['chest']:
        ecg = signals['chest']['ECG'][:5000].flatten()
        axes[0, 0].plot(ecg, linewidth=0.5)
        axes[0, 0].set_title('ECG Signal (First 5000 samples)')
        axes[0, 0].set_ylabel('Amplitude')
    
    # Plot 2: BVP
    if 'wrist' in signals and 'BVP' in signals['wrist']:
        bvp = signals['wrist']['BVP'][:5000].flatten()
        axes[0, 1].plot(bvp, linewidth=0.5)
        axes[0, 1].set_title('BVP Signal (First 5000 samples)')
        axes[0, 1].set_ylabel('Amplitude')
    
    # Plot 3: Respiration
    if 'chest' in signals and 'Resp' in signals['chest']:
        resp = signals['chest']['Resp'][:5000].flatten()
        axes[0, 2].plot(resp, linewidth=0.5)
        axes[0, 2].set_title('Respiration Signal (First 5000 samples)')
        axes[0, 2].set_ylabel('Amplitude')
    
    # Plot 4: EDA
    if 'chest' in signals and 'EDA' in signals['chest']:
        eda = signals['chest']['EDA'][:5000].flatten()
        axes[1, 0].plot(eda, linewidth=0.5)
        axes[1, 0].set_title('EDA Signal (First 5000 samples)')
        axes[1, 0].set_ylabel('Amplitude')
    
    # Plot 5: Temperature
    if 'chest' in signals and 'Temp' in signals['chest']:
        temp = signals['chest']['Temp'][:5000].flatten()
        axes[1, 1].plot(temp, linewidth=0.5)
        axes[1, 1].set_title('Temperature Signal (First 5000 samples)')
        axes[1, 1].set_ylabel('Temperature (°C)')
    
    # Plot 6: Label distribution
    if 'label' in data:
        labels = data['label']
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_names = {0: 'Baseline', 1: 'Stress', 2: 'Amusement', 3: 'Meditation'}
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
        bars = axes[1, 2].bar(unique_labels, counts, color=colors[:len(unique_labels)])
        axes[1, 2].set_title('Label Distribution')
        axes[1, 2].set_xlabel('Label')
        axes[1, 2].set_ylabel('Count')
        
        # Add labels on bars
        for bar, label in zip(bars, unique_labels):
            name = label_names.get(label, f'Label {label}')
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{name}\n({int(height):,})',
                           ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('wesad_quick_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create histogram plots
    create_quick_histograms(data)
    
    # Create correlation plot
    create_quick_correlations(data)

def load_subject_data(subject='S2', data_path="WESAD"):
    """Load data for a specific subject"""
    subject_path = Path(data_path) / subject
    pkl_file = subject_path / f"{subject}.pkl"
    
    if pkl_file.exists():
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data
    else:
        print(f"No data found for {subject}")
        return None

def create_quick_histograms(data):
    """Create quick histograms"""
    print("Creating histograms...")
    
    signals = data.get('signal', {})
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('WESAD Signal Histograms - S2', fontsize=16)
    
    # Collect data for histograms (sample 10000 points each)
    signal_data = {}
    
    for signal_type in ['chest', 'wrist']:
        if signal_type in signals:
            for signal_name, signal_array in signals[signal_type].items():
                if isinstance(signal_array, np.ndarray):
                    # Sample data
                    if len(signal_array) > 10000:
                        indices = np.random.choice(len(signal_array), 10000, replace=False)
                        sampled_data = signal_array[indices]
                    else:
                        sampled_data = signal_array.flatten()
                    
                    signal_data[f'{signal_type}_{signal_name}'] = sampled_data
    
    # Create histograms
    positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    
    for idx, (signal_name, values) in enumerate(list(signal_data.items())[:6]):
        row, col = positions[idx]
        
        axes[row, col].hist(values, bins=30, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'{signal_name}')
        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Frequency')
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        axes[row, col].text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                           transform=axes[row, col].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('wesad_quick_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_quick_correlations(data):
    """Create quick correlation plot"""
    print("Creating correlations...")
    
    signals = data.get('signal', {})
    
    # Collect 1D signals
    signal_data = {}
    
    for signal_type in ['chest', 'wrist']:
        if signal_type in signals:
            for signal_name, signal_array in signals[signal_type].items():
                if isinstance(signal_array, np.ndarray) and len(signal_array.shape) == 1:
                    # Sample data
                    if len(signal_array) > 10000:
                        indices = np.random.choice(len(signal_array), 10000, replace=False)
                        sampled_data = signal_array[indices]
                    else:
                        sampled_data = signal_array
                    
                    signal_data[f'{signal_type}_{signal_name}'] = sampled_data
                elif isinstance(signal_array, np.ndarray) and len(signal_array.shape) > 1:
                    # Use first dimension
                    if len(signal_array) > 10000:
                        indices = np.random.choice(len(signal_array), 10000, replace=False)
                        sampled_data = signal_array[indices, 0]
                    else:
                        sampled_data = signal_array[:, 0]
                    
                    signal_data[f'{signal_type}_{signal_name}'] = sampled_data
    
    if len(signal_data) > 1:
        # Create correlation matrix
        signal_df = pd.DataFrame(signal_data)
        corr_matrix = signal_df.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('WESAD Signal Correlations - S2 (Sampled Data)')
        plt.tight_layout()
        plt.savefig('wesad_quick_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function"""
    print("=== Quick WESAD Plots ===")
    quick_plots()
    print("=== Quick Plots Complete ===")
    print("Generated files:")
    print("- wesad_quick_overview.png")
    print("- wesad_quick_histograms.png") 
    print("- wesad_quick_correlations.png")

if __name__ == "__main__":
    main() 