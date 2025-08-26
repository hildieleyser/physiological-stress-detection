import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_subject_data(subject, data_path="WESAD"):
    """Load data for a specific subject efficiently"""
    subject_path = Path(data_path) / subject
    pkl_file = subject_path / f"{subject}.pkl"
    
    if pkl_file.exists():
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            return data
        except Exception as e:
            print(f"Error loading {subject}: {e}")
            return None
    return None

def explain_signals():
    """Explain what each signal represents"""
    print("=== Signal Explanations ===")
    
    signals_info = {
        'ECG': 'Electrocardiogram - Heart electrical activity (chest)',
        'BVP': 'Blood Volume Pulse - Blood flow changes (wrist)',
        'Resp': 'Respiration - Breathing rate and depth (chest)',
        'EDA': 'Electrodermal Activity - Skin conductance (chest/wrist)',
        'Temp': 'Body Temperature - Skin temperature (chest/wrist)',
        'EMG': 'Electromyography - Muscle electrical activity (chest)',
        'ACC': 'Accelerometer - Movement and orientation (chest/wrist)'
    }
    
    for signal, description in signals_info.items():
        print(f"{signal}: {description}")

def analyze_data_structure_efficient():
    """Analyze data structure efficiently"""
    print("\n=== Data Structure Analysis ===")
    
    data_path = Path("WESAD")
    subjects = []
    
    for item in data_path.iterdir():
        if item.is_dir() and item.name.startswith('S'):
            subjects.append(item.name)
    
    print(f"Number of subjects: {len(subjects)}")
    print(f"Subject IDs: {sorted(subjects)}")
    
    # Analyze first subject in detail
    if subjects:
        sample_subject = subjects[0]
        data = load_subject_data(sample_subject)
        
        if data and 'signal' in data:
            print(f"\nSample data structure ({sample_subject}):")
            for signal_type in ['chest', 'wrist']:
                if signal_type in data['signal']:
                    print(f"  {signal_type.upper()}:")
                    for signal_name, signal_data in data['signal'][signal_type].items():
                        if isinstance(signal_data, np.ndarray):
                            print(f"    {signal_name}: {signal_data.shape}")
            
            if 'label' in data:
                labels = data['label']
                unique_labels, counts = np.unique(labels, return_counts=True)
                print(f"\nLabel distribution:")
                label_names = {0: 'Baseline', 1: 'Stress', 2: 'Amusement', 3: 'Meditation'}
                for label, count in zip(unique_labels, counts):
                    name = label_names.get(label, f'Label {label}')
                    print(f"  {name}: {count:,} samples")

def calculate_efficient_statistics():
    """Calculate statistics efficiently using sampling"""
    print("\n=== Efficient Statistical Analysis ===")
    
    data_path = Path("WESAD")
    subjects = []
    
    for item in data_path.iterdir():
        if item.is_dir() and item.name.startswith('S'):
            subjects.append(item.name)
    
    stats_data = []
    
    for subject in sorted(subjects):
        print(f"Processing {subject}...")
        data = load_subject_data(subject)
        
        if data and 'signal' in data:
            for signal_type in ['chest', 'wrist']:
                if signal_type in data['signal']:
                    for signal_name, signal_data in data['signal'][signal_type].items():
                        if isinstance(signal_data, np.ndarray):
                            # Sample data for efficiency
                            if len(signal_data) > 100000:
                                indices = np.random.choice(len(signal_data), 100000, replace=False)
                                sampled_data = signal_data[indices]
                            else:
                                sampled_data = signal_data.flatten()
                            
                            # Handle multi-dimensional signals
                            if len(sampled_data.shape) > 1:
                                for dim in range(sampled_data.shape[1]):
                                    dim_data = sampled_data[:, dim]
                                    stats = {
                                        'Subject': subject,
                                        'Signal': f"{signal_type}_{signal_name}_dim{dim}",
                                        'Mean': np.mean(dim_data),
                                        'Std': np.std(dim_data),
                                        'Min': np.min(dim_data),
                                        'Max': np.max(dim_data),
                                        'Samples': len(signal_data)
                                    }
                                    stats_data.append(stats)
                            else:
                                stats = {
                                    'Subject': subject,
                                    'Signal': f"{signal_type}_{signal_name}",
                                    'Mean': np.mean(sampled_data),
                                    'Std': np.std(sampled_data),
                                    'Min': np.min(sampled_data),
                                    'Max': np.max(sampled_data),
                                    'Samples': len(signal_data)
                                }
                                stats_data.append(stats)
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        print("\nStatistical Summary (sampled data):")
        print(stats_df.to_string(index=False))
        return stats_df
    return None

def detect_outliers_efficient():
    """Detect outliers efficiently using sampling"""
    print("\n=== Efficient Outlier Detection ===")
    
    outlier_rules = {
        'ECG': {'min': -2, 'max': 2},
        'BVP': {'min': -1000, 'max': 1000},
        'Resp': {'min': -50, 'max': 50},
        'EDA': {'min': 0, 'max': 20},
        'Temp': {'min': 25, 'max': 42},
        'EMG': {'min': -1, 'max': 1},
        'ACC': {'min': -5, 'max': 5}
    }
    
    data_path = Path("WESAD")
    subjects = []
    
    for item in data_path.iterdir():
        if item.is_dir() and item.name.startswith('S'):
            subjects.append(item.name)
    
    for subject in sorted(subjects):
        print(f"\n{subject}:")
        data = load_subject_data(subject)
        
        if data and 'signal' in data:
            for signal_type in ['chest', 'wrist']:
                if signal_type in data['signal']:
                    for signal_name, signal_data in data['signal'][signal_type].items():
                        if isinstance(signal_data, np.ndarray) and signal_name in outlier_rules:
                            # Sample for efficiency
                            if len(signal_data) > 50000:
                                indices = np.random.choice(len(signal_data), 50000, replace=False)
                                sampled_data = signal_data[indices]
                            else:
                                sampled_data = signal_data.flatten()
                            
                            rule = outlier_rules[signal_name]
                            outliers = np.sum((sampled_data < rule['min']) | (sampled_data > rule['max']))
                            outlier_pct = (outliers / len(sampled_data)) * 100
                            
                            if outliers > 0:
                                print(f"  {signal_type}_{signal_name}: {outliers} outliers ({outlier_pct:.2f}%)")

def create_efficient_plots():
    """Create plots efficiently using sampling"""
    print("\n=== Creating Efficient Plots ===")
    
    data_path = Path("WESAD")
    subjects = []
    
    for item in data_path.iterdir():
        if item.is_dir() and item.name.startswith('S'):
            subjects.append(item.name)
    
    # Focus on key signals for efficiency
    key_signals = ['chest_ECG', 'chest_Resp', 'chest_EDA', 'chest_Temp']
    
    # Collect sampled data for each signal
    signal_data = {}
    
    for subject in sorted(subjects):
        print(f"Sampling {subject}...")
        data = load_subject_data(subject)
        
        if data and 'signal' in data:
            for signal_name in key_signals:
                signal_type, signal_key = signal_name.split('_', 1)
                
                if signal_type in data['signal'] and signal_key in data['signal'][signal_type]:
                    signal_array = data['signal'][signal_type][signal_key]
                    
                    if isinstance(signal_array, np.ndarray):
                        # Sample data efficiently
                        if len(signal_array) > 10000:
                            indices = np.random.choice(len(signal_array), 10000, replace=False)
                            sampled_data = signal_array[indices]
                        else:
                            sampled_data = signal_array.flatten()
                        
                        if signal_name not in signal_data:
                            signal_data[signal_name] = []
                        signal_data[signal_name].extend(sampled_data)
    
    # Create histograms
    n_signals = len(signal_data)
    cols = 2
    rows = (n_signals + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    fig.suptitle('WESAD Signal Distributions (Sampled Data)', fontsize=16)
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (signal_name, values) in enumerate(signal_data.items()):
        row = idx // cols
        col = idx % cols
        
        if rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'{signal_name}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide empty subplots
    for idx in range(len(signal_data), rows * cols):
        row = idx // cols
        col = idx % cols
        if rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('wesad_efficient_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_subject_comparison():
    """Create efficient subject comparison"""
    print("\n=== Creating Subject Comparison ===")
    
    data_path = Path("WESAD")
    subjects = []
    
    for item in data_path.iterdir():
        if item.is_dir() and item.name.startswith('S'):
            subjects.append(item.name)
    
    # Focus on ECG for comparison
    comparison_data = []
    
    for subject in sorted(subjects):
        data = load_subject_data(subject)
        
        if data and 'signal' in data and 'chest' in data['signal'] and 'ECG' in data['signal']['chest']:
            ecg_data = data['signal']['chest']['ECG']
            
            # Sample for efficiency
            if len(ecg_data) > 50000:
                indices = np.random.choice(len(ecg_data), 50000, replace=False)
                sampled_data = ecg_data[indices]
            else:
                sampled_data = ecg_data.flatten()
            
            comparison_data.append({
                'Subject': subject,
                'Mean': np.mean(sampled_data),
                'Std': np.std(sampled_data),
                'Min': np.min(sampled_data),
                'Max': np.max(sampled_data)
            })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Subject-wise ECG Comparison (Sampled Data)', fontsize=16)
        
        # Mean comparison
        axes[0, 0].bar(comp_df['Subject'], comp_df['Mean'])
        axes[0, 0].set_title('ECG Mean Values')
        axes[0, 0].set_ylabel('Mean Amplitude')
        
        # Std comparison
        axes[0, 1].bar(comp_df['Subject'], comp_df['Std'])
        axes[0, 1].set_title('ECG Standard Deviation')
        axes[0, 1].set_ylabel('Standard Deviation')
        
        # Range comparison
        ranges = comp_df['Max'] - comp_df['Min']
        axes[1, 0].bar(comp_df['Subject'], ranges)
        axes[1, 0].set_title('ECG Range (Max - Min)')
        axes[1, 0].set_ylabel('Range')
        
        # Box plot
        axes[1, 1].boxplot([comp_df['Mean'], comp_df['Std'], ranges], 
                           labels=['Mean', 'Std', 'Range'])
        axes[1, 1].set_title('Distribution of Statistics')
        axes[1, 1].set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig('wesad_subject_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main efficient analysis function"""
    print("=== WESAD Dataset Efficient Analysis ===")
    
    # Explain signals
    explain_signals()
    
    # Analyze data structure
    analyze_data_structure_efficient()
    
    # Calculate statistics efficiently
    stats_df = calculate_efficient_statistics()
    
    # Detect outliers efficiently
    detect_outliers_efficient()
    
    # Create efficient plots
    create_efficient_plots()
    
    # Create subject comparison
    create_subject_comparison()
    
    print("\n=== Efficient Analysis Complete ===")
    print("Generated files:")
    print("- wesad_efficient_histograms.png")
    print("- wesad_subject_comparison.png")

if __name__ == "__main__":
    main() 