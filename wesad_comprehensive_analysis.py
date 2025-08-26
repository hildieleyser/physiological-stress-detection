import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_all_subjects(data_path="WESAD"):
    """Load data for all available subjects"""
    data_path = Path(data_path)
    subjects = []
    all_data = {}
    
    for item in data_path.iterdir():
        if item.is_dir() and item.name.startswith('S'):
            subjects.append(item.name)
    
    print(f"Found {len(subjects)} subjects: {sorted(subjects)}")
    
    for subject in subjects:
        subject_path = data_path / subject
        pkl_file = subject_path / f"{subject}.pkl"
        
        if pkl_file.exists():
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                all_data[subject] = data
            except Exception as e:
                print(f"Error loading {subject}: {e}")
    
    return all_data, subjects

def explain_signals():
    """Explain what each signal represents"""
    print("=== Signal Explanations ===")
    
    signals_info = {
        'ECG': {
            'description': 'Electrocardiogram - measures electrical activity of the heart',
            'units': 'mV (millivolts)',
            'normal_range': 'Heart rate: 60-100 BPM',
            'outlier_range': 'Heart rate < 30 or > 220 BPM',
            'location': 'Chest',
            'purpose': 'Primary heart rate monitoring, arrhythmia detection'
        },
        'BVP': {
            'description': 'Blood Volume Pulse - measures blood flow changes in wrist',
            'units': 'Raw sensor values',
            'normal_range': 'Heart rate: 60-100 BPM',
            'outlier_range': 'Heart rate < 30 or > 220 BPM',
            'location': 'Wrist',
            'purpose': 'Alternative heart rate monitoring, more convenient placement'
        },
        'Resp': {
            'description': 'Respiration - measures breathing rate and depth',
            'units': 'Arbitrary units',
            'normal_range': '12-20 breaths per minute',
            'outlier_range': 'Breathing rate < 5 or > 50 BPM',
            'location': 'Chest',
            'purpose': 'Breathing pattern analysis, stress indicator'
        },
        'EDA': {
            'description': 'Electrodermal Activity - measures skin conductance',
            'units': 'Microsiemens (μS)',
            'normal_range': '0.1-10 μS',
            'outlier_range': 'Values < 0 or > 50 μS',
            'location': 'Chest and Wrist',
            'purpose': 'Stress and emotional arousal detection'
        },
        'Temp': {
            'description': 'Body Temperature - measures skin temperature',
            'units': '°C (Celsius)',
            'normal_range': '32-37°C (skin temperature)',
            'outlier_range': 'Temperature < 25°C or > 42°C',
            'location': 'Chest and Wrist',
            'purpose': 'Thermal stress, physiological state monitoring'
        },
        'EMG': {
            'description': 'Electromyography - measures muscle electrical activity',
            'units': 'μV (microvolts)',
            'normal_range': 'Variable based on muscle tension',
            'outlier_range': 'Extreme values beyond normal muscle activity',
            'location': 'Chest',
            'purpose': 'Muscle tension, stress, physical activity detection'
        },
        'ACC': {
            'description': 'Accelerometer - measures movement and orientation',
            'units': 'g (gravitational acceleration)',
            'normal_range': '-2g to +2g per axis',
            'outlier_range': 'Values beyond ±5g',
            'location': 'Chest and Wrist',
            'purpose': 'Activity level, movement patterns, posture detection'
        }
    }
    
    for signal_name, info in signals_info.items():
        print(f"\n{signal_name} ({info['location']}):")
        print(f"  Description: {info['description']}")
        print(f"  Units: {info['units']}")
        print(f"  Normal Range: {info['normal_range']}")
        print(f"  Outlier Range: {info['outlier_range']}")
        print(f"  Purpose: {info['purpose']}")

def analyze_data_structure(all_data, subjects):
    """Analyze the structure of data across all subjects"""
    print("\n=== Data Structure Analysis ===")
    
    # Analyze shapes and unique subjects
    print(f"Number of subjects: {len(subjects)}")
    print(f"Subject IDs: {sorted(subjects)}")
    
    # Analyze signal availability across subjects
    signal_availability = {}
    
    for subject in subjects:
        if subject in all_data:
            data = all_data[subject]
            if 'signal' in data:
                for signal_type in ['chest', 'wrist']:
                    if signal_type in data['signal']:
                        for signal_name in data['signal'][signal_type].keys():
                            key = f"{signal_type}_{signal_name}"
                            if key not in signal_availability:
                                signal_availability[key] = []
                            signal_availability[key].append(subject)
    
    print("\nSignal availability across subjects:")
    for signal, available_subjects in signal_availability.items():
        print(f"  {signal}: {len(available_subjects)} subjects ({', '.join(available_subjects)})")
    
    # Analyze data shapes
    print("\nData shapes for each subject:")
    for subject in subjects:
        if subject in all_data:
            data = all_data[subject]
            if 'signal' in data:
                print(f"\n{subject}:")
                for signal_type in ['chest', 'wrist']:
                    if signal_type in data['signal']:
                        print(f"  {signal_type.upper()}:")
                        for signal_name, signal_data in data['signal'][signal_type].items():
                            if isinstance(signal_data, np.ndarray):
                                print(f"    {signal_name}: {signal_data.shape}")

def detect_outliers(all_data, subjects):
    """Detect outliers in each signal"""
    print("\n=== Outlier Detection ===")
    
    outlier_rules = {
        'ECG': {'min': -2, 'max': 2, 'description': 'ECG amplitude'},
        'BVP': {'min': -1000, 'max': 1000, 'description': 'BVP amplitude'},
        'Resp': {'min': -50, 'max': 50, 'description': 'Respiration amplitude'},
        'EDA': {'min': 0, 'max': 20, 'description': 'Skin conductance'},
        'Temp': {'min': 25, 'max': 42, 'description': 'Temperature (°C)'},
        'EMG': {'min': -1, 'max': 1, 'description': 'EMG amplitude'},
        'ACC': {'min': -5, 'max': 5, 'description': 'Acceleration (g)'}
    }
    
    outlier_summary = {}
    
    for subject in subjects:
        if subject in all_data:
            data = all_data[subject]
            if 'signal' in data:
                print(f"\n{subject}:")
                
                for signal_type in ['chest', 'wrist']:
                    if signal_type in data['signal']:
                        for signal_name, signal_data in data['signal'][signal_type].items():
                            if isinstance(signal_data, np.ndarray):
                                # Handle multi-dimensional signals
                                if len(signal_data.shape) > 1:
                                    for dim in range(signal_data.shape[1]):
                                        dim_data = signal_data[:, dim]
                                        signal_key = f"{signal_type}_{signal_name}_dim{dim}"
                                        
                                        if signal_name in outlier_rules:
                                            rule = outlier_rules[signal_name]
                                            outliers = np.sum((dim_data < rule['min']) | (dim_data > rule['max']))
                                            outlier_pct = (outliers / len(dim_data)) * 100
                                            
                                            if outliers > 0:
                                                print(f"    {signal_key}: {outliers} outliers ({outlier_pct:.2f}%)")
                                                print(f"      Range: [{np.min(dim_data):.4f}, {np.max(dim_data):.4f}]")
                                                print(f"      Expected: [{rule['min']}, {rule['max']}]")
                                else:
                                    signal_key = f"{signal_type}_{signal_name}"
                                    
                                    if signal_name in outlier_rules:
                                        rule = outlier_rules[signal_name]
                                        outliers = np.sum((signal_data < rule['min']) | (signal_data > rule['max']))
                                        outlier_pct = (outliers / len(signal_data)) * 100
                                        
                                        if outliers > 0:
                                            print(f"    {signal_key}: {outliers} outliers ({outlier_pct:.2f}%)")
                                            print(f"      Range: [{np.min(signal_data):.4f}, {np.max(signal_data):.4f}]")
                                            print(f"      Expected: [{rule['min']}, {rule['max']}]")

def calculate_descriptive_statistics(all_data, subjects):
    """Calculate descriptive statistics for each signal"""
    print("\n=== Descriptive Statistics ===")
    
    stats_data = []
    
    for subject in subjects:
        if subject in all_data:
            data = all_data[subject]
            if 'signal' in data:
                for signal_type in ['chest', 'wrist']:
                    if signal_type in data['signal']:
                        for signal_name, signal_data in data['signal'][signal_type].items():
                            if isinstance(signal_data, np.ndarray):
                                # Handle multi-dimensional signals
                                if len(signal_data.shape) > 1:
                                    for dim in range(signal_data.shape[1]):
                                        dim_data = signal_data[:, dim]
                                        stats = {
                                            'Subject': subject,
                                            'Signal': f"{signal_type}_{signal_name}_dim{dim}",
                                            'Mean': np.mean(dim_data),
                                            'Median': np.median(dim_data),
                                            'Std': np.std(dim_data),
                                            'Min': np.min(dim_data),
                                            'Max': np.max(dim_data),
                                            'Samples': len(dim_data)
                                        }
                                        stats_data.append(stats)
                                else:
                                    stats = {
                                        'Subject': subject,
                                        'Signal': f"{signal_type}_{signal_name}",
                                        'Mean': np.mean(signal_data),
                                        'Median': np.median(signal_data),
                                        'Std': np.std(signal_data),
                                        'Min': np.min(signal_data),
                                        'Max': np.max(signal_data),
                                        'Samples': len(signal_data)
                                    }
                                    stats_data.append(stats)
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        print(stats_df.to_string(index=False))
        return stats_df
    else:
        print("No statistics calculated")
        return None

def create_histograms(all_data, subjects):
    """Create histograms for each signal across all subjects"""
    print("\n=== Creating Histograms ===")
    
    # Collect all signals by type
    signal_collections = {}
    
    for subject in subjects:
        if subject in all_data:
            data = all_data[subject]
            if 'signal' in data:
                for signal_type in ['chest', 'wrist']:
                    if signal_type in data['signal']:
                        for signal_name, signal_data in data['signal'][signal_type].items():
                            if isinstance(signal_data, np.ndarray):
                                signal_key = f"{signal_type}_{signal_name}"
                                
                                if signal_key not in signal_collections:
                                    signal_collections[signal_key] = []
                                
                                # Handle multi-dimensional signals
                                if len(signal_data.shape) > 1:
                                    # Use first dimension for simplicity
                                    signal_collections[signal_key].extend(signal_data[:, 0])
                                else:
                                    signal_collections[signal_key].extend(signal_data)
    
    # Create histograms
    n_signals = len(signal_collections)
    cols = 3
    rows = (n_signals + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    fig.suptitle('Signal Distributions Across All Subjects', fontsize=16)
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (signal_name, signal_values) in enumerate(signal_collections.items()):
        row = idx // cols
        col = idx % cols
        
        if rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        ax.hist(signal_values, bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'{signal_name}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        
        # Add statistics text
        mean_val = np.mean(signal_values)
        std_val = np.std(signal_values)
        ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide empty subplots
    for idx in range(len(signal_collections), rows * cols):
        row = idx // cols
        col = idx % cols
        if rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('wesad_signal_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_subject_violin_plots(all_data, subjects):
    """Create violin plots to show subject-wise variability"""
    print("\n=== Creating Subject-wise Violin Plots ===")
    
    # Focus on key signals for violin plots
    key_signals = ['chest_ECG', 'chest_Resp', 'chest_EDA', 'chest_Temp']
    
    for signal_name in key_signals:
        # Collect data for this signal across subjects
        subject_data = []
        subject_labels = []
        
        for subject in subjects:
            if subject in all_data:
                data = all_data[subject]
                if 'signal' in data:
                    signal_type, signal_key = signal_name.split('_', 1)
                    
                    if signal_type in data['signal'] and signal_key in data['signal'][signal_type]:
                        signal_data = data['signal'][signal_type][signal_key]
                        
                        if isinstance(signal_data, np.ndarray):
                            # Sample data to avoid memory issues
                            if len(signal_data) > 10000:
                                indices = np.random.choice(len(signal_data), 10000, replace=False)
                                sampled_data = signal_data[indices]
                            else:
                                sampled_data = signal_data.flatten()
                            
                            subject_data.extend(sampled_data)
                            subject_labels.extend([subject] * len(sampled_data))
        
        if subject_data:
            # Create violin plot
            plt.figure(figsize=(15, 8))
            
            # Create DataFrame for seaborn
            plot_df = pd.DataFrame({
                'Subject': subject_labels,
                'Value': subject_data
            })
            
            sns.violinplot(data=plot_df, x='Subject', y='Value')
            plt.title(f'{signal_name} - Subject-wise Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'wesad_violin_{signal_name}.png', dpi=300, bbox_inches='tight')
            plt.show()

def main():
    """Main analysis function"""
    print("=== WESAD Dataset Comprehensive Analysis ===")
    
    # Load all subjects
    all_data, subjects = load_all_subjects()
    
    if not all_data:
        print("No data loaded")
        return
    
    # Explain signals
    explain_signals()
    
    # Analyze data structure
    analyze_data_structure(all_data, subjects)
    
    # Detect outliers
    detect_outliers(all_data, subjects)
    
    # Calculate descriptive statistics
    stats_df = calculate_descriptive_statistics(all_data, subjects)
    
    # Create visualizations
    create_histograms(all_data, subjects)
    create_subject_violin_plots(all_data, subjects)
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- wesad_signal_histograms.png")
    print("- wesad_violin_*.png (for each key signal)")

if __name__ == "__main__":
    main() 