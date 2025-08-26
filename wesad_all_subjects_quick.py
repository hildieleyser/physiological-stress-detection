import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_all_subjects(data_path="WESAD"):
    """Load data for all subjects"""
    data_path = Path(data_path)
    subjects = []
    all_data = {}
    
    # Find all subject directories
    for item in data_path.iterdir():
        if item.is_dir() and item.name.startswith('S'):
            subjects.append(item.name)
    
    subjects.sort()
    print(f"Found subjects: {subjects}")
    
    # Load data for each subject
    for subject in subjects:
        pkl_file = data_path / subject / f"{subject}.pkl"
        if pkl_file.exists():
            with open(pkl_file, 'rb') as f:
                all_data[subject] = pickle.load(f, encoding='latin1')
            print(f"Loaded {subject}")
        else:
            print(f"No data found for {subject}")
    
    return all_data, subjects

def create_subject_comparison_plots(all_data, subjects):
    """Create plots comparing all subjects"""
    print("Creating subject comparison plots...")
    
    # 1. Signal overview for each subject
    fig, axes = plt.subplots(len(subjects), 3, figsize=(15, 4*len(subjects)))
    fig.suptitle('WESAD Dataset - All Subjects Overview', fontsize=16)
    
    for idx, subject in enumerate(subjects):
        data = all_data[subject]
        signals = data.get('signal', {})
        
        # ECG plot
        if 'chest' in signals and 'ECG' in signals['chest']:
            ecg = signals['chest']['ECG'][:3000].flatten()  # First 3000 samples
            axes[idx, 0].plot(ecg, linewidth=0.5)
            axes[idx, 0].set_title(f'{subject} - ECG')
            axes[idx, 0].set_ylabel('Amplitude')
        
        # BVP plot
        if 'wrist' in signals and 'BVP' in signals['wrist']:
            bvp = signals['wrist']['BVP'][:3000].flatten()  # First 3000 samples
            axes[idx, 1].plot(bvp, linewidth=0.5)
            axes[idx, 1].set_title(f'{subject} - BVP')
            axes[idx, 1].set_ylabel('Amplitude')
        
        # Label distribution
        if 'label' in data:
            labels = data['label']
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_names = {0: 'Baseline', 1: 'Stress', 2: 'Amusement', 3: 'Meditation'}
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
            bars = axes[idx, 2].bar(unique_labels, counts, color=colors[:len(unique_labels)])
            axes[idx, 2].set_title(f'{subject} - Labels')
            axes[idx, 2].set_xlabel('Label')
            axes[idx, 2].set_ylabel('Count')
            
            # Add counts on bars
            for bar, label in zip(bars, unique_labels):
                height = bar.get_height()
                axes[idx, 2].text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height):,}',
                               ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('wesad_all_subjects_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Subject-wise violin plots for key signals
    create_subject_violin_plots(all_data, subjects)
    
    # 3. Statistical comparison table
    create_statistics_comparison(all_data, subjects)

def create_subject_violin_plots(all_data, subjects):
    """Create violin plots comparing subjects"""
    print("Creating violin plots...")
    
    # Collect data for violin plots (sample 5000 points per subject)
    signal_data = {}
    
    for subject in subjects:
        data = all_data[subject]
        signals = data.get('signal', {})
        
        for signal_type in ['chest', 'wrist']:
            if signal_type in signals:
                for signal_name, signal_array in signals[signal_type].items():
                    if isinstance(signal_array, np.ndarray):
                        # Sample data
                        if len(signal_array) > 5000:
                            indices = np.random.choice(len(signal_array), 5000, replace=False)
                            sampled_data = signal_array[indices]
                        else:
                            sampled_data = signal_array.flatten()
                        
                        key = f'{signal_type}_{signal_name}'
                        if key not in signal_data:
                            signal_data[key] = {'data': [], 'subjects': []}
                        
                        signal_data[key]['data'].extend(sampled_data)
                        signal_data[key]['subjects'].extend([subject] * len(sampled_data))
    
    # Create violin plots for top 6 signals
    top_signals = list(signal_data.keys())[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('WESAD Dataset - Subject-wise Signal Variability', fontsize=16)
    
    positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    
    for idx, signal_name in enumerate(top_signals):
        row, col = positions[idx]
        
        data = signal_data[signal_name]['data']
        subjects_list = signal_data[signal_name]['subjects']
        
        # Create DataFrame for seaborn
        df = pd.DataFrame({
            'Signal': data,
            'Subject': subjects_list
        })
        
        # Create violin plot
        sns.violinplot(data=df, x='Subject', y='Signal', ax=axes[row, col])
        axes[row, col].set_title(f'{signal_name}')
        axes[row, col].tick_params(axis='x', rotation=45)
        
        # Add statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        axes[row, col].text(0.02, 0.98, f'Overall Mean: {mean_val:.3f}\nOverall Std: {std_val:.3f}', 
                           transform=axes[row, col].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('wesad_subject_violin_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_statistics_comparison(all_data, subjects):
    """Create statistical comparison table"""
    print("Creating statistics comparison...")
    
    # Collect statistics for each subject
    stats_data = []
    
    for subject in subjects:
        data = all_data[subject]
        signals = data.get('signal', {})
        
        subject_stats = {'Subject': subject}
        
        # Add label counts
        if 'label' in data:
            labels = data['label']
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_names = {0: 'Baseline', 1: 'Stress', 2: 'Amusement', 3: 'Meditation'}
            
            for label, count in zip(unique_labels, counts):
                name = label_names.get(label, f'Label_{label}')
                subject_stats[f'{name}_count'] = count
        
        # Add signal statistics (sample 10000 points)
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
                        
                        key = f'{signal_type}_{signal_name}'
                        subject_stats[f'{key}_mean'] = np.mean(sampled_data)
                        subject_stats[f'{key}_std'] = np.std(sampled_data)
                        subject_stats[f'{key}_min'] = np.min(sampled_data)
                        subject_stats[f'{key}_max'] = np.max(sampled_data)
        
        stats_data.append(subject_stats)
    
    # Create DataFrame and display
    stats_df = pd.DataFrame(stats_data)
    
    # Save to CSV
    stats_df.to_csv('wesad_subject_statistics.csv', index=False)
    
    # Display summary
    print("\n=== Subject Statistics Summary ===")
    print(f"Number of subjects: {len(subjects)}")
    print(f"Statistics saved to: wesad_subject_statistics.csv")
    
    # Show label distribution summary
    label_cols = [col for col in stats_df.columns if 'count' in col]
    if label_cols:
        print("\nLabel Distribution Summary:")
        print(stats_df[['Subject'] + label_cols].to_string(index=False))
    
    # Show signal statistics summary
    signal_cols = [col for col in stats_df.columns if 'mean' in col]
    if signal_cols:
        print(f"\nSignal Statistics Summary (showing first 3 signals):")
        first_3_signals = signal_cols[:3]
        print(stats_df[['Subject'] + first_3_signals].to_string(index=False))

def main():
    """Main function"""
    print("=== WESAD All Subjects Quick Analysis ===")
    
    # Load all subjects
    all_data, subjects = load_all_subjects()
    
    if not all_data:
        print("No data loaded!")
        return
    
    print(f"\nLoaded {len(subjects)} subjects: {subjects}")
    
    # Create comparison plots
    create_subject_comparison_plots(all_data, subjects)
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- wesad_all_subjects_overview.png")
    print("- wesad_subject_violin_plots.png")
    print("- wesad_subject_statistics.csv")

if __name__ == "__main__":
    main() 