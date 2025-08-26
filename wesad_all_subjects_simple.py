import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def create_subject_overview_plots(all_data, subjects):
    """Create overview plots for all subjects"""
    print("Creating subject overview plots...")
    
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

def create_subject_boxplots(all_data, subjects):
    """Create boxplots comparing subjects"""
    print("Creating subject boxplots...")
    
    # Collect data for boxplots (sample 3000 points per subject)
    signal_data = {}
    
    for subject in subjects:
        data = all_data[subject]
        signals = data.get('signal', {})
        
        for signal_type in ['chest', 'wrist']:
            if signal_type in signals:
                for signal_name, signal_array in signals[signal_type].items():
                    if isinstance(signal_array, np.ndarray):
                        # Sample data
                        if len(signal_array) > 3000:
                            indices = np.random.choice(len(signal_array), 3000, replace=False)
                            sampled_data = signal_array[indices]
                        else:
                            sampled_data = signal_array.flatten()
                        
                        key = f'{signal_type}_{signal_name}'
                        if key not in signal_data:
                            signal_data[key] = []
                        
                        signal_data[key].extend(sampled_data)
    
    # Create boxplots for top 6 signals
    top_signals = list(signal_data.keys())[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('WESAD Dataset - Subject-wise Signal Variability (Boxplots)', fontsize=16)
    
    positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    
    for idx, signal_name in enumerate(top_signals):
        row, col = positions[idx]
        
        data = signal_data[signal_name]
        
        # Create boxplot
        axes[row, col].boxplot(data, labels=[signal_name])
        axes[row, col].set_title(f'{signal_name}')
        axes[row, col].set_ylabel('Value')
        
        # Add statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        axes[row, col].text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                           transform=axes[row, col].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('wesad_subject_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_subject_statistics(all_data, subjects):
    """Create statistical comparison table"""
    print("Creating subject statistics...")
    
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
        
        # Add signal statistics (sample 5000 points)
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

def create_subject_comparison_heatmap(all_data, subjects):
    """Create heatmap comparing subjects"""
    print("Creating subject comparison heatmap...")
    
    # Collect mean values for each subject and signal
    comparison_data = []
    
    for subject in subjects:
        data = all_data[subject]
        signals = data.get('signal', {})
        
        subject_data = {'Subject': subject}
        
        # Add signal means (sample 2000 points)
        for signal_type in ['chest', 'wrist']:
            if signal_type in signals:
                for signal_name, signal_array in signals[signal_type].items():
                    if isinstance(signal_array, np.ndarray):
                        # Sample data
                        if len(signal_array) > 2000:
                            indices = np.random.choice(len(signal_array), 2000, replace=False)
                            sampled_data = signal_array[indices]
                        else:
                            sampled_data = signal_array.flatten()
                        
                        key = f'{signal_type}_{signal_name}'
                        subject_data[key] = np.mean(sampled_data)
        
        comparison_data.append(subject_data)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create heatmap (exclude Subject column)
    signal_cols = [col for col in comparison_df.columns if col != 'Subject']
    if len(signal_cols) > 0:
        heatmap_data = comparison_df[signal_cols].T
        heatmap_data.columns = comparison_df['Subject']
        
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
        plt.colorbar(label='Mean Signal Value')
        plt.title('WESAD Dataset - Subject-wise Signal Comparison')
        plt.xlabel('Subject')
        plt.ylabel('Signal')
        plt.xticks(range(len(comparison_df['Subject'])), comparison_df['Subject'], rotation=45)
        plt.yticks(range(len(signal_cols)), signal_cols)
        
        # Add text annotations
        for i in range(len(signal_cols)):
            for j in range(len(comparison_df['Subject'])):
                value = heatmap_data.iloc[i, j]
                plt.text(j, i, f'{value:.2f}', ha='center', va='center', color='white', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('wesad_subject_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function"""
    print("=== WESAD All Subjects Simple Analysis ===")
    
    # Load all subjects
    all_data, subjects = load_all_subjects()
    
    if not all_data:
        print("No data loaded!")
        return
    
    print(f"\nLoaded {len(subjects)} subjects: {subjects}")
    
    # Create comparison plots
    create_subject_overview_plots(all_data, subjects)
    create_subject_boxplots(all_data, subjects)
    create_subject_comparison_heatmap(all_data, subjects)
    create_subject_statistics(all_data, subjects)
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- wesad_all_subjects_overview.png")
    print("- wesad_subject_boxplots.png")
    print("- wesad_subject_heatmap.png")
    print("- wesad_subject_statistics.csv")

if __name__ == "__main__":
    main() 