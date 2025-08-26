import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class WESADDetailedAnalyzer:
    def __init__(self, data_path="WESAD"):
        self.data_path = Path(data_path)
        self.subjects = []
        self.data = {}
        self.load_subjects()
    
    def load_subjects(self):
        """Load all available subjects"""
        print("Loading WESAD dataset...")
        for item in self.data_path.iterdir():
            if item.is_dir() and item.name.startswith('S'):
                self.subjects.append(item.name)
        
        print(f"Found {len(self.subjects)} subjects: {sorted(self.subjects)}")
    
    def load_subject_data(self, subject):
        """Load data for a specific subject"""
        subject_path = self.data_path / subject
        pkl_file = subject_path / f"{subject}.pkl"
        
        if pkl_file.exists():
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                return data
            except Exception as e:
                print(f"Error loading {subject}: {e}")
                return None
        else:
            print(f"No pickle file found for {subject}")
            return None
    
    def explore_data_structure(self, subject='S2'):
        """Explore the structure of the data"""
        print(f"\n=== Exploring data structure for {subject} ===")
        data = self.load_subject_data(subject)
        
        if data is None:
            return
        
        print(f"Data type: {type(data)}")
        print(f"Data keys: {list(data.keys())}")
        
        for key, value in data.items():
            print(f"\n{key}:")
            if isinstance(value, dict):
                print(f"  Type: dict with keys: {list(value.keys())}")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        print(f"    {subkey}: numpy array, shape: {subvalue.shape}, dtype: {subvalue.dtype}")
                    elif isinstance(subvalue, dict):
                        print(f"    {subkey}: dict with keys: {list(subvalue.keys())}")
                    else:
                        print(f"    {subkey}: {type(subvalue)}")
            elif isinstance(value, np.ndarray):
                print(f"  Type: numpy array, shape: {value.shape}, dtype: {value.dtype}")
            else:
                print(f"  Type: {type(value)}")
    
    def analyze_signal_quality(self, subject='S2'):
        """Analyze signal quality and missing values"""
        print(f"\n=== Signal Quality Analysis for {subject} ===")
        data = self.load_subject_data(subject)
        
        if data is None:
            return
        
        # Analyze each signal type
        for signal_type in ['chest', 'wrist']:
            if signal_type in data.get('signal', {}):
                print(f"\n{signal_type.upper()} signals:")
                signals = data['signal'][signal_type]
                
                for signal_name, signal_data in signals.items():
                    if isinstance(signal_data, np.ndarray):
                        print(f"  {signal_name}:")
                        print(f"    Shape: {signal_data.shape}")
                        print(f"    NaN count: {np.isnan(signal_data).sum()}")
                        print(f"    Inf count: {np.isinf(signal_data).sum()}")
                        print(f"    Min: {np.nanmin(signal_data):.4f}")
                        print(f"    Max: {np.nanmax(signal_data):.4f}")
                        print(f"    Mean: {np.nanmean(signal_data):.4f}")
                        print(f"    Std: {np.nanstd(signal_data):.4f}")
    
    def create_comprehensive_plots(self, subject='S2'):
        """Create comprehensive plots for data visualization"""
        print(f"\n=== Creating comprehensive plots for {subject} ===")
        data = self.load_subject_data(subject)
        
        if data is None:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'WESAD Dataset Analysis - {subject}', fontsize=16)
        
        signals = data.get('signal', {})
        
        # Plot 1: Heart rate comparison (chest ECG vs wrist BVP)
        if 'chest' in signals and 'wrist' in signals:
            chest_ecg = signals['chest'].get('ECG', None)
            wrist_bvp = signals['wrist'].get('BVP', None)
            
            if chest_ecg is not None:
                axes[0, 0].plot(chest_ecg[:2000], label='Chest ECG', alpha=0.7, linewidth=0.5)
            if wrist_bvp is not None:
                axes[0, 0].plot(wrist_bvp[:2000], label='Wrist BVP', alpha=0.7, linewidth=0.5)
            axes[0, 0].set_title('Heart Rate Signals (First 2000 samples)')
            axes[0, 0].legend()
            axes[0, 0].set_ylabel('Amplitude')
        
        # Plot 2: Breathing signal
        if 'chest' in signals and 'Resp' in signals['chest']:
            axes[0, 1].plot(signals['chest']['Resp'][:2000])
            axes[0, 1].set_title('Chest Respiration Signal')
            axes[0, 1].set_ylabel('Amplitude')
        
        # Plot 3: EDA (Electrodermal Activity)
        if 'chest' in signals and 'EDA' in signals['chest']:
            axes[0, 2].plot(signals['chest']['EDA'][:2000])
            axes[0, 2].set_title('Chest EDA Signal')
            axes[0, 2].set_ylabel('Amplitude')
        
        # Plot 4: Temperature
        if 'chest' in signals and 'Temp' in signals['chest']:
            axes[1, 0].plot(signals['chest']['Temp'][:2000])
            axes[1, 0].set_title('Chest Temperature Signal')
            axes[1, 0].set_ylabel('Temperature')
        
        # Plot 5: EMG
        if 'chest' in signals and 'EMG' in signals['chest']:
            axes[1, 1].plot(signals['chest']['EMG'][:2000])
            axes[1, 1].set_title('Chest EMG Signal')
            axes[1, 1].set_ylabel('Amplitude')
        
        # Plot 6: ACC (Accelerometer)
        if 'chest' in signals and 'ACC' in signals['chest']:
            acc_data = signals['chest']['ACC']
            if len(acc_data.shape) > 1:
                for i in range(min(3, acc_data.shape[1])):
                    axes[1, 2].plot(acc_data[:2000, i], label=f'ACC_{i}', alpha=0.7)
            else:
                axes[1, 2].plot(acc_data[:2000])
            axes[1, 2].set_title('Chest Accelerometer Signal')
            axes[1, 2].set_ylabel('Amplitude')
            if len(acc_data.shape) > 1:
                axes[1, 2].legend()
        
        # Plot 7: Histogram of ECG signal
        if 'chest' in signals and 'ECG' in signals['chest']:
            axes[2, 0].hist(signals['chest']['ECG'], bins=50, alpha=0.7, edgecolor='black')
            axes[2, 0].set_title('ECG Signal Distribution')
            axes[2, 0].set_xlabel('Amplitude')
            axes[2, 0].set_ylabel('Frequency')
        
        # Plot 8: Labels distribution
        if 'label' in data:
            labels = data['label']
            unique_labels, counts = np.unique(labels, return_counts=True)
            axes[2, 1].bar(unique_labels, counts)
            axes[2, 1].set_title('Label Distribution')
            axes[2, 1].set_xlabel('Label')
            axes[2, 1].set_ylabel('Count')
            
            # Add label meanings
            label_meanings = {0: 'Baseline', 1: 'Stress', 2: 'Amusement', 3: 'Meditation'}
            for label, count in zip(unique_labels, counts):
                meaning = label_meanings.get(label, f'Label {label}')
                axes[2, 1].text(label, count, f'{meaning}\n({count})', 
                               ha='center', va='bottom')
        
        # Plot 9: Signal quality heatmap
        if 'chest' in signals and 'wrist' in signals:
            quality_data = []
            signal_names = []
            
            for signal_type in ['chest', 'wrist']:
                for signal_name, signal_data in signals[signal_type].items():
                    if isinstance(signal_data, np.ndarray):
                        quality_data.append([
                            np.nanmean(signal_data),
                            np.nanstd(signal_data),
                            np.isnan(signal_data).sum() / len(signal_data) * 100
                        ])
                        signal_names.append(f'{signal_type}_{signal_name}')
            
            if quality_data:
                quality_df = pd.DataFrame(quality_data, 
                                        columns=['Mean', 'Std', 'NaN%'],
                                        index=signal_names)
                
                sns.heatmap(quality_df.T, annot=True, fmt='.3f', 
                           cmap='viridis', ax=axes[2, 2])
                axes[2, 2].set_title('Signal Quality Metrics')
        
        plt.tight_layout()
        plt.savefig(f'wesad_comprehensive_plots_{subject}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def correlation_analysis(self, subject='S2'):
        """Analyze correlations between different signals"""
        print(f"\n=== Correlation Analysis for {subject} ===")
        data = self.load_subject_data(subject)
        
        if data is None:
            return
        
        signals = data.get('signal', {})
        
        # Collect all signals
        signal_data = {}
        
        for signal_type in ['chest', 'wrist']:
            if signal_type in signals:
                for signal_name, signal_array in signals[signal_type].items():
                    if isinstance(signal_array, np.ndarray):
                        # Handle multi-dimensional signals (like ACC)
                        if len(signal_array.shape) > 1:
                            for i in range(signal_array.shape[1]):
                                signal_data[f'{signal_type}_{signal_name}_{i}'] = signal_array[:, i]
                        else:
                            signal_data[f'{signal_type}_{signal_name}'] = signal_array
        
        # Create correlation matrix
        if len(signal_data) > 1:
            # Sample data for correlation (first 10000 samples to avoid memory issues)
            sample_size = min(10000, min(len(signal) for signal in signal_data.values()))
            signal_df = pd.DataFrame({
                name: signal[:sample_size] for name, signal in signal_data.items()
            })
            
            # Calculate correlation matrix
            corr_matrix = signal_df.corr()
            
            # Plot correlation heatmap
            plt.figure(figsize=(15, 12))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.3f', cbar_kws={"shrink": .8})
            plt.title(f'Signal Correlations - {subject}')
            plt.tight_layout()
            plt.savefig(f'wesad_correlations_{subject}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("\nTop correlations:")
            # Get top correlations (excluding self-correlations)
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_val):
                        corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            # Sort by absolute correlation value
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            print("Top 10 correlations:")
            for i, (signal1, signal2, corr_val) in enumerate(corr_pairs[:10]):
                print(f"{i+1}. {signal1} ↔ {signal2}: {corr_val:.3f}")
    
    def statistical_summary(self, subject='S2'):
        """Generate statistical summary of all signals"""
        print(f"\n=== Statistical Summary for {subject} ===")
        data = self.load_subject_data(subject)
        
        if data is None:
            return
        
        signals = data.get('signal', {})
        stats_data = []
        
        for signal_type in ['chest', 'wrist']:
            if signal_type in signals:
                for signal_name, signal_array in signals[signal_type].items():
                    if isinstance(signal_array, np.ndarray):
                        if len(signal_array.shape) > 1:
                            # Handle multi-dimensional signals
                            for i in range(signal_array.shape[1]):
                                stats = {
                                    'Signal': f'{signal_type}_{signal_name}_{i}',
                                    'Mean': np.nanmean(signal_array[:, i]),
                                    'Std': np.nanstd(signal_array[:, i]),
                                    'Min': np.nanmin(signal_array[:, i]),
                                    'Max': np.nanmax(signal_array[:, i]),
                                    'NaN_Count': np.isnan(signal_array[:, i]).sum(),
                                    'Inf_Count': np.isinf(signal_array[:, i]).sum(),
                                    'Length': len(signal_array)
                                }
                                stats_data.append(stats)
                        else:
                            stats = {
                                'Signal': f'{signal_type}_{signal_name}',
                                'Mean': np.nanmean(signal_array),
                                'Std': np.nanstd(signal_array),
                                'Min': np.nanmin(signal_array),
                                'Max': np.nanmax(signal_array),
                                'NaN_Count': np.isnan(signal_array).sum(),
                                'Inf_Count': np.isinf(signal_array).sum(),
                                'Length': len(signal_array)
                            }
                            stats_data.append(stats)
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            print(stats_df.to_string(index=False))
            return stats_df
        else:
            print("No signal data found")
            return None
    
    def compare_subjects(self, subjects=['S2', 'S3', 'S4']):
        """Compare data across different subjects"""
        print(f"\n=== Comparing Subjects: {subjects} ===")
        
        comparison_data = []
        
        for subject in subjects:
            data = self.load_subject_data(subject)
            if data is None:
                continue
            
            signals = data.get('signal', {})
            
            # Analyze chest ECG if available
            if 'chest' in signals and 'ECG' in signals['chest']:
                ecg_data = signals['chest']['ECG']
                comparison_data.append({
                    'Subject': subject,
                    'ECG_Mean': np.nanmean(ecg_data),
                    'ECG_Std': np.nanstd(ecg_data),
                    'ECG_Length': len(ecg_data),
                    'ECG_NaN_Count': np.isnan(ecg_data).sum(),
                    'ECG_NaN_Percent': np.isnan(ecg_data).sum() / len(ecg_data) * 100
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            print(comp_df.to_string(index=False))
            
            # Plot comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # ECG mean comparison
            axes[0, 0].bar(comp_df['Subject'], comp_df['ECG_Mean'])
            axes[0, 0].set_title('ECG Mean Values Across Subjects')
            axes[0, 0].set_ylabel('Mean Amplitude')
            
            # ECG std comparison
            axes[0, 1].bar(comp_df['Subject'], comp_df['ECG_Std'])
            axes[0, 1].set_title('ECG Standard Deviation Across Subjects')
            axes[0, 1].set_ylabel('Standard Deviation')
            
            # ECG NaN percentage
            axes[1, 0].bar(comp_df['Subject'], comp_df['ECG_NaN_Percent'])
            axes[1, 0].set_title('ECG NaN Percentage Across Subjects')
            axes[1, 0].set_ylabel('NaN Percentage (%)')
            
            # Data length comparison
            axes[1, 1].bar(comp_df['Subject'], comp_df['ECG_Length'])
            axes[1, 1].set_title('ECG Data Length Across Subjects')
            axes[1, 1].set_ylabel('Number of Samples')
            
            plt.tight_layout()
            plt.savefig('wesad_subject_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def comprehensive_analysis(self):
        """Run comprehensive analysis on the dataset"""
        print("=== WESAD Dataset Comprehensive Analysis ===")
        
        # Explore data structure
        self.explore_data_structure()
        
        # Analyze signal quality
        self.analyze_signal_quality()
        
        # Create comprehensive plots
        self.create_comprehensive_plots()
        
        # Correlation analysis
        self.correlation_analysis()
        
        # Statistical summary
        self.statistical_summary()
        
        # Compare subjects
        self.compare_subjects()

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = WESADDetailedAnalyzer()
    
    # Run comprehensive analysis
    analyzer.comprehensive_analysis() 