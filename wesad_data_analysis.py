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
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WESADAnalyzer:
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
        
        print(f"Found {len(self.subjects)} subjects: {self.subjects}")
    
    def load_subject_data(self, subject):
        """Load data for a specific subject"""
        subject_path = self.data_path / subject
        pkl_file = subject_path / f"{subject}.pkl"
        
        if pkl_file.exists():
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            return data
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
            if signal_type in data:
                print(f"\n{signal_type.upper()} signals:")
                signals = data[signal_type]
                
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
    
    def create_basic_plots(self, subject='S2'):
        """Create basic plots for data visualization"""
        print(f"\n=== Creating basic plots for {subject} ===")
        data = self.load_subject_data(subject)
        
        if data is None:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'WESAD Dataset Analysis - {subject}', fontsize=16)
        
        # Plot 1: Heart rate comparison (chest vs wrist)
        if 'chest' in data and 'wrist' in data:
            chest_hr = data['chest']['ECG'] if 'ECG' in data['chest'] else None
            wrist_hr = data['wrist']['BVP'] if 'BVP' in data['wrist'] else None
            
            if chest_hr is not None:
                axes[0, 0].plot(chest_hr[:1000], label='Chest ECG', alpha=0.7)
            if wrist_hr is not None:
                axes[0, 0].plot(wrist_hr[:1000], label='Wrist BVP', alpha=0.7)
            axes[0, 0].set_title('Heart Rate Signals (First 1000 samples)')
            axes[0, 0].legend()
            axes[0, 0].set_ylabel('Amplitude')
        
        # Plot 2: Breathing signals
        if 'chest' in data:
            chest_signals = data['chest']
            if 'Resp' in chest_signals:
                axes[0, 1].plot(chest_signals['Resp'][:1000])
                axes[0, 1].set_title('Chest Respiration Signal')
                axes[0, 1].set_ylabel('Amplitude')
        
        # Plot 3: Histogram of ECG signal
        if 'chest' in data and 'ECG' in data['chest']:
            axes[1, 0].hist(data['chest']['ECG'], bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('ECG Signal Distribution')
            axes[1, 0].set_xlabel('Amplitude')
            axes[1, 0].set_ylabel('Frequency')
        
        # Plot 4: Labels distribution
        if 'label' in data:
            labels = data['label']
            unique_labels, counts = np.unique(labels, return_counts=True)
            axes[1, 1].bar(unique_labels, counts)
            axes[1, 1].set_title('Label Distribution')
            axes[1, 1].set_xlabel('Label')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f'wesad_basic_plots_{subject}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def correlation_analysis(self, subject='S2'):
        """Analyze correlations between different signals"""
        print(f"\n=== Correlation Analysis for {subject} ===")
        data = self.load_subject_data(subject)
        
        if data is None:
            return
        
        # Collect all signals
        signals = {}
        
        if 'chest' in data:
            for signal_name, signal_data in data['chest'].items():
                if isinstance(signal_data, np.ndarray):
                    signals[f'chest_{signal_name}'] = signal_data
        
        if 'wrist' in data:
            for signal_name, signal_data in data['wrist'].items():
                if isinstance(signal_data, np.ndarray):
                    signals[f'wrist_{signal_name}'] = signal_data
        
        # Create correlation matrix
        if len(signals) > 1:
            # Sample data for correlation (first 10000 samples to avoid memory issues)
            sample_size = min(10000, min(len(signal) for signal in signals.values()))
            signal_df = pd.DataFrame({
                name: signal[:sample_size] for name, signal in signals.items()
            })
            
            # Calculate correlation matrix
            corr_matrix = signal_df.corr()
            
            # Plot correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.3f')
            plt.title(f'Signal Correlations - {subject}')
            plt.tight_layout()
            plt.savefig(f'wesad_correlations_{subject}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("\nCorrelation matrix:")
            print(corr_matrix)
    
    def statistical_summary(self, subject='S2'):
        """Generate statistical summary of all signals"""
        print(f"\n=== Statistical Summary for {subject} ===")
        data = self.load_subject_data(subject)
        
        if data is None:
            return
        
        stats_data = []
        
        for signal_type in ['chest', 'wrist']:
            if signal_type in data:
                for signal_name, signal_data in data[signal_type].items():
                    if isinstance(signal_data, np.ndarray):
                        stats = {
                            'Signal': f'{signal_type}_{signal_name}',
                            'Mean': np.nanmean(signal_data),
                            'Std': np.nanstd(signal_data),
                            'Min': np.nanmin(signal_data),
                            'Max': np.nanmax(signal_data),
                            'NaN_Count': np.isnan(signal_data).sum(),
                            'Inf_Count': np.isinf(signal_data).sum(),
                            'Length': len(signal_data)
                        }
                        stats_data.append(stats)
        
        stats_df = pd.DataFrame(stats_data)
        print(stats_df.to_string(index=False))
        
        return stats_df
    
    def compare_subjects(self, subjects=['S2', 'S3', 'S4']):
        """Compare data across different subjects"""
        print(f"\n=== Comparing Subjects: {subjects} ===")
        
        comparison_data = []
        
        for subject in subjects:
            data = self.load_subject_data(subject)
            if data is None:
                continue
            
            # Analyze chest ECG if available
            if 'chest' in data and 'ECG' in data['chest']:
                ecg_data = data['chest']['ECG']
                comparison_data.append({
                    'Subject': subject,
                    'ECG_Mean': np.nanmean(ecg_data),
                    'ECG_Std': np.nanstd(ecg_data),
                    'ECG_Length': len(ecg_data),
                    'ECG_NaN_Count': np.isnan(ecg_data).sum()
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            print(comp_df.to_string(index=False))
            
            # Plot comparison
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # ECG mean comparison
            axes[0].bar(comp_df['Subject'], comp_df['ECG_Mean'])
            axes[0].set_title('ECG Mean Values Across Subjects')
            axes[0].set_ylabel('Mean Amplitude')
            
            # ECG std comparison
            axes[1].bar(comp_df['Subject'], comp_df['ECG_Std'])
            axes[1].set_title('ECG Standard Deviation Across Subjects')
            axes[1].set_ylabel('Standard Deviation')
            
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
        
        # Create basic plots
        self.create_basic_plots()
        
        # Correlation analysis
        self.correlation_analysis()
        
        # Statistical summary
        self.statistical_summary()
        
        # Compare subjects
        self.compare_subjects()

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = WESADAnalyzer()
    
    # Run comprehensive analysis
    analyzer.comprehensive_analysis() 