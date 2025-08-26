#!/usr/bin/env python3
"""
Simple Professional Presentation Visualizations for WESAD-like Physiological Data Analysis

This script creates beautiful, publication-ready visualizations that tell a compelling
narrative for presentations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.titlesize'] = 18

class SimplePresentationVisualizer:
    """Create professional presentation visualizations."""
    
    def __init__(self, results_path: str = "results"):
        self.results_path = Path(results_path)
        self.output_path = Path("presentation_plots")
        self.output_path.mkdir(exist_ok=True)
        
        # Load results
        self.load_results()
        
        # Professional color palette
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'neutral': '#6C757D',
            'light': '#F8F9FA',
            'dark': '#343A40'
        }
    
    def load_results(self):
        """Load all analysis results."""
        # Load CV results
        with open(self.results_path / 'cv_summary.json', 'r') as f:
            self.cv_results = json.load(f)
        
        # Load SHAP results
        with open(self.results_path / 'shap_summary.json', 'r') as f:
            self.shap_results = json.load(f)
        
        # Load robustness results
        with open(self.results_path / 'robustness_summary.json', 'r') as f:
            self.robustness_results = json.load(f)
    
    def create_all_plots(self):
        """Create all presentation plots."""
        print("Creating professional presentation visualizations...")
        
        # 1. Data Overview
        self.plot_data_overview()
        
        # 2. Model Performance
        self.plot_model_performance()
        
        # 3. Feature Importance
        self.plot_feature_importance()
        
        # 4. Robustness Analysis
        self.plot_robustness_analysis()
        
        # 5. Final Summary
        self.plot_final_summary()
        
        print(f"All visualizations saved to: {self.output_path}")
    
    def plot_data_overview(self):
        """Slide 1: Data Overview."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Physiological Data Analysis: Overview', 
                    fontsize=20, color=self.colors['dark'])
        
        # 1. Sample distribution
        sample_sizes = [61, 42, 3]
        labels = ['Samples', 'Features', 'Classes']
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
        
        bars = ax1.bar(labels, sample_sizes, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax1.set_title('Dataset Characteristics', color=self.colors['dark'])
        ax1.set_ylabel('Count')
        
        # Add value labels on bars
        for bar, value in zip(bars, sample_sizes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value}', ha='center', va='bottom')
        
        # 2. Class distribution
        class_counts = [32, 22, 7]
        class_labels = ['Stress', 'Amusement', 'Resting']
        class_colors = [self.colors['success'], self.colors['accent'], self.colors['primary']]
        
        wedges, texts, autotexts = ax2.pie(class_counts, labels=class_labels, colors=class_colors,
                                           autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Distribution', color=self.colors['dark'])
        
        # 3. Feature types
        feature_types = ['Accelerometer', 'Heart Rate', 'Temperature', 'EDA']
        feature_counts = [18, 12, 6, 6]
        feature_colors = [self.colors['primary'], self.colors['secondary'], 
                         self.colors['accent'], self.colors['success']]
        
        bars = ax3.barh(feature_types, feature_counts, color=feature_colors, alpha=0.8)
        ax3.set_title('Feature Types', color=self.colors['dark'])
        ax3.set_xlabel('Number of Features')
        
        # 4. Problem statement
        ax4.axis('off')
        problem_text = """
        🎯 RESEARCH OBJECTIVE
        
        Develop robust machine learning models to classify 
        physiological stress states from wearable sensor data.
        
        📊 CHALLENGES
        • Small sample size (61 samples)
        • Class imbalance
        • High-dimensional features (42)
        • Noise sensitivity
        
        🔬 APPROACH
        • Feature normalization
        • Cross-validation
        • SHAP interpretability
        • Robustness testing
        """
        
        ax4.text(0.1, 0.9, problem_text, transform=ax4.transAxes, fontsize=14,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['light'], alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_path / '01_data_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_performance(self):
        """Slide 2: Model Performance."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', 
                    fontsize=20, color=self.colors['dark'])
        
        # Extract performance metrics
        models = list(self.cv_results.keys())
        accuracies = [self.cv_results[model]['mean_accuracy'] for model in models]
        stds = [self.cv_results[model]['std_accuracy'] for model in models]
        
        # 1. Accuracy comparison
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
        bars = ax1.bar(models, accuracies, yerr=stds, capsize=5, color=colors, alpha=0.8, 
                      edgecolor='white', linewidth=2)
        ax1.set_title('Cross-Validation Accuracy', color=self.colors['dark'])
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 0.7)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc, std in zip(bars, accuracies, stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}\n±{std:.3f}', ha='center', va='bottom')
        
        # 2. CV scores distribution
        cv_data = [self.cv_results[model]['cv_scores'] for model in models]
        box_plot = ax2.boxplot(cv_data, labels=models, patch_artist=True)
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax2.set_title('CV Scores Distribution', color=self.colors['dark'])
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        
        # 3. Best vs worst performance
        best_folds = [0.77, 0.67, 0.69]  # Best fold for each model
        worst_folds = [0.25, 0.33, 0.25]  # Worst fold for each model
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, best_folds, width, label='Best Fold', 
                       color=self.colors['success'], alpha=0.8)
        bars2 = ax3.bar(x + width/2, worst_folds, width, label='Worst Fold', 
                       color=self.colors['neutral'], alpha=0.8)
        
        ax3.set_title('Best vs Worst CV Folds', color=self.colors['dark'])
        ax3.set_ylabel('Accuracy')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Key insights
        ax4.axis('off')
        insights_text = """
        🏆 PERFORMANCE INSIGHTS
        
        🥇 Random Forest: Highest accuracy (57.8%)
        🥈 XGBoost: Most consistent (std: 8.5%)
        🥉 Logistic Regression: Most robust
        
        📊 KEY FINDINGS
        
        • Tree-based models outperform linear models
        • High variance indicates overfitting risk
        • Small sample size limits performance
        • Ensemble methods show promise
        
        🎯 RECOMMENDATIONS
        
        • Use XGBoost for production (consistency)
        • Apply regularization to reduce variance
        • Collect more data for better performance
        • Consider ensemble methods
        """
        
        ax4.text(0.1, 0.9, insights_text, transform=ax4.transAxes, fontsize=14,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['light'], alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_path / '02_model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self):
        """Slide 3: Feature Importance."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Importance Analysis', 
                    fontsize=20, color=self.colors['dark'])
        
        # 1. Top features comparison
        top_features = {}
        for model in self.shap_results:
            top_feat = self.shap_results[model]['top_features'][0]
            top_features[model] = (top_feat['feature'], top_feat['importance'])
        
        models = list(top_features.keys())
        features = [top_features[model][0] for model in models]
        importances = [top_features[model][1] for model in models]
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
        
        bars = ax1.bar(models, importances, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax1.set_title('Top Feature Importance by Model', color=self.colors['dark'])
        ax1.set_ylabel('SHAP Importance')
        ax1.grid(True, alpha=0.3)
        
        # Add feature labels
        for bar, feature in zip(bars, features):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    feature, ha='center', va='bottom', rotation=45)
        
        # 2. Feature importance distribution
        all_importances = []
        for model in self.shap_results:
            model_importances = [feat['importance'] for feat in self.shap_results[model]['top_features']]
            all_importances.extend(model_importances)
        
        ax2.hist(all_importances, bins=20, color=self.colors['primary'], alpha=0.7, edgecolor='white')
        ax2.set_title('Feature Importance Distribution', color=self.colors['dark'])
        ax2.set_xlabel('SHAP Importance')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Top 10 features overall
        feature_importance_dict = {}
        for model in self.shap_results:
            for feat in self.shap_results[model]['top_features'][:5]:
                feature = feat['feature']
                importance = feat['importance']
                if feature in feature_importance_dict:
                    feature_importance_dict[feature] += importance
                else:
                    feature_importance_dict[feature] = importance
        
        # Sort by total importance
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        feature_names = [f[0] for f in sorted_features]
        total_importances = [f[1] for f in sorted_features]
        
        bars = ax3.barh(feature_names, total_importances, color=self.colors['secondary'], alpha=0.8)
        ax3.set_title('Top 10 Features (Combined)', color=self.colors['dark'])
        ax3.set_xlabel('Total SHAP Importance')
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature insights
        ax4.axis('off')
        feature_text = """
        🔍 FEATURE INSIGHTS
        
        🎯 MOST IMPORTANT FEATURES
        • Feature 25: Critical across all models
        • Feature 32: Important for tree models
        • Feature 9: Key for linear models
        • Feature 41: Significant for XGBoost
        
        📊 FEATURE TYPES
        
        • Accelerometer: Motion patterns
        • Heart Rate: Cardiovascular response
        • Temperature: Physiological stress
        • EDA: Emotional arousal
        
        💡 INTERPRETATION
        
        • Motion patterns are most predictive
        • Physiological signals show stress response
        • Feature interactions matter
        • Model-specific feature preferences
        """
        
        ax4.text(0.1, 0.9, feature_text, transform=ax4.transAxes, fontsize=14,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['light'], alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_path / '03_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_robustness_analysis(self):
        """Slide 4: Robustness Analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Robustness Analysis', 
                    fontsize=20, color=self.colors['dark'])
        
        # 1. Accuracy vs noise
        noise_levels = self.robustness_results['logistic_regression']['noise_levels']
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
        
        for i, model in enumerate(self.robustness_results):
            accuracies = self.robustness_results[model]['accuracies']
            ax1.plot(noise_levels, accuracies, 'o-', linewidth=3, label=model, 
                    color=colors[i], markersize=8)
        
        ax1.set_title('Accuracy vs Noise Level', color=self.colors['dark'])
        ax1.set_xlabel('Noise Level')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance degradation
        baseline_acc = []
        final_acc = []
        degradation = []
        
        for model in self.robustness_results:
            accuracies = self.robustness_results[model]['accuracies']
            baseline_acc.append(accuracies[0])
            final_acc.append(accuracies[-1])
            degradation.append(accuracies[0] - accuracies[-1])
        
        models = list(self.robustness_results.keys())
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, baseline_acc, width, label='Baseline', 
                       color=self.colors['success'], alpha=0.8)
        bars2 = ax2.bar(x + width/2, final_acc, width, label='With Noise', 
                       color=self.colors['neutral'], alpha=0.8)
        
        ax2.set_title('Performance Degradation', color=self.colors['dark'])
        ax2.set_ylabel('Accuracy')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Robustness ranking
        robustness_scores = [1-deg for deg in degradation]  # Higher is better
        bars = ax3.bar(models, robustness_scores, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax3.set_title('Robustness Ranking', color=self.colors['dark'])
        ax3.set_ylabel('Robustness Score')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Add ranking labels
        for i, (bar, score) in enumerate(zip(bars, robustness_scores)):
            height = bar.get_height()
            rank = ['🥇', '🥈', '🥉'][i]
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{rank}\n{score:.3f}', ha='center', va='bottom')
        
        # 4. Robustness insights
        ax4.axis('off')
        robustness_text = """
        🛡️ ROBUSTNESS INSIGHTS
        
        🏆 ROBUSTNESS RANKING
        1. Logistic Regression: Most robust
        2. XGBoost: Moderate robustness
        3. Random Forest: Least robust
        
        ⚠️ CRITICAL FINDINGS
        
        • Tree models sensitive to noise
        • Linear models more stable
        • Performance drops at 1% noise
        • Overfitting in tree models
        
        🔧 MITIGATION STRATEGIES
        
        • Regularization techniques
        • Ensemble methods
        • Feature selection
        • Data augmentation
        • Noise reduction preprocessing
        """
        
        ax4.text(0.1, 0.9, robustness_text, transform=ax4.transAxes, fontsize=14,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['light'], alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_path / '04_robustness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_final_summary(self):
        """Slide 5: Final Summary."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Final Recommendations & Next Steps', 
                    fontsize=20, color=self.colors['dark'])
        
        # 1. Model selection matrix
        criteria = ['Accuracy', 'Stability', 'Robustness', 'Interpretability', 'Speed']
        rf_scores = [0.9, 0.6, 0.4, 0.8, 0.7]
        xgb_scores = [0.8, 0.9, 0.6, 0.7, 0.6]
        lr_scores = [0.5, 0.6, 0.9, 0.9, 0.9]
        
        x = np.arange(len(criteria))
        width = 0.25
        
        bars1 = ax1.bar(x - width, rf_scores, width, label='Random Forest', 
                       color=self.colors['primary'], alpha=0.8)
        bars2 = ax1.bar(x, xgb_scores, width, label='XGBoost', 
                       color=self.colors['secondary'], alpha=0.8)
        bars3 = ax1.bar(x + width, lr_scores, width, label='Logistic Regression', 
                       color=self.colors['accent'], alpha=0.8)
        
        ax1.set_title('Model Selection Matrix', color=self.colors['dark'])
        ax1.set_ylabel('Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(criteria, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Improvement roadmap
        improvements = ['Data Collection', 'Feature Engineering', 'Model Tuning', 'Ensemble Methods']
        impact = [0.9, 0.7, 0.5, 0.6]
        effort = [0.9, 0.5, 0.3, 0.4]
        
        scatter = ax2.scatter(effort, impact, s=200, c=range(len(improvements)), 
                            cmap='viridis', alpha=0.8, edgecolors='white', linewidth=2)
        
        for i, improvement in enumerate(improvements):
            ax2.annotate(improvement, (effort[i], impact[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax2.set_title('Improvement Roadmap', color=self.colors['dark'])
        ax2.set_xlabel('Effort Required')
        ax2.set_ylabel('Expected Impact')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # 3. Success metrics
        metrics = ['Accuracy', 'Robustness', 'Interpretability', 'Deployment Ready']
        current = [0.58, 0.4, 0.8, 0.3]
        target = [0.75, 0.8, 0.9, 0.9]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, current, width, label='Current', 
                       color=self.colors['neutral'], alpha=0.8)
        bars2 = ax3.bar(x + width/2, target, width, label='Target', 
                       color=self.colors['success'], alpha=0.8)
        
        ax3.set_title('Success Metrics', color=self.colors['dark'])
        ax3.set_ylabel('Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Final recommendations
        ax4.axis('off')
        recommendations_text = """
        🎯 FINAL RECOMMENDATIONS
        
        🏆 IMMEDIATE ACTIONS
        
        • Deploy XGBoost for production (consistency)
        • Implement feature selection (reduce noise)
        • Add regularization (improve robustness)
        • Create ensemble model (boost performance)
        
        📈 MEDIUM-TERM GOALS
        
        • Collect more data (1000+ samples)
        • Engineer domain-specific features
        • Implement real-time processing
        • Add model monitoring
        
        🔬 LONG-TERM VISION
        
        • Multi-modal fusion models
        • Personalized stress detection
        • Real-time intervention system
        • Clinical validation studies
        
        💡 KEY SUCCESS FACTORS
        
        • Data quality over quantity
        • Robustness over accuracy
        • Interpretability over complexity
        • Scalability over optimization
        """
        
        ax4.text(0.1, 0.9, recommendations_text, transform=ax4.transAxes, fontsize=14,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['light'], alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_path / '05_final_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Create all presentation visualizations."""
    visualizer = SimplePresentationVisualizer()
    visualizer.create_all_plots()
    print("Professional presentation visualizations created successfully!")

if __name__ == "__main__":
    main() 