# Physiological Stress Detection: Advanced Machine Learning Pipeline

A comprehensive machine learning framework for physiological stress detection using the WESAD (Wearable Stress and Affect Detection) dataset. This project demonstrates advanced data science techniques including time-series analysis, ensemble methods, interpretability analysis, and robust model evaluation.

## Project Overview

This repository contains a production-ready machine learning pipeline for analyzing physiological signals to detect stress states. The project implements state-of-the-art techniques in:

- **Time-Series Feature Engineering**: Advanced signal processing and statistical feature extraction
- **Ensemble Learning**: Multiple model architectures with sophisticated ensemble strategies
- **Model Interpretability**: SHAP analysis for feature importance and model explainability
- **Robustness Analysis**: Comprehensive testing against noise and data perturbations
- **Cross-Validation**: Rigorous evaluation with repeated stratified k-fold validation
- **Hyperparameter Optimization**: Automated hyperparameter tuning with grid and random search

## Key Features

### Advanced Data Processing
- **Multi-modal Signal Analysis**: Processing of ACC (accelerometer), BVP (blood volume pulse), EDA (electrodermal activity), and TEMP (temperature) signals
- **Time-Series Windowing**: Configurable sliding window approach with overlap control
- **Feature Engineering**: Statistical features including mean, std, min, max, percentiles, and frequency domain features
- **Data Normalization**: Global and per-subject normalization strategies

### Machine Learning Models
- **Logistic Regression**: Baseline linear model with regularization
- **Random Forest**: Ensemble of decision trees with feature importance analysis
- **XGBoost**: Gradient boosting with advanced optimization
- **Neural Networks**: Multi-layer perceptron for complex pattern recognition
- **Ensemble Methods**: Weighted voting and stacking approaches

### Model Evaluation & Interpretability
- **SHAP Analysis**: Global and local feature importance with stability analysis
- **Cross-Validation**: Repeated stratified k-fold with comprehensive metrics
- **Robustness Testing**: Performance evaluation under noise injection
- **Statistical Significance**: Univariate feature screening and correlation analysis

## Project Structure

```
├── src/                          # Core source code
│   ├── data.py                   # Data loading and preprocessing
│   ├── modeling.py               # Model training and evaluation
│   ├── shap_utils.py             # SHAP analysis utilities
│   ├── robustness.py             # Robustness testing framework
│   ├── correlation.py            # Correlation analysis
│   └── utils.py                  # Utility functions
├── notebooks/                    # Jupyter notebooks for analysis
├── results/                      # Generated outputs and models
├── config.yaml                   # Configuration management
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Technical Implementation

### Data Pipeline Architecture
The project implements a modular data processing pipeline that handles:
- Multi-subject physiological data from wearable sensors
- Time-series segmentation with configurable window parameters
- Feature extraction from raw sensor signals
- Class balancing and data augmentation techniques

### Machine Learning Framework
The modeling framework includes:
- Automated hyperparameter optimization using scikit-learn
- Cross-validation with stratification for imbalanced classes
- Model persistence and loading for production deployment
- Comprehensive evaluation metrics (accuracy, F1-score, ROC-AUC)

### Interpretability Framework
Advanced interpretability features:
- SHAP (SHapley Additive exPlanations) for model explainability
- Feature importance ranking with cross-fold stability analysis
- Statistical significance testing for feature selection
- Correlation analysis between physiological signals

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/physiological-stress-detection.git
cd physiological-stress-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

1. **Configure the pipeline**:
   ```bash
   # Edit configuration file
   nano config.yaml
   ```

2. **Prepare data**:
   ```bash
   python -m src.data --config config.yaml --subject S2
   ```

3. **Train models**:
   ```bash
   python -m src.modeling --config config.yaml --data-path results
   ```

4. **Analyze results**:
   ```bash
   python -m src.shap_utils --config config.yaml --data-path results
   python -m src.robustness --config config.yaml --data-path results
   ```

### Advanced Usage

For comprehensive analysis with all features:

```bash
python enhanced_ml_analysis.py
```

This script runs the complete pipeline including:
- Data preprocessing and normalization
- Model training with cross-validation
- SHAP analysis for interpretability
- Robustness testing with noise injection
- Ensemble method evaluation

## Configuration

The `config.yaml` file allows fine-tuning of all pipeline parameters:

```yaml
# Data preprocessing
preprocessing:
  window_len: 1000    # Time series window length
  stride: 500         # Stride between windows
  normalize_method: "global"  # Normalization strategy

# Model training
cv:
  type: "repeated"    # Cross-validation type
  n_splits: 5         # Number of folds
  n_repeats: 3        # Number of repetitions

# Analysis settings
shap:
  top_k_features: 10  # Top features for analysis
  max_samples: 100    # SHAP computation samples
```

## Results and Outputs

The pipeline generates comprehensive outputs:

### Model Performance
- Cross-validation results with confidence intervals
- Confusion matrices and ROC curves
- Feature importance rankings
- Ensemble performance metrics

### Interpretability Analysis
- SHAP summary plots and dependence plots
- Feature importance stability across folds
- Statistical significance testing results
- Correlation analysis between signals

### Robustness Evaluation
- Performance degradation under noise
- Model stability analysis
- Sensitivity to data perturbations

## Technical Highlights

### Advanced Feature Engineering
The pipeline implements sophisticated feature extraction techniques:
- Statistical moments (mean, variance, skewness, kurtosis)
- Frequency domain features (FFT coefficients, power spectral density)
- Time-domain features (zero-crossing rate, peak detection)
- Cross-signal correlation features

### Ensemble Learning Strategy
Multiple ensemble approaches are implemented:
- **Bagging**: Random Forest with bootstrap sampling
- **Boosting**: XGBoost with gradient-based optimization
- **Stacking**: Meta-learning with cross-validation
- **Voting**: Weighted combination of base models

### Robustness Framework
Comprehensive testing framework includes:
- **Noise Injection**: Gaussian and uniform noise at varying levels
- **Feature Perturbation**: Random feature masking and scaling
- **Data Augmentation**: Synthetic sample generation
- **Adversarial Testing**: Worst-case scenario analysis

## Performance Metrics

The pipeline evaluates models using multiple metrics:
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Ranking**: ROC-AUC, PR-AUC
- **Calibration**: Brier Score, Reliability Diagrams
- **Robustness**: Performance degradation curves

## Contributing

This project follows best practices for machine learning development:

1. **Code Quality**: Type hints, comprehensive documentation, and unit tests
2. **Reproducibility**: Fixed random seeds, version-controlled dependencies
3. **Modularity**: Clean separation of concerns with configurable components
4. **Scalability**: Memory-efficient processing for large datasets

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{physiological_stress_detection,
  title={Physiological Stress Detection: Advanced Machine Learning Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/physiological-stress-detection}
}
```

## Contact

For questions, issues, or collaboration opportunities:
- Create an issue on GitHub
- Review the documentation in the `docs/` directory
- Check the example notebooks for usage patterns

---

**Note**: This project demonstrates advanced machine learning techniques for physiological data analysis. The modular architecture allows easy adaptation to other time-series classification tasks. 