# Repository Structure Overview

This document provides a comprehensive overview of the repository structure and organization for the Physiological Stress Detection project.

## Directory Structure

```
physiological-stress-detection/
├── .github/                          # GitHub-specific files
│   └── workflows/                     # CI/CD workflows
│       └── ci.yml                     # Automated testing and deployment
├── docs/                             # Documentation
│   └── source/                       # Sphinx documentation source
│       ├── conf.py                   # Sphinx configuration
│       ├── index.rst                 # Main documentation index
│       └── installation.rst          # Installation guide
├── src/                              # Core source code
│   ├── __init__.py                   # Package initialization
│   ├── data.py                       # Data loading and preprocessing
│   ├── modeling.py                   # Model training and evaluation
│   ├── shap_utils.py                 # SHAP analysis utilities
│   ├── robustness.py                 # Robustness testing framework
│   ├── correlation.py                # Correlation analysis
│   └── utils.py                      # Utility functions
├── tests/                            # Test suite
│   ├── __init__.py                   # Test package initialization
│   └── test_data.py                  # Data processing tests
├── notebooks/                        # Jupyter notebooks
│   ├── eda_correlations.ipynb        # Exploratory data analysis
│   └── enhanced_analysis_demo.ipynb  # Enhanced analysis demo
├── results/                          # Generated outputs (gitignored)
│   ├── cv/                           # Cross-validation results
│   ├── shap/                         # SHAP analysis results
│   ├── robustness/                   # Robustness testing results
│   ├── corr/                         # Correlation analysis results
│   └── hparams/                      # Hyperparameter search results
├── .gitignore                        # Git ignore patterns
├── .pre-commit-config.yaml           # Pre-commit hooks configuration
├── CHANGELOG.md                      # Project changelog
├── CONTRIBUTING.md                   # Contributing guidelines
├── LICENSE                           # MIT License
├── PROJECT_SUMMARY.md                # Technical project summary
├── README.md                         # Main project documentation
├── REPOSITORY_STRUCTURE.md           # This file
├── config.yaml                       # Configuration file
├── pyproject.toml                    # Modern Python packaging configuration
├── requirements.txt                  # Core dependencies
├── requirements-dev.txt              # Development dependencies
└── setup.py                         # Package setup script
```

## Key Components

### 1. Source Code (`src/`)

The core implementation is organized into modular components:

- **`data.py`**: Data loading, preprocessing, and feature engineering
- **`modeling.py`**: Machine learning model training and evaluation
- **`shap_utils.py`**: SHAP analysis for model interpretability
- **`robustness.py`**: Robustness testing and noise injection
- **`correlation.py`**: Correlation analysis between signals
- **`utils.py`**: Utility functions and helpers

### 2. Documentation (`docs/`)

Comprehensive documentation using Sphinx:

- **`conf.py`**: Sphinx configuration with extensions
- **`index.rst`**: Main documentation landing page
- **`installation.rst`**: Detailed installation guide
- Additional documentation files for API reference and examples

### 3. Testing (`tests/`)

Comprehensive test suite with:

- **`test_data.py`**: Data processing and validation tests
- Unit tests for all major components
- Integration tests for end-to-end workflows
- Mock testing for external dependencies

### 4. Configuration Management

Multiple configuration approaches:

- **`config.yaml`**: Main configuration file for pipeline parameters
- **`pyproject.toml`**: Modern Python packaging configuration
- **`setup.py`**: Traditional package setup for compatibility

### 5. Development Tools

Professional development setup:

- **`.pre-commit-config.yaml`**: Automated code quality checks
- **`.github/workflows/ci.yml`**: CI/CD pipeline with multiple jobs
- **`requirements-dev.txt`**: Development dependencies
- **`.gitignore`**: Comprehensive ignore patterns

## File Descriptions

### Core Documentation

| File | Purpose |
|------|---------|
| `README.md` | Main project documentation and quick start guide |
| `PROJECT_SUMMARY.md` | Technical achievements and skills demonstration |
| `CHANGELOG.md` | Version history and change tracking |
| `CONTRIBUTING.md` | Guidelines for contributors |
| `LICENSE` | MIT License for open source distribution |

### Configuration Files

| File | Purpose |
|------|---------|
| `config.yaml` | Pipeline configuration and parameters |
| `pyproject.toml` | Modern Python packaging and tool configuration |
| `setup.py` | Traditional package installation |
| `requirements.txt` | Core Python dependencies |
| `requirements-dev.txt` | Development and testing dependencies |

### Development Tools

| File | Purpose |
|------|---------|
| `.pre-commit-config.yaml` | Automated code quality checks |
| `.github/workflows/ci.yml` | GitHub Actions CI/CD pipeline |
| `.gitignore` | Git ignore patterns for clean repository |

## Code Organization Principles

### 1. Modularity

- Each module has a single responsibility
- Clear separation between data processing, modeling, and analysis
- Configurable components for flexibility

### 2. Reproducibility

- Fixed random seeds for deterministic results
- Configuration-driven workflows
- Comprehensive logging and error handling

### 3. Scalability

- Memory-efficient processing for large datasets
- Modular design for easy extension
- CLI interfaces for batch processing

### 4. Quality Assurance

- Comprehensive test coverage
- Type hints throughout the codebase
- Automated code formatting and linting
- Pre-commit hooks for quality control

## Development Workflow

### 1. Local Development

```bash
# Clone and setup
git clone <repository-url>
cd physiological-stress-detection
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/
```

### 2. CI/CD Pipeline

The GitHub Actions workflow includes:

- **Testing**: Multi-Python version testing with pytest
- **Linting**: Code quality checks with flake8 and mypy
- **Formatting**: Code formatting verification with Black
- **Security**: Security scanning with bandit and safety
- **Documentation**: Automated documentation building
- **Deployment**: Package building and documentation deployment

### 3. Code Quality Standards

- **Type Hints**: 100% coverage for better maintainability
- **Docstrings**: Google-style docstrings for all public functions
- **Testing**: >90% test coverage with unit and integration tests
- **Formatting**: Black formatting with 88-character line length
- **Linting**: flake8 and mypy for code quality

## Repository Features

### 1. Professional Standards

- MIT License for open source distribution
- Comprehensive documentation with Sphinx
- Automated testing and quality checks
- Professional project structure

### 2. Machine Learning Best Practices

- Cross-validation with stratification
- Hyperparameter optimization
- Model interpretability with SHAP
- Robustness testing with noise injection
- Ensemble learning methods

### 3. Production Readiness

- Configuration management
- Error handling and logging
- Modular architecture
- Scalable design patterns
- Comprehensive testing

### 4. Educational Value

- Well-documented code with examples
- Clear project structure
- Best practices demonstration
- Comprehensive testing examples

## Future Enhancements

### Planned Structure Additions

- **`examples/`**: Additional usage examples
- **`benchmarks/`**: Performance benchmarking
- **`deployment/`**: Deployment configurations
- **`api/`**: REST API implementation
- **`web/`**: Web interface components

### Documentation Enhancements

- API reference documentation
- Tutorial notebooks
- Video demonstrations
- Case studies and applications

This repository structure demonstrates professional software engineering practices while showcasing advanced machine learning techniques and comprehensive documentation standards. 