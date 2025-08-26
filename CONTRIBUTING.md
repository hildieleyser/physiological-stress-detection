# Contributing to Physiological Stress Detection

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Physiological Stress Detection machine learning pipeline.

## Code of Conduct

This project is committed to providing a welcoming and inclusive environment for all contributors. We expect all contributors to:

- Be respectful and considerate of others
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the problem
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, package versions)
- Any relevant error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! When suggesting new features:

- Use a clear and descriptive title
- Provide a detailed description of the proposed functionality
- Explain why this enhancement would be useful
- Include any relevant research or references

### Pull Requests

We welcome pull requests! Please follow these guidelines:

#### Before Submitting a Pull Request

1. **Fork the repository** and create a feature branch
2. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8 mypy
   ```
3. **Run tests** to ensure everything works:
   ```bash
   pytest tests/
   ```
4. **Format your code**:
   ```bash
   black src/ tests/
   ```
5. **Check code quality**:
   ```bash
   flake8 src/ tests/
   mypy src/
   ```

#### Pull Request Guidelines

- **Keep changes focused**: Each PR should address a single issue or feature
- **Write clear commit messages**: Use conventional commit format
- **Add tests**: Include tests for new functionality
- **Update documentation**: Update README, docstrings, and other docs as needed
- **Follow the existing code style**: Use the same formatting and conventions

#### Code Style Guidelines

- **Python**: Follow PEP 8 with line length of 88 characters (Black default)
- **Type hints**: Use type hints for all function parameters and return values
- **Docstrings**: Use Google-style docstrings for all public functions and classes
- **Imports**: Group imports (standard library, third-party, local) with blank lines

Example:
```python
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from .utils import load_config
```

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/physiological-stress-detection.git
   cd physiological-stress-detection
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

### Testing

We use pytest for testing. To run tests:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_data.py

# Run tests in parallel
pytest -n auto
```

### Documentation

When adding new features, please update:

1. **Docstrings**: Add comprehensive docstrings to all new functions and classes
2. **README.md**: Update if adding new major features
3. **Configuration**: Update `config.yaml` examples if adding new parameters
4. **Examples**: Add usage examples in the notebooks directory

### Release Process

For maintainers, the release process involves:

1. **Version bump**: Update version in `__init__.py` and `setup.py`
2. **Changelog**: Update `CHANGELOG.md` with new features and fixes
3. **Tag release**: Create a git tag for the release
4. **Publish**: Push to PyPI if applicable

## Questions or Need Help?

If you have questions or need help:

1. Check the existing documentation
2. Search existing issues
3. Create a new issue with the "question" label
4. Join our community discussions

## Recognition

Contributors will be recognized in:

- The project README
- Release notes
- Contributor statistics

Thank you for contributing to making this project better! 