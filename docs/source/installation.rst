Installation Guide
==================

This guide will help you install the Physiological Stress Detection package and its dependencies.

Prerequisites
------------

* Python 3.8 or higher
* pip package manager
* Git (for cloning the repository)

Installation Methods
-------------------

Standard Installation
~~~~~~~~~~~~~~~~~~~~~

Install the package directly from PyPI:

.. code-block:: bash

   pip install physiological-stress-detection

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development or to get the latest version:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/physiological-stress-detection.git
   cd physiological-stress-detection
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode
   pip install -e .

Installation with Extras
~~~~~~~~~~~~~~~~~~~~~~~~

Install with additional dependencies for development:

.. code-block:: bash

   # Install with development dependencies
   pip install -e ".[dev]"
   
   # Install with documentation dependencies
   pip install -e ".[docs]"
   
   # Install with notebook dependencies
   pip install -e ".[notebooks]"

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~

* **numpy**: Numerical computing
* **pandas**: Data manipulation and analysis
* **scikit-learn**: Machine learning algorithms
* **matplotlib**: Plotting and visualization
* **seaborn**: Statistical data visualization
* **shap**: Model interpretability
* **xgboost**: Gradient boosting
* **scipy**: Scientific computing
* **statsmodels**: Statistical modeling
* **torch**: Deep learning (optional)
* **joblib**: Parallel computing
* **tqdm**: Progress bars
* **pyyaml**: YAML configuration files
* **pathlib2**: Path manipulation

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

* **pytest**: Testing framework
* **black**: Code formatting
* **flake8**: Code linting
* **mypy**: Type checking
* **pre-commit**: Git hooks
* **sphinx**: Documentation generation

Verifying Installation
---------------------

Test that the installation was successful:

.. code-block:: python

   import src
   print("Installation successful!")

Or run the test suite:

.. code-block:: bash

   pytest tests/

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

1. **Import Errors**: Ensure you're using Python 3.8+ and have installed all dependencies
2. **CUDA Issues**: If using PyTorch, ensure CUDA drivers are properly installed
3. **Memory Issues**: For large datasets, consider using data streaming or reducing batch sizes

Getting Help
~~~~~~~~~~~

If you encounter issues:

1. Check the troubleshooting section in the README
2. Search existing issues on GitHub
3. Create a new issue with detailed error information

Next Steps
----------

After installation, see the :doc:`quickstart` guide to begin using the package. 