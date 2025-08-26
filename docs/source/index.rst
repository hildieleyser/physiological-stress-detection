Welcome to Physiological Stress Detection's documentation!
================================================================

A comprehensive machine learning framework for physiological stress detection using the WESAD (Wearable Stress and Affect Detection) dataset.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api_reference
   examples
   contributing

Features
--------

* **Advanced Data Processing**: Multi-modal signal analysis with time-series feature engineering
* **Machine Learning Models**: Multiple algorithms including ensemble methods
* **Model Interpretability**: SHAP analysis for feature importance and model explainability
* **Robustness Testing**: Comprehensive evaluation against noise and data perturbations
* **Cross-Validation**: Rigorous evaluation with repeated stratified k-fold validation
* **Hyperparameter Optimization**: Automated tuning with grid and random search

Installation
-----------

.. code-block:: bash

   pip install physiological-stress-detection

Quick Start
----------

.. code-block:: python

   from src.data import load_and_prepare_data
   from src.modeling import train_models
   
   # Load and prepare data
   X, y = load_and_prepare_data("S2")
   
   # Train models
   results = train_models(X, y)

For more detailed information, see the :doc:`quickstart` guide.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 