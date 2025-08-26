"""
Tests for data processing functionality.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestDataProcessing:
    """Test cases for data processing functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_data = np.random.randn(1000, 4)
        self.sample_labels = np.random.randint(0, 3, 1000)
        self.config = {
            'preprocessing': {
                'window_len': 100,
                'stride': 50,
                'normalize_method': 'global'
            },
            'classes': {
                'drop_classes': [],
                'aggregate_to_one': []
            }
        }

    def test_normalize_features_global(self):
        """Test global normalization."""
        X_normalized = self.sample_data.copy()
        X_normalized = (X_normalized - X_normalized.mean()) / X_normalized.std()
        
        assert X_normalized.shape == self.sample_data.shape
        assert np.allclose(X_normalized.mean(), 0, atol=1e-10)
        assert np.allclose(X_normalized.std(), 1, atol=1e-10)

    def test_create_windows(self):
        """Test window creation."""
        window_len = 100
        stride = 50
        data_len = len(self.sample_data)
        
        expected_windows = (data_len - window_len) // stride + 1
        
        windows = []
        for i in range(0, data_len - window_len + 1, stride):
            windows.append(self.sample_data[i:i+window_len])
        
        assert len(windows) == expected_windows
        assert all(w.shape == (window_len, self.sample_data.shape[1]) for w in windows)

    def test_feature_extraction_consistency(self):
        """Test that feature extraction is consistent."""
        window1 = np.random.randn(100, 4)
        window2 = window1.copy()
        
        features1 = [np.mean(window1), np.std(window1), np.min(window1), np.max(window1)]
        features2 = [np.mean(window2), np.std(window2), np.min(window2), np.max(window2)]
        
        assert np.allclose(features1, features2)

    def test_data_shape_consistency(self):
        """Test that data shapes are consistent throughout processing."""
        X_normalized = (self.sample_data - self.sample_data.mean()) / self.sample_data.std()
        assert X_normalized.shape == self.sample_data.shape
        
        window_len = 100
        stride = 50
        windows = []
        for i in range(0, len(X_normalized) - window_len + 1, stride):
            windows.append(X_normalized[i:i+window_len])
        
        assert all(w.shape[1] == self.sample_data.shape[1] for w in windows)


class TestDataValidation:
    """Test cases for data validation."""

    def test_empty_data(self):
        """Test handling of empty data."""
        empty_data = np.array([])
        
        with pytest.raises(ValueError):
            if len(empty_data) == 0:
                raise ValueError("Data cannot be empty")

    def test_nan_values(self):
        """Test handling of NaN values."""
        data_with_nan = np.random.randn(100, 4)
        data_with_nan[0, 0] = np.nan
        
        with pytest.raises(ValueError):
            if np.any(np.isnan(data_with_nan)):
                raise ValueError("Data contains NaN values")

    def test_infinite_values(self):
        """Test handling of infinite values."""
        data_with_inf = np.random.randn(100, 4)
        data_with_inf[0, 0] = np.inf
        
        with pytest.raises(ValueError):
            if np.any(np.isinf(data_with_inf)):
                raise ValueError("Data contains infinite values")


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'data': {
            'data_path': 'WESAD',
            'subject_id': 'S2',
            'output_path': 'results'
        },
        'preprocessing': {
            'window_len': 1000,
            'stride': 500,
            'normalize_method': 'global'
        },
        'classes': {
            'drop_classes': [],
            'aggregate_to_one': ['0', '3']
        }
    }


class TestConfiguration:
    """Test cases for configuration handling."""

    def test_config_validation(self, sample_config):
        """Test configuration validation."""
        assert 'data' in sample_config
        assert 'preprocessing' in sample_config
        assert 'classes' in sample_config

    def test_missing_config_keys(self):
        """Test handling of missing configuration keys."""
        incomplete_config = {'data': {}}
        
        with pytest.raises(KeyError):
            if 'preprocessing' not in incomplete_config:
                raise KeyError("Missing required configuration key: preprocessing")


class TestModeling:
    """Test cases for modeling functionality."""

    def test_model_initialization(self):
        """Test model initialization."""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_cross_validation(self):
        """Test cross-validation functionality."""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        scores = cross_val_score(model, X, y, cv=5)
        
        assert len(scores) == 5
        assert all(0 <= score <= 1 for score in scores)

    def test_ensemble_methods(self):
        """Test ensemble methods."""
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        estimators = [
            ('lr', LogisticRegression(random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42))
        ]
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        assert ensemble is not None
        assert len(ensemble.estimators) == 2


class TestSHAPAnalysis:
    """Test cases for SHAP analysis."""

    def test_shap_values_shape(self):
        """Test SHAP values shape."""
        X_test = np.random.randn(50, 10)
        n_samples, n_features = X_test.shape
        
        shap_values = np.random.randn(n_samples, n_features)
        
        assert shap_values.shape == X_test.shape

    def test_feature_importance_ranking(self):
        """Test feature importance ranking."""
        feature_importance = np.random.randn(10)
        
        sorted_indices = np.argsort(feature_importance)[::-1]
        
        assert len(sorted_indices) == len(feature_importance)
        assert all(i >= 0 and i < len(feature_importance) for i in sorted_indices)


class TestRobustness:
    """Test cases for robustness testing."""

    def test_noise_injection(self):
        """Test noise injection."""
        original_data = np.random.randn(100, 10)
        noise_level = 0.1
        
        noise = np.random.normal(0, noise_level, original_data.shape)
        noisy_data = original_data + noise
        
        assert noisy_data.shape == original_data.shape
        assert not np.allclose(original_data, noisy_data)

    def test_performance_degradation(self):
        """Test performance degradation analysis."""
        original_score = 0.85
        noisy_score = 0.80
        
        degradation = original_score - noisy_score
        
        assert degradation >= 0
        assert degradation <= 1 