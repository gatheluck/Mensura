import numpy as np
import pytest

from mensura.v_info.regression import ridge_regression


def test_perfect_linear_relation() -> None:
    """Test perfect linear data yields R² of 1.0 when no regularization is used.

    Creates y = 3x + 1 exactly and calls ridge_regression with alpha=0.0
    (ordinary least squares). Asserts that the computed R² equals 1.0 within
    floating-point tolerance.
    """
    x: np.ndarray = np.linspace(0, 1, 100).reshape(-1, 1)
    y: np.ndarray = 3 * x.flatten() + 1
    score: float = ridge_regression(x, y, test_size=0.2, random_state=42, alpha=0.0)
    assert score == pytest.approx(1.0, rel=1e-6)  # rel tolerance handles FP error


def test_use_cv_reproducibility() -> None:
    """Test that enabling cross-validation produces reproducible results.

    With the same random_state and data, calling
    ridge_regression twice with use_cv=True should yield identical scores.
    """
    rng = np.random.RandomState(0)
    x: np.ndarray = rng.randn(100, 1)
    y: np.ndarray = 2 * x.flatten() + rng.randn(100) * 1e-6
    score1: float = ridge_regression(
        x, y, test_size=0.2, random_state=0, use_cv=True, cv_folds=3
    )
    score2: float = ridge_regression(
        x, y, test_size=0.2, random_state=0, use_cv=True, cv_folds=3
    )
    assert score1 == pytest.approx(score2)


def test_default_parameters_run() -> None:
    """Test that the function runs with default parameters and returns a float.

    Ensures that, when using defaults, the returned R² is within [-1, 1].
    """
    rng = np.random.RandomState(1)
    x: np.ndarray = rng.randn(50, 5)
    y: np.ndarray = x.dot(np.arange(5)) + rng.randn(50) * 1e-3
    score: float = ridge_regression(x, y)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_invalid_shapes_raises_value_error() -> None:
    """Test that mismatched sample sizes between x and y raise ValueError.

    Passing x with 10 samples and y with 11 should trigger train_test_split error.
    """
    x: np.ndarray = np.zeros((10, 2))
    y: np.ndarray = np.zeros(11)
    with pytest.raises(ValueError):
        ridge_regression(x, y)


def test_invalid_test_size_raises_value_error() -> None:
    """Test that out-of-range test_size values raise ValueError.

    A test_size >1.0 or <0.0 should be invalid.
    """
    x: np.ndarray = np.zeros((10, 2))
    y: np.ndarray = np.zeros(10)
    with pytest.raises(ValueError):
        ridge_regression(x, y, test_size=1.5)
