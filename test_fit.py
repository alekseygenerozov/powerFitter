import numpy as np
import pytest

from power_law_fit import (
    fit_power,
    fit_power_with_error,
    validate_and_clean,
    sample_power_law,
)


def test_power_fit():
    """Testing fit for different power-law indices"""
    test_pows = np.arange(-2, 2.01, 0.5)
    for test_pow in test_pows:
        obs = [sample_power_law(test_pow, 0.1, 1) for ii in range(4000)]
        obs = np.array(obs)
        bf, bferr = fit_power_with_error(obs)

        assert np.abs(bf - test_pow) < 5 * bferr


def test_fit_power_raises_value_error():
    """Test input validation"""
    # Create an array with non-positive values
    invalid_data = np.array([1.0, 2.0, 0.0, 3.0, -1.0])

    # Use pytest.raises to check if ValueError is raised
    with pytest.raises(
        ValueError, match="Observed data must be positive, real numbers."
    ):
        fit_power_with_error(invalid_data)


def test_fit_power_raises_value_error_2():
    """Testing input validation"""
    # Create an array with non-positive values
    invalid_data = np.array([1.0, 2.0, 0.0, 3.0, "Hippo"])

    # Use pytest.raises to check if ValueError is raised
    with pytest.raises(
        ValueError, match="Observed data must be positive, real numbers."
    ):
        fit_power_with_error(invalid_data)


def test_min_max_value_1():
    """Testing input validation"""
    invalid_data = np.array([1.0, 2.0, 4.0, 3.0, 5.0])

    # Use pytest.raises to check if ValueError is raised
    with pytest.raises(
        ValueError,
        match="min_value must be less than or equal to max_value, and both must be postive.",
    ):
        fit_power_with_error(invalid_data, min_value=10)


def test_min_max_value_2():
    """Testing input validation"""
    invalid_data = np.array([1.0, 2.0, 4.0, 3.0, 5.0])

    # Use pytest.raises to check if ValueError is raised
    with pytest.raises(ValueError, match="Too few points in domain for fitting."):
        fit_power_with_error(invalid_data, min_value=10, max_value=11)


def test_validate_and_clean():
    """Testing data filtering"""
    data = np.array([1.0, 2.0, 4.0, 3.0, 5.0])

    data, min_value, max_value = validate_and_clean(data, min_value=2)
    assert np.all(data == np.array([2.0, 4.0, 3.0, 5.0]))
    assert min_value == 2.0
    assert max_value == 5.0
