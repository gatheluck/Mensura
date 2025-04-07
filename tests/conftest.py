import pytest


@pytest.fixture
def dimensions():
    """Fixture that provides common dimensions for tests.
    num_sample: number of samples, num_feature: number of features.
    """
    num_sample = 100  # Number of samples
    num_feature = 3  # Number of features
    return num_sample, num_feature
