import numpy as np

from mensura.v_information import compute_approximate_v_information


class TestComputeApproximateVInformation:
    """Test suite for the compute_approximate_v_information function."""

    def test_r2_one(self, dimensions):
        """Test case where the coefficient of determination (R^2) is 1.
        To ensure z is non-negative, we generate features using np.random.rand
        (which produces values in [0, 1)) and use a beta vector with only non-negative values.
        In this case, the computed v-information should equal the unbiased variance of z.
        """
        num_sample, num_feature = dimensions
        np.random.seed(42)
        # Using np.random.rand to generate non-negative feature values
        feature = np.random.rand(num_sample, num_feature)
        beta = np.array([2.0, 3.0, 5.0])  # Non-negative coefficients
        z = feature.dot(beta)

        v_info = compute_approximate_v_information(feature, z)
        expected = np.var(z, ddof=1)
        assert np.allclose(v_info, expected, rtol=1e-5)

    def test_r2_zero(self, dimensions):
        """Test case where the coefficient of determination (R^2) is 0.
        This is done by using a constant feature matrix, which forces the linear regression
        model to always predict the mean of z, leading to an R^2 of 0. Therefore, the v-information
        should also be 0.
        """
        num_sample, num_feature = dimensions
        feature = np.ones((num_sample, num_feature))  # Constant feature matrix
        z = np.linspace(0, 1, num_sample)  # Non-constant target values

        v_info = compute_approximate_v_information(feature, z)
        assert np.allclose(v_info, 0.0, rtol=1e-5)
