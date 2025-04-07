from typing import Iterable

import numpy as np
from sklearn.linear_model import LinearRegression


def compute_approximate_v_information(feature: np.ndarray, z: np.ndarray) -> float:
    """Compute approximate v-information for a specific atom by using an assumption that
    predictive family V as a class of linear probes with Gaussian prior.

    For details, please check "C Complexity measure" section of the original paper.

    Args:
        feature (np.ndarray): shape (B, C)

        z (np.ndarray): shape (B,)
            Coefficients of a specific atom.

    Returns:
        float: An estimated value of v-information.

    """
    assert feature.ndim == 2
    assert z.ndim == 1
    assert np.all(z >= 0)

    # specify ddof=1 to use unbiased variance.
    var_z = np.var(z, ddof=1)

    # estimate the coefficient of determination (R^2).
    model = LinearRegression().fit(feature, z)
    r2 = model.score(feature, z)

    return float((var_z * r2))


def compute_complexity_k(  # noqa: N802
    layer_outputs: Iterable[np.ndarray],
    z: np.ndarray,
) -> float:
    """Estimate the complexity measure K for a specific atom.

    In the original paper, this is defined as K(z,x) in equation (2).

    Args:
        layer_outputs (list[np.ndarray]):
            A list of layer outputs. Each element must have shape (B, C).

        z (np.ndarray): shape (B,)
            Coefficients of a specific atom.

    Returns:
        float: An estimated value of complexity K.
            A higher value implies that the feature z is readily
            accessible and persists throughout the model layers.

    """
    v_informations = list()

    # TODO: use multiprocessing for parallel computation.
    for layer_output in layer_outputs:
        v_informations.append(compute_approximate_v_information(layer_output, z))

    return 1.0 - float(np.mean(v_informations))
