from typing import Iterable, cast

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from timm.models.resnetv2 import ResNetV2
from torchvision.models.feature_extraction import create_feature_extractor


def compute_approximate_v_information(
    intermediate_feature: np.ndarray, z: np.ndarray
) -> float:
    """Compute approximate v-information for a specific atom by using an assumption that
    predictive family V as a class of linear probes with Gaussian prior.

    For details, please check "C Complexity measure" section of the original paper.

    Args:
        intermediate_feature (np.ndarray): shape (B, C)
            An intermediate feature representation of a specific layer.

        z (np.ndarray): shape (B,)
            A final feature representation. In the original paper, these are
            coefficients of a specific atom.

    Returns:
        float: An estimated value of v-information.

    """
    assert intermediate_feature.ndim == 2
    assert z.ndim == 1

    # specify ddof=1 to use unbiased variance.
    var_z = np.var(z, ddof=1)

    # estimate the coefficient of determination (R^2).
    model = LinearRegression().fit(intermediate_feature, z)
    r2 = model.score(intermediate_feature, z)

    return float((var_z * r2))


def compute_complexity_k(  # noqa: N802
    intermediate_features: Iterable[np.ndarray],
    z: np.ndarray,
) -> float:
    """Estimate the complexity measure K for a specific atom.

    In the original paper, this is defined as K(z,x) in equation (2).

    Args:
        intermediate_features (list[np.ndarray]):
            A list of intermediate features. Each element must have shape (B, C).

        z (np.ndarray): shape (B,)
            Coefficients of a specific atom.

    Returns:
        float: An estimated value of complexity K.
            A higher value implies that the feature z is readily
            accessible and persists throughout the model layers.

    """
    v_informations = list()

    # TODO: use multiprocessing for parallel computation.
    for intermediate_feature in intermediate_features:
        v_informations.append(
            compute_approximate_v_information(intermediate_feature, z)
        )

    return 1.0 - float(np.mean(v_informations))


def get_feature_extractor(model: torch.nn.Module) -> torch.nn.Module:
    """A utility function to get a feature extractor for a given model.

    Args:
        model (torch.nn.Module): A model to extract features from.

    Returns:
        torch.nn.Module: A feature extractor.

    """
    if isinstance(model, ResNetV2):
        return_nodes = {
            "stages.0": "intermediate.0",
            "stages.1": "intermediate.1",
            "stages.2": "intermediate.2",
            "norm": "penultimate",
        }
    else:
        raise NotImplementedError(f"Model type `{type(model)}` is not supported.")

    return cast(
        torch.nn.Module, create_feature_extractor(model, return_nodes=return_nodes)
    )
