from dataclasses import dataclass, field
from typing import OrderedDict as TypingOrderedDict

import numpy as np


@dataclass(frozen=True)
class VInformation:
    """Stores V-information metrics for a single feature.

    Attributes:
        var_z (float): Variance of the feature z.
        r2 (float): Coefficient of determination (R²) for the feature.
        value (float): Computed V-information value (var_z * r2).
    """

    var_z: float
    r2: float
    value: float = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to compute the combined V-information value."""
        object.__setattr__(self, "value", self.var_z * self.r2)


@dataclass(frozen=True)
class ComplexityMeasureK:
    """Calculates the complexity measure K for a set of V-information values.

    Attributes:
        v_infos (OrderedDict[str, VInformation]): Mapping from feature keys
            to their V-information objects.
        value (float): Computed complexity measure K = 1 - mean(V-information values).
    """

    v_infos: TypingOrderedDict[str, VInformation]
    value: float = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to compute the complexity value K from V-information."""
        mean_v_info = float(np.mean([vi.value for vi in self.v_infos.values()]))
        object.__setattr__(self, "value", 1.0 - mean_v_info)
