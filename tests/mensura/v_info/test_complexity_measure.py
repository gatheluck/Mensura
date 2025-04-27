import dataclasses
import math
from collections import OrderedDict
from typing import OrderedDict as TypingOrderedDict

import pytest

from mensura.v_info.complexity_measure import ComplexityMeasureK, VInformation


def test_vinformation_computation() -> None:
    """Test that VInformation computes `value = var_z * r2` correctly."""
    vi = VInformation(var_z=3.0, r2=0.2)
    expected_value: float = 3.0 * 0.2
    assert vi.value == pytest.approx(expected_value), (
        "VInformation.value should be var_z * r2"
    )


def test_vinformation_is_frozen() -> None:
    """Test that VInformation is immutable (frozen dataclass).
    Attempting to set an attribute should raise FrozenInstanceError.
    """
    vi = VInformation(var_z=1.5, r2=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        vi.var_z = 2.0  # type: ignore


@pytest.mark.parametrize(
    ("r2_values", "expected_mean_r2"),
    [
        ([0.0, 1.0], 0.5),
        ([0.25, 0.75, 0.5], (0.25 + 0.75 + 0.5) / 3),
        ([1.0, 1.0, 1.0], 1.0),
    ],
)
def test_complexity_measure_k_computation(
    r2_values: list[float],
    expected_mean_r2: float,
) -> None:
    """Test that ComplexityMeasureK computes `value = 1.0 - mean(VInformation.value)`
    correctly when var_z == 1 for all entries.
    """
    # Build an OrderedDict[str, VInformation] where var_z=1.0 so value == r2
    v_infos: TypingOrderedDict[str, VInformation] = OrderedDict(
        (f"feat{i}", VInformation(var_z=1.0, r2=r2)) for i, r2 in enumerate(r2_values)
    )
    cm = ComplexityMeasureK(v_infos=v_infos)
    assert cm.value == pytest.approx(1.0 - expected_mean_r2), (
        "ComplexityMeasureK.value should equal 1 - mean of VInformation.value"
    )


def test_complexity_measure_k_empty_dict_results_in_nan() -> None:
    """Test that passing an empty OrderedDict yields cm.value == nan (no exception)."""
    empty: TypingOrderedDict[str, VInformation] = OrderedDict()
    cm = ComplexityMeasureK(v_infos=empty)
    assert math.isnan(cm.value), (
        "ComplexityMeasureK.value should be nan for empty input"
    )


def test_complexity_measure_k_is_frozen() -> None:
    """Test that ComplexityMeasureK is immutable (frozen dataclass).
    Attempting to set its `value` should raise FrozenInstanceError.
    """
    single: TypingOrderedDict[str, VInformation] = OrderedDict(
        {"only": VInformation(var_z=2.0, r2=0.3)}
    )
    cm = ComplexityMeasureK(v_infos=single)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cm.value = 0.0  # type: ignore
