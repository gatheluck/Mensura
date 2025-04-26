import pathlib

import pytest
import torch
from PIL import Image
from torchvision.transforms import Compose

from mensura.v_info.feature import build_feature_extractor


def test_returns_correct_types() -> None:
    """Test that build_feature_extractor returns a Module and a Compose transform.

    Uses a ResNet18 backbone (pretrained=False) and a single node name
    to verify that the returned feature extractor is a torch.nn.Module
    and the transform is a torchvision.transforms.Compose object.
    """
    # Prepare feature extractor
    node_names: list[str] = ["global_pool"]
    feature_extractor, transform = build_feature_extractor(
        model_name="resnet18",
        node_names=node_names,
        device=torch.device("cpu"),
        weights_path=None
    )

    # Type assertions
    assert isinstance(feature_extractor, torch.nn.Module)  # FX extractor is a Module :contentReference[oaicite:2]{index=2}
    assert isinstance(transform, Compose)  # Transform pipeline is Compose 


def test_feature_extractor_outputs_expected_keys() -> None:
    """Test that the feature extractor produces a dict with sanitized node keys.

    Creates a dummy RGB image, applies the returned transform,
    passes the resulting tensor through the feature extractor,
    and checks for the presence of the expected sanitized key.
    """
    # Prepare feature extractor
    node_names: list[str] = ["layer4.0.act1"]
    feature_extractor, transform = build_feature_extractor(
        model_name="resnet18",
        node_names=node_names,
        device=torch.device("cpu"),
        weights_path=None
    )

    # Create and preprocess a dummy image
    dummy_img: Image.Image = Image.new("RGB", (224, 224))  # PIL Image 
    inp_tensor: torch.Tensor = transform(dummy_img).unsqueeze(0)  # add batch dimension 

    # Extract features
    output: dict[str, torch.Tensor] = feature_extractor(inp_tensor)

    # Functional assertions
    expected_key: str = "layer4_0_act1"  # dot replaced by underscore :contentReference[oaicite:4]{index=4}
    assert isinstance(output, dict)  # Output must be a dict :contentReference[oaicite:5]{index=5}
    assert expected_key in output  # Expected key present :contentReference[oaicite:6]{index=6}
    assert isinstance(output[expected_key], torch.Tensor)  # Value is a Tensor :contentReference[oaicite:7]{index=7}


def test_invalid_weights_path_raises_error() -> None:
    """Test that passing a non-existent weights_path raises a FileNotFoundError.

    Ensures that build_feature_extractor attempts to load the checkpoint
    and raises an error when the file does not exist.
    """
    node_names: list[str] = ["global_pool"]
    bad_path: pathlib.Path = pathlib.Path("/nonexistent/checkpoint.pth")

    with pytest.raises(FileNotFoundError):
        build_feature_extractor(
            model_name="resnet18",
            node_names=node_names,
            device=torch.device("cpu"),
            weights_path=bad_path
        )  # Loading invalid checkpoint path should error :contentReference[oaicite:8]{index=8}