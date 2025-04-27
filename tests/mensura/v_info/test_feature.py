import pathlib

import pytest
import torch
from PIL import Image
from torchvision.transforms import Compose

from mensura.v_info.feature import build_feature_extractor, extract_features


def test_returns_correct_types() -> None:
    """Test that build_feature_extractor returns a Module and a Compose transform.

    Uses a ResNet18 backbone (pretrained=False) and a single node name
    to verify that the returned feature extractor is a torch.nn.Module
    and the transform is a torchvision.transforms.Compose object.
    """
    # Prepare feature extractor
    node_keys: list[str] = ["global_pool"]
    feature_extractor, transform = build_feature_extractor(
        model_name="resnet18",
        node_keys=node_keys,
        device=torch.device("cpu"),
        weights_path=None,
    )

    # Type assertions
    assert isinstance(
        feature_extractor, torch.nn.Module
    )  # FX extractor is a Module :contentReference[oaicite:2]{index=2}
    assert isinstance(transform, Compose)  # Transform pipeline is Compose


def test_feature_extractor_outputs_expected_keys() -> None:
    """Test that the feature extractor produces a dict with sanitized node keys.

    Creates a dummy RGB image, applies the returned transform,
    passes the resulting tensor through the feature extractor,
    and checks for the presence of the expected sanitized key.
    """
    # Prepare feature extractor
    node_keys: list[str] = ["layer4.0.act1"]
    feature_extractor, transform = build_feature_extractor(
        model_name="resnet18",
        node_keys=node_keys,
        device=torch.device("cpu"),
        weights_path=None,
    )

    # Create and preprocess a dummy image
    dummy_img: Image.Image = Image.new("RGB", (224, 224))  # PIL Image
    inp_tensor: torch.Tensor = transform(dummy_img).unsqueeze(0)  # add batch dimension

    # Extract features
    output: dict[str, torch.Tensor] = feature_extractor(inp_tensor)

    # Functional assertions
    expected_key: str = "layer4_0_act1"  # dot replaced by underscore :contentReference[oaicite:4]{index=4}
    assert isinstance(
        output, dict
    )  # Output must be a dict :contentReference[oaicite:5]{index=5}
    assert (
        expected_key in output
    )  # Expected key present :contentReference[oaicite:6]{index=6}
    assert isinstance(
        output[expected_key], torch.Tensor
    )  # Value is a Tensor :contentReference[oaicite:7]{index=7}


def test_invalid_weights_path_raises_error() -> None:
    """Test that passing a non-existent weights_path raises a FileNotFoundError.

    Ensures that build_feature_extractor attempts to load the checkpoint
    and raises an error when the file does not exist.
    """
    node_keys: list[str] = ["global_pool"]
    bad_path: pathlib.Path = pathlib.Path("/nonexistent/checkpoint.pth")

    with pytest.raises(FileNotFoundError):
        build_feature_extractor(
            model_name="resnet18",
            node_keys=node_keys,
            device=torch.device("cpu"),
            weights_path=bad_path,
        )  # Loading invalid checkpoint path should error :contentReference[oaicite:8]{index=8}


class DummyDataset(torch.utils.data.Dataset[tuple[torch.Tensor, int]]):
    """A dataset yielding random tensors of a given shape for testing."""

    def __init__(self, num_samples: int, tensor_shape: tuple[int, ...]) -> None:
        """Initialize the dataset with a given number of samples and tensor shape."""
        self.num_samples = num_samples
        self.tensor_shape = tensor_shape

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return a random tensor and dummy label."""
        return torch.randn(self.tensor_shape), 0


class StubExtractor(torch.nn.Module):
    """Feature extractor stub returning dicts with 4D and 2D tensors."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Simulate a spatial map and a flat feature."""
        # Simulate a spatial map and a flat feature
        spatial = x  # e.g. (B, C, H, W)
        flat = x.mean(dim=(-1, -2)) if x.dim() == 4 else x  # e.g. (B, C)
        return {"feat4d": spatial, "feat2d": flat}


def test_valid_feature_extraction_with_keys() -> None:
    """Test extraction when explicit node_keys list is provided."""
    ds = DummyDataset(4, (3, 4, 4))
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    extractor = StubExtractor().eval()
    buffers = extract_features(
        feature_extractor=extractor,
        loader=loader,
        sample_limit=4,
        device=torch.device("cpu"),
        node_keys=["feat4d"],
    )
    # Both buffers must exist and have shape (4, 3)
    assert set(buffers.keys()) == {"feat4d"}
    for buf in buffers.values():
        assert isinstance(buf, torch.Tensor)
        assert buf.shape == (4, 3)


def test_valid_feature_extraction_default_keys() -> None:
    """Test that omitting node_keys extracts all available features."""
    ds = DummyDataset(4, (3, 4, 4))
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    extractor = StubExtractor().eval()
    # No node_keys passed → should pick up both "feat4d" and "feat2d"
    buffers = extract_features(
        feature_extractor=extractor,
        loader=loader,
        sample_limit=4,
        device=torch.device("cpu"),
    )
    assert set(buffers.keys()) == {"feat4d", "feat2d"}
    for buf in buffers.values():
        assert isinstance(buf, torch.Tensor)
        assert buf.shape == (4, 3)


def test_empty_loader_raises() -> None:
    """Test that an empty DataLoader causes a ValueError."""
    empty_ds = DummyDataset(num_samples=0, tensor_shape=(3, 4, 4))
    empty_loader = torch.utils.data.DataLoader(empty_ds, batch_size=2)
    with pytest.raises(
        ValueError, match="DataLoader is empty; cannot extract features."
    ):
        extract_features(
            feature_extractor=StubExtractor(),
            loader=empty_loader,
            sample_limit=1,
            device=torch.device("cpu"),
        )


def test_sample_limit_overflow_raises() -> None:
    """Test that a too-small sample_limit raises ValueError on overflow."""
    ds = DummyDataset(num_samples=5, tensor_shape=(3, 4, 4))
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    with pytest.raises(ValueError, match=r"sample_limit=4 too small"):
        extract_features(
            feature_extractor=StubExtractor(),
            loader=loader,
            sample_limit=4,
            device=torch.device("cpu"),
        )


def test_unsupported_dim_raises() -> None:
    """Test that tensors with unsupported dims raise a ValueError."""

    class BadExtractor(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            # Return a 3D tensor to trigger the error
            return {"bad": torch.randn(x.size(0), 2, 2)}

    ds = DummyDataset(num_samples=1, tensor_shape=(2, 2))
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    bad_extractor = BadExtractor().eval()
    with pytest.raises(ValueError, match="Unsupported tensor dimension: 3"):
        extract_features(
            feature_extractor=bad_extractor,
            loader=loader,
            sample_limit=1,
            device=torch.device("cpu"),
        )
