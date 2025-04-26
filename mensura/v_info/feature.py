import pathlib

import timm
import torch
import torchvision
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor


def build_feature_extractor(
    model_name: str,
    node_names: list[str],
    device: torch.device,
    weights_path: pathlib.Path | None = None,
) -> tuple[torch.nn.Module, torchvision.transforms.Compose]:
    """Create a timm backbone (with optional custom weights) and an FX feature extractor.

    Constructs a feature extractor that hooks into the specified nodes of a timm model.
    If `weights_path` is provided, loads its checkpoint after initializing the model;
    otherwise uses the official pretrained weights. Returns both the extractor module
    and the evaluation transform pipeline.

    Args:
        model_name (str): Name accepted by `timm.create_model`.
        node_names (list[str]): FX graph node names to extract features from,
            e.g. ["stages.0", "head.global_pool"].
        device (torch.device): Device on which to place the model.
        weights_path (pathlib.Path | None): Optional path to a local checkpoint file.
            If given, loads this checkpoint via the `checkpoint_path` argument in
            `timm.create_model`; otherwise sets `pretrained=True`.

    Returns:
        tuple[torch.nn.Module, torchvision.transforms.Compose]:
            - feature_extractor: An FX GraphModule that returns a dict mapping each
              sanitized node name (dots → underscores) to its activation tensor.
            - transform: A `Compose` transform pipeline for evaluation (resize,
              center-crop, to-tensor, normalize).
    """
    # Initialize backbone (load custom checkpoint if provided)
    if weights_path is not None:
        backbone = timm.create_model(
            model_name,
            pretrained=False,
            checkpoint_path=str(weights_path),
        )
    else:
        backbone = timm.create_model(model_name, pretrained=True)

    backbone = backbone.eval().to(device)

    # Prepare FX return nodes mapping
    return_nodes = {name: name.replace(".", "_") for name in node_names}

    # Build the feature extractor
    feature_extractor = create_feature_extractor(
        backbone, return_nodes=return_nodes
    ).eval().to(device)

    # Create evaluation transform
    config = resolve_data_config({}, model=backbone)
    transform = create_transform(**config)

    return feature_extractor, transform