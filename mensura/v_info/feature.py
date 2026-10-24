import pathlib

import timm
import torch
import torchvision
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm.contrib import tenumerate


def build_feature_extractor(
    model_name: str,
    node_keys: list[str],
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
        node_keys (list[str]): FX graph node keys to extract features from,
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
        print(f"Loading custom weights from {weights_path}...")
        backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=1000,
            checkpoint_path=str(weights_path),
        )
    else:
        print(f"Loading pretrained weights for {model_name}...")
        backbone = timm.create_model(model_name, pretrained=True)

    backbone = backbone.eval().to(device)

    # Prepare FX return nodes mapping
    return_nodes = {name: name.replace(".", "_") for name in node_keys}

    # Build the feature extractor
    feature_extractor = (
        create_feature_extractor(backbone, return_nodes=return_nodes).eval().to(device)
    )

    # Create evaluation transform
    # config = resolve_data_config({}, model=backbone)  # type: ignore[no-untyped-call]
    # transform = create_transform(**config)
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=2),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    return feature_extractor, transform


def extract_features(
    feature_extractor: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    sample_limit: int,
    device: torch.device,
    node_keys: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Extract features from specified nodes into NumPy arrays.

    Performs a warm-up pass to infer output dimensions, allocates
    fixed-size buffers, then iterates over the DataLoader in no-grad
    mode to fill those buffers with flattened or pooled features.

    Args:
        feature_extractor (torch.nn.Module):
            An FX GraphModule returning a dict mapping each node key
            to its activation Tensor when called.
        loader (DataLoader):
            Yields (input, label) pairs. Its batch_size determines slice offsets.
        sample_limit (int):
            Total number of samples to extract; must be ≥ total samples in `loader`.
        device (torch.device):
            Device on which to perform inference (e.g., `torch.device('cuda')`).
        node_keys (list[str] | None):
            List of feature map keys to extract. If `None`, all keys from the first
            output of `feature_extractor` are used.

    Returns:
        dict[str, torch.Tensor]:
            Mapping from each node key to a tensor of shape
            `(sample_limit, feature_dim)`, where `feature_dim` is either
            the channel count (for 2D maps) or the flattened size.

    Raises:
        ValueError: If `loader` is empty, if `sample_limit` is too small,
            or if any tensor has unsupported dimensions.
    """

    def _flatten_feature(tensor: torch.Tensor) -> torch.Tensor:
        """Pool 2D features or flatten 2D tensors; error on other dims.

        Args:
            tensor (Tensor): Activation tensor of shape (B, C, H, W) or (B, D).

        Returns:
            Tensor: Pooled or flattened features of shape (B, C) or (B, D).

        Raises:
            ValueError: If `tensor.dim()` is not 2 or 4.
        """
        dim = tensor.dim()  # number of dimensions
        if dim == 4:
            return tensor.mean(dim=(-1, -2))
        if dim == 2:
            return tensor
        raise ValueError(f"Unsupported tensor dimension: {dim}. Expected 2 or 4.")

    # Warm-up to infer feature dimensions
    try:
        first_batch, _ = next(iter(loader))
    except StopIteration:
        raise ValueError("DataLoader is empty; cannot extract features.") from None

    with torch.no_grad():
        sample_out = feature_extractor(first_batch.to(device))

    # Determine node keys if not provided
    if node_keys is None:
        node_keys = list(sample_out.keys())

    # Allocate buffers
    # NOTE: don't use torch.cat() to avoid memory spiking
    buffers: dict[str, torch.Tensor] = {}
    for key in node_keys:
        feat0 = _flatten_feature(sample_out[key])
        buffers[key] = torch.empty((sample_limit, feat0.size(1)), dtype=feat0.dtype)
        if key == "layer4":
            # 生テンソルのshapeでバッファ作るぞ！
            raw_layer4_buffer = torch.empty(
                (sample_limit, *sample_out[key].shape[1:]), dtype=sample_out[key].dtype
            )

    # Fill buffers
    # NOTE: total_samples is a needed for the case drop_last=False
    total_samples = 0  # running count of samples processed
    with torch.no_grad():
        for _, (x_batch, _) in tenumerate(loader, desc="Extracting"):
            batch_size = x_batch.size(0)
            # Detect overflow by checking cumulative total
            if total_samples + batch_size > sample_limit:
                raise ValueError(
                    f"sample_limit={sample_limit} too small for dataset "
                    f"({total_samples + batch_size} samples required)."
                )
            start = total_samples
            end = start + batch_size
            out = feature_extractor(x_batch.to(device))
            for key in node_keys:
                flattened = _flatten_feature(out[key]).cpu()
                buffers[key][start:end] = flattened
                if key == "layer4" and raw_layer4_buffer is not None:
                    # flattenせずに生テンソルも保存！
                    raw_layer4_buffer[start:end] = out[key].cpu()
            total_samples = end  # update running total
    # layer4の生テンソルも返す
    if raw_layer4_buffer is not None:
        buffers["layer4_raw"] = raw_layer4_buffer

    return buffers
