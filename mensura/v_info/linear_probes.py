import numpy as np
import torch
import torch.nn as nn


class LinearProbes(nn.Module):
    """Linear probes for multiple intermediate layers.

    Each probe is a single-output fully connected layer that optionally
    receives a fixed random projection of the original activation tensor.
    All probes and projection matrices are handled as part of one
    :class:`torch.nn.Module`, so they can be saved and restored with the
    regular ``state_dict`` workflow.

    Attributes:
        proj_dim (int): Target dimensionality used for the random projection.
        probes (nn.ModuleDict): Mapping from layer names to the corresponding
            :class:`torch.nn.Linear` heads.
        proj_<layer> (Tensor): Fixed ``(d, proj_dim)`` Gaussian projection
            matrix registered as a buffer for every layer whose flattened size
            is larger than ``proj_dim``.  These buffers are not trainable, but
            they are included in the module's state di

    Example:
        >>> probes = LinearProbes({'layer1': 100352, 'layer4': 50176}, proj_dim=4096)
        >>> feats = {
        ...     'layer1': torch.randn(32, 256, 7, 7),
        ...     'layer4': torch.randn(32, 512, 7, 7),
        ... }
        >>> preds = probes(feats)  # {'layer1': (32,), 'layer4': (32,)}
    """

    def __init__(self, in_dims: dict[str, int], proj_dim: int = 4096) -> None:
        """Initializes a linear probe for each specified layer.

        Args:
            in_dims: A mapping from layer names to their flattened input
                dimensionalities.
            proj_dim: The dimensionality of the random projection. If a
                flattened layer size ``d`` is less than or equal to
                ``proj_dim``, no projection is applied for that layer.

        """
        super().__init__()
        self.proj_dim = int(proj_dim)
        self.probes = nn.ModuleDict()

        for name, d in in_dims.items():
            d_in = min(d, self.proj_dim)
            self.probes[name] = nn.Linear(d_in, 1, bias=True)

            if d > self.proj_dim:
                proj = torch.randn(d, self.proj_dim, dtype=torch.float16) / np.sqrt(self.proj_dim)
                self.register_buffer(f"proj_{name}", proj, persistent=True)

    def forward(self, feat_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Runs each probe on the corresponding feature tensor.

        Args:
            feat_dict: A mapping from layer names to activation tensors.
                Tensors can have shape ``(B, C, H, W)`` or ``(B, d)``. Spatial
                dimensions are automatically flattened.

        Returns:
            dict[str, torch.Tensor]: A mapping from layer names to the probe
            predictions, each with shape ``(B,)``.

        Raises:
            KeyError: If a key in ``feat_dict`` is not present in
                :pyattr:`probes`.
        """
        outputs: dict[str, torch.Tensor] = {}
        for name, feat in feat_dict.items():
            x = feat.flatten(start_dim=1)  # (B, d)
            if x.shape[1] > self.proj_dim:
                proj = getattr(self, f"proj_{name}")
                x = x @ proj
            outputs[name] = self.probes[name](x).squeeze(1)
        return outputs
