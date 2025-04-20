import pytest
import torch

from mensura.v_info.linear_probes import (
    LinearProbes,  # adjust import path to your project
)


@pytest.fixture(scope="module")
def dummy_feats() -> dict[str, torch.Tensor]:
    """Return fake feature maps for three layers."""
    B = 8  # noqa: N806
    return {
        "l_small": torch.randn(B, 64, 4, 4),  # flattened d = 1024  (≤ proj_dim)
        "l_proj": torch.randn(B, 256, 7, 7),  # flattened d = 12544 (> proj_dim)
        "l_large": torch.randn(B, 512, 7, 7),  # flattened d = 25088 (> proj_dim)
    }


def test_init_creates_probes(dummy_feats):
    """Constructor creates linear heads and optional projection buffers."""
    in_dims = {k: v.flatten(1).shape[1] for k, v in dummy_feats.items()}
    probes = LinearProbes(in_dims, proj_dim=4096)

    # correct keys exist
    assert set(probes.probes.keys()) == set(in_dims)

    # check proj buffer present only for high-d layers
    assert not hasattr(probes, "proj_l_small")
    assert hasattr(probes, "proj_l_proj")
    assert hasattr(probes, "proj_l_large")

    # projection shape is (d, proj_dim)
    assert probes.proj_l_proj.shape == (in_dims["l_proj"], 4096)
    assert probes.proj_l_large.shape == (in_dims["l_large"], 4096)


def test_forward_shapes(dummy_feats):
    """Forward returns a scalar prediction (B,) for each layer."""
    in_dims = {k: v.flatten(1).shape[1] for k, v in dummy_feats.items()}
    probes = LinearProbes(in_dims, proj_dim=4096)

    outputs = probes(dummy_feats)

    assert set(outputs.keys()) == set(in_dims)
    for out in outputs.values():
        assert out.shape == (dummy_feats["l_small"].size(0),)  # (B,)


def test_state_dict_roundtrip(tmp_path, dummy_feats):
    """state_dict() should save & reload weights + projection matrices."""
    in_dims = {k: v.flatten(1).shape[1] for k, v in dummy_feats.items()}
    probes = LinearProbes(in_dims, proj_dim=4096)

    # run one forward pass to create proj buffers
    _ = probes(dummy_feats)

    file_path = tmp_path / "probes.pt"
    torch.save(probes.state_dict(), file_path)

    # load into a fresh instance
    probes2 = LinearProbes(in_dims, proj_dim=4096)
    probes2.load_state_dict(torch.load(file_path))

    # predictions must match after reload
    out1 = probes(dummy_feats)
    out2 = probes2(dummy_feats)
    for k in out1:
        torch.testing.assert_close(out1[k], out2[k], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("proj_dim", [4096, 2048])
def test_projection_effect(dummy_feats, proj_dim):
    """Check that high-dim layers are projected to at most proj_dim."""
    in_dims = {k: v.flatten(1).shape[1] for k, v in dummy_feats.items()}
    probes = LinearProbes(in_dims, proj_dim=proj_dim)
    # forward once to ensure buffers built
    probes(dummy_feats)

    for name, d in in_dims.items():
        if d > proj_dim:
            buf = getattr(probes, f"proj_{name}")
            assert buf.shape[1] == proj_dim
        else:
            assert not hasattr(probes, f"proj_{name}")
