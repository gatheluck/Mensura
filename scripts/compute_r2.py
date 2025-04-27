import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from mensura.v_info.feature import build_feature_extractor, extract_features
from mensura.v_info.regression import ridge_regression

if __name__ == "__main__":
    import argparse

    torch.manual_seed(0)
    np.random.seed(0)

    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="resnetv2_50x1_bit.goog_distilled_in1k")
    p.add_argument("--nodes",
                   default="stages.0,stages.1,stages.2,stages.3,head.global_pool",
                   help="Comma separated list of FX graph node names.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ridge-alpha", type=float, default=100)
    args = p.parse_args()

    device = torch.device(args.device)

   # prepare backbone
    print(f"Loading model {args.model_name}...")
    node_keys = args.nodes.split(",")
    feature_extractor, transform = build_feature_extractor(
        model_name=args.model_name,
        node_keys=node_keys,
        device=device,
    )

    # prepare datasets
    dataset_dir_path = pathlib.Path("data/imagenet")
    dataset = ImageFolder(root=dataset_dir_path / "val", transform=transform)
    subset = Subset(dataset, torch.randperm(len(dataset))[:args.num_samples])
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print("transform:", transform)
    print("len(subset):", len(subset))

    print("Extracting features...")
    features = extract_features(feature_extractor, loader, args.num_samples, device)
    feat0_np = features["stages_0"].numpy()
    feat1_np = features["stages_1"].numpy()
    feat2_np = features["stages_2"].numpy()
    feat3_np = features["stages_3"].numpy()
    feat_gp_np = features["head_global_pool"].numpy()

    print("Fitting Ridge regression...")
    z_index = 100
    r2 = ridge_regression(feat3_np, feat_gp_np[:, z_index])
    print(f"R^2 = {r2:.4f}")
