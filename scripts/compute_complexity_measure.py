import pathlib
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from mensura.v_info.complexity_measure import ComplexityMeasureK, VInformation
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
    features_np = {k: v.numpy() for k, v in features.items() if k != "head_global_pool"}
    z_all = features["head_global_pool"].numpy()

    # compute complexity measure K
    with tqdm(enumerate(z_all.T), total=z_all.shape[1]) as pbar:
        for i, z in pbar:
            v_infos = OrderedDict()
            for k, feat_np in features_np.items():
                print(f"{k}: {feat_np.shape}")
                var_z = float(np.var(z, ddof=0))
                r2 = ridge_regression(feat_np, z)
                v_infos[k] = VInformation(var_z, r2)
            complexity_measure_k = ComplexityMeasureK(v_infos)
            pbar.set_postfix(complexity_measure_k=f"{complexity_measure_k.value:.4f}")
            print(f"Complexity measure K of z_{i} = {complexity_measure_k.value:.4f}")
            print(complexity_measure_k)
