import pathlib

import numpy as np
import timm
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm.contrib import tenumerate

if __name__ == "__main__":
    import argparse

    torch.manual_seed(0)
    np.random.seed(0)

    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="resnetv2_50x1_bit.goog_distilled_in1k")
    p.add_argument("--nodes",
                   default="stages.0,stages.2",
                   help="Comma separated list of FX graph node names.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-samples", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ridge-alpha", type=float, default=1e-3)
    args = p.parse_args()

    device = torch.device(args.device)

   # prepare backbone
    print(f"Loading model {args.model_name}...")
    backbone = timm.create_model(args.model_name, pretrained=True).eval().to(device)
    return_nodes = {node_name: node_name.replace(".", "_") for node_name in args.nodes.split(",")}
    feature_extractor = create_feature_extractor(backbone, return_nodes=return_nodes).to(device).eval()

    # prepare datasets
    print("Preparing datasets...")
    config = resolve_data_config({}, model=backbone)
    transform = create_transform(**config)
    dataset_dir_path = pathlib.Path("data/imagenet")
    dataset = ImageFolder(root=dataset_dir_path / "val", transform=transform)
    subset   = Subset(dataset, torch.randperm(len(dataset))[:args.num_samples])
    loader   = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print("transform:", transform)
    print("len(subset):", len(subset))

    # NOTE: don't use torch.cat() to avoid memory spiking
    with torch.no_grad():
        x0, _ = next(iter(loader))
        out0  = feature_extractor(x0.to(device))
        stage0_sample = out0["stages_0"].mean((-1, -2)).cpu()   # (bsz, C0)
        stage2_sample = out0["stages_2"].mean((-1, -2)).cpu()   # (bsz, C2)

    feat0 = torch.empty((args.num_samples, stage0_sample.size(1)), dtype=torch.float32)
    feat2 = torch.empty((args.num_samples, stage2_sample.size(1)), dtype=torch.float32)

    print("Extracting features...")
    with torch.no_grad():
        for i, (x, _) in tenumerate(loader, desc="[extracting]"):
            start = i * args.batch_size
            end = start + x.size(0)
            out = feature_extractor(x.to(device))
            feat0[start:end] = out["stages_0"].mean((-1,-2)).cpu()
            feat2[start:end] = out["stages_2"].mean((-1,-2)).cpu()

    print("feat0.shape", feat0.shape)
    print("feat2.shape", feat2.shape)

    feat0_np = feat0.numpy()
    feat2_np = feat2.numpy()
    x_train, x_test, y_train, y_test = train_test_split(feat0_np, feat2_np, test_size=0.2, random_state=0)

    print("x_train.shape", x_train.shape)
    print("x_test.shape", x_test.shape)
    print("y_train.shape", y_train.shape)
    print("y_test.shape", y_test.shape)

    print("Fitting Ridge regression...")
    reg = Ridge(alpha=args.ridge_alpha, fit_intercept=True)
    reg.fit(x_train, y_train)

    r2 = r2_score(y_test, reg.predict(x_test), multioutput="variance_weighted")
    print(f"R^2 = {r2:.4f}")
