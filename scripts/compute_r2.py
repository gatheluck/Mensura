import pathlib

import numpy as np
import torch
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm.contrib import tenumerate

from mensura.v_info.feature import build_feature_extractor
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
    feature_extractor, transform = build_feature_extractor(
        model_name=args.model_name,
        node_names=args.nodes.split(","),
        device=device,
    )
    # backbone = timm.create_model(args.model_name, pretrained=True).eval().to(device)
    # return_nodes = {node_name: node_name.replace(".", "_") for node_name in args.nodes.split(",")}
    # feature_extractor = create_feature_extractor(backbone, return_nodes=return_nodes).to(device).eval()

    # prepare datasets
    # print("Preparing datasets...")
    # config = resolve_data_config({}, model=backbone)
    # transform = create_transform(**config)
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
        stage0_sample = out0["stages_0"].mean((-1, -2)).cpu()
        stage1_sample = out0["stages_1"].mean((-1, -2)).cpu()
        stage2_sample = out0["stages_2"].mean((-1, -2)).cpu()
        stage3_sample = out0["stages_3"].mean((-1, -2)).cpu()
        gp_sample = out0["head_global_pool"].flatten(1).cpu()

    feat0 = torch.empty((args.num_samples, stage0_sample.size(1)), dtype=torch.float32)
    feat1 = torch.empty((args.num_samples, stage1_sample.size(1)), dtype=torch.float32)
    feat2 = torch.empty((args.num_samples, stage2_sample.size(1)), dtype=torch.float32)
    feat3 = torch.empty((args.num_samples, stage3_sample.size(1)), dtype=torch.float32)
    feat_gp = torch.empty((args.num_samples, gp_sample.size(1)), dtype=torch.float32)

    print("Extracting features...")
    with torch.no_grad():
        for i, (x, _) in tenumerate(loader, desc="[extracting]"):
            start = i * args.batch_size
            end = start + x.size(0)
            out = feature_extractor(x.to(device))
            feat0[start:end] = out["stages_0"].mean((-1,-2)).cpu()
            feat1[start:end] = out["stages_1"].mean((-1,-2)).cpu()
            feat2[start:end] = out["stages_2"].mean((-1,-2)).cpu()
            feat3[start:end] = out["stages_3"].mean((-1,-2)).cpu()
            feat_gp[start:end] = out["head_global_pool"].flatten(1).cpu()

    print("feat0.shape", feat0.shape)
    print("feat1.shape", feat1.shape)
    print("feat2.shape", feat2.shape)
    print("feat3.shape", feat3.shape)
    print("feat_gp.shape", feat_gp.shape)

    feat0_np = feat0.numpy()
    feat1_np = feat1.numpy()
    feat2_np = feat2.numpy()
    feat3_np = feat3.numpy()
    feat_gp_np = feat_gp.numpy()

    # z_index = 100
    # x_train, x_test, y_train, y_test = train_test_split(feat3_np, feat_gp_np[:, z_index], test_size=0.2, random_state=0)

    # print("x_train.shape", x_train.shape)
    # print("x_test.shape", x_test.shape)
    # print("y_train.shape", y_train.shape)
    # print("y_test.shape", y_test.shape)

    # print("Fitting Ridge regression...")
    # if False:
    #     # Use RidgeCV to find the best alpha
    #     alphas = np.logspace(-3, 4, 50) # 1e-3 ~ 1e4
    #     reg = make_pipeline(StandardScaler(with_mean=True), RidgeCV(alphas=alphas, cv=5, fit_intercept=True))
    #     reg.fit(x_train, y_train)
    #     print("best alpha =", reg[-1].alpha_)
    # else:
    #     # Use a fixed alpha
    #     reg = make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=args.ridge_alpha, fit_intercept=True))
    #     reg.fit(x_train, y_train)

    # r2 = r2_score(y_test, reg.predict(x_test), multioutput="variance_weighted")

    print("Fitting Ridge regression...")
    z_index = 100
    r2 = ridge_regression(feat3_np, feat_gp_np[:, z_index])
    print(f"R^2 = {r2:.4f}")
