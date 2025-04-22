import pathlib

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
from tqdm import tqdm

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="resnetv2_50x1_bit.goog_distilled_in1k")
    p.add_argument("--nodes",
                   default="stages.0,stages.2",
                   help="Comma separated list of FX graph node names.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-samples", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ridge-alpha", type=float, default=1e-3)
    args = p.parse_args()

    device = torch.device(args.device)

   # prepare backbone
    print(f"Loading model {args.model_name}...")
    backbone = timm.create_model(args.model_name, pretrained=True).eval().to(device)
    return_nodes = {node_name: node_name.replace(".", "_") for node_name in args.nodes.split(",")}
    feature_extractor = create_feature_extractor(backbone, return_nodes=return_nodes)

    # prepare datasets
    print("Preparing datasets...")
    config = resolve_data_config({}, model=backbone)
    transform = create_transform(**config)
    dataset_dir_path = pathlib.Path("data/imagenet")
    dataset = ImageFolder(root=dataset_dir_path / "val", transform=transform)
    subset   = Subset(dataset, torch.randperm(len(dataset))[:args.num_samples])
    loader   = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print("transform", transform)
    print("len(subset)", len(subset))

    out_stage_0 = []
    out_stage_2 = []

    print("Extracting features...")
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="[extracting]"):
            out = feature_extractor(x.to(device))
            out_stage_0.append(out["stages_0"].mean((-1, -2)).cpu())
            out_stage_2.append(out["stages_2"].flatten(1).cpu())

    out_stage_0 = torch.cat(out_stage_0, 0).numpy()
    out_stage_2 = torch.cat(out_stage_2, 0).numpy()

    print("out_stage_0.shape", out_stage_0.shape)
    print("out_stage_2.shape", out_stage_2.shape)

    x_train, x_test, y_train, y_test = train_test_split(out_stage_0, out_stage_2, test_size=0.2, random_state=0)

    print("x_train.shape", x_train.shape)
    print("x_test.shape", x_test.shape)
    print("y_train.shape", y_train.shape)
    print("y_test.shape", y_test.shape)

    reg = Ridge(alpha=args.ridge_alpha, fit_intercept=True)
    reg.fit(x_train, y_train)

    r2 = r2_score(y_test, reg.predict(x_test), multioutput="variance_weighted")
    print(f"R^2 = {r2:.4f}")
