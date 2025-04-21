import datetime as dt
import json
import pathlib

import numpy as np
import timm
import torch
import torchvision
from torchmetrics.functional import r2_score
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm, trange

from mensura.v_info.linear_probes import LinearProbes


@torch.no_grad()
def evaluate_on_val(
    extractor: torch.nn.Module,
    probes: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Return mean R^2 over all probe layers on the full val set."""
    probes.eval()

    z_all: list[torch.Tensor] = []
    preds_accum: dict[str, list[torch.Tensor]] = {k: [] for k in probes.probes}

    for img, lbl in tqdm(val_loader, desc="val", unit="batch", leave=False):
        img, lbl = img.to(device), lbl.to(device)
        feats = extractor(img)
        # TODO: change this to use z_strategy
        z_index = 0
        z = feats["stages_2"].flatten(1)[:, z_index]
        z_all.append(z)

        out = probes(feats)
        for k, p in out.items():
            preds_accum[k].append(p)

    z_full = torch.cat(z_all, 0)
    r2_list = []
    for k in preds_accum:
        preds = torch.cat(preds_accum[k], 0)
        r2_list.append(float(r2_score(preds, z_full)))

    probes.train()
    return float(np.mean(r2_list))

def train_pipeline(
    extractor: torch.nn.Module,
    proj_dim: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    save_dir_path: pathlib.Path,
):
    # build LinearProbes
    print("Building LinearProbes...")
    with torch.no_grad():
        sample, _ = next(iter(train_loader))
        dims = {k: v.flatten(1).shape[1]
                for k, v in extractor(sample.to(device)).items()}
    probes = LinearProbes(dims, proj_dim=proj_dim).to(device)

    print("Initializing optimizer...")
    optim = torch.optim.SGD(probes.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    print("Training...")    
    for _ in trange(epochs, desc="epoch"):
        running = 0.0
        for img, lbl in tqdm(train_loader, desc="train", unit="batch", leave=False):
            img, lbl = img.to(device), lbl.to(device)
            with torch.no_grad():
                feats = extractor(img)
                
                # TODO: change this to use z_strategy
                z_index = 0
                z = feats["stages_2"].flatten(1)[:, z_index]
            optim.zero_grad()
            loss = sum(loss_fn(p, z) for p in probes(feats).values())
            loss.backward()
            optim.step()
            running += loss.item()
        val_r2 = evaluate_on_val(backbone, extractor, probes, val_loader, device)
        tqdm.write(f"val mean R^2 = {val_r2:.4f}")

    save_dir_path.mkdir(exist_ok=True, parents=True)
    torch.save(probes.state_dict(), save_dir_path / "linear_probes.pth")
    tqdm.write(f"linear probes checkpoint saved to `{str(save_dir_path)}`")

if __name__ == "__main__":
    import argparse

    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="resnetv2_50x1_bit.goog_distilled_in1k",
                   help="Any timm model name.")
    p.add_argument("--nodes",
                   default="stages.0,stages.1,stages.2",
                   help="Comma separated list of FX graph node names.")
    p.add_argument("--save-dir-path", type=pathlib.Path, default="outputs/")
    p.add_argument("--device", default="cuda")
    p.add_argument("--proj-dim", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-2)
    args = p.parse_args()

    # save config
    args.save_dir_path.mkdir(exist_ok=True, parents=True)
    cfg: dict = vars(args).copy()
    cfg["saved_at"] = dt.datetime.now().isoformat()
    with (args.save_dir_path / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(cfg, fp, indent=2, default=str) 

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
    train_dataset = torchvision.datasets.ImageFolder(root=dataset_dir_path / "train", transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root=dataset_dir_path / "val", transform=transform)

    # prepare dataloaders
    print("Preparing dataloaders...")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train_pipeline(feature_extractor, args.proj_dim, train_loader, val_loader, args.epochs, args.lr, device, args.save_dir_path)