import json
import pathlib
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from mensura.v_info.complexity_measure import ComplexityMeasureK, VInformation
from mensura.v_info.feature import build_feature_extractor, extract_features
from mensura.v_info.regression import ridge_regression


def path_converter(obj: object) -> str:
    """JSON serializer for `pathlib.Path` objects.

    Converts a `Path` object to its string representation so that it
    can be serialized by the standard `json` module.

    Args:
        obj (object): The object to serialize. Expected to be a `Path`.

    Returns:
        str: The string representation of the `Path`.

    Raises:
        TypeError: If `obj` is not an instance of `pathlib.Path`.
    """
    if isinstance(obj, pathlib.Path):
        return str(obj)
    raise TypeError(f"{obj!r} is not JSON serializable")


if __name__ == "__main__":
    import argparse

    torch.manual_seed(0)
    np.random.seed(0)

    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="resnet50")
    p.add_argument(
        "--nodes",
        # default="stages.0,stages.1,stages.2,stages.3,head.global_pool",
        default="layer1,layer2,layer3,layer4,global_pool",
        help="Comma separated list of FX graph node names.",
    )
    p.add_argument("--weights-path", type=pathlib.Path, default=None)
    p.add_argument("--dataset-dir-path", type=pathlib.Path, default="FOOD101")
    p.add_argument(
        "--output-dir-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/compute_complexity_measure"),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-samples", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ridge-alpha", type=float, default=100)
    args = p.parse_args()

    device = torch.device(args.device)

    # create output directory
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    dir_name = (
        f"{ts}_model={args.model_name}_weights={str(args.weights_path)}"
        if args.weights_path
        else f"{ts}_model={args.model_name}"
    )
    output_dir_path = args.output_dir_path / dir_name
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # dump arguments
    arguments_output_path = output_dir_path / "args.json"
    with arguments_output_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, default=path_converter)

    # prepare backbone
    print(f"Loading model {args.model_name}...")
    node_keys = args.nodes.split(",")
    feature_extractor, transform = build_feature_extractor(
        model_name=args.model_name,
        node_keys=node_keys,
        device=device,
        weights_path=args.weights_path,
    )

    # prepare datasets
    import torchvision
    # if args.dataset_dir_path == pathlib.Path("CIFAR10"):
    # print("Loading CIFAR10 dataset...")
    # dataset = torchvision.datasets.CIFAR10(
    #     root="./data",
    #     train=False,
    #     download=True,
    #     transform=transform
    # )

    print("Loading FOOD101 dataset...")
    dataset = torchvision.datasets.Food101(
        root="./data",
        split="test",
        download=True,
        transform=transform
    )

    # else:
    # dataset = ImageFolder(root=args.dataset_dir_path, transform=transform)
    print("dataset:", dataset)
    subset = Subset(dataset, torch.randperm(len(dataset))[: args.num_samples])
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    print("transform:", transform)
    print("len(subset):", len(subset))

    print("Extracting features...")
    features = extract_features(feature_extractor, loader, args.num_samples, device)
    features_np = {k: v.numpy() for k, v in features.items() if k not in ("global_pool", "layer4_raw")}
    z_all = features["global_pool"].numpy() # shape: (num_samples, 2048)
    penultimate = features["layer4_raw"].numpy()
    print(f"penultimate.shape: {penultimate.shape}") # shape: (num_samples, 2048, 7, 7)

    # Reshape penultimate features for processing
    # Original shape: (num_samples, 2048, 7, 7)
    # Reshape to: (num_samples, 7*7*2048) to iterate over all spatial positions and channels
    num_samples, channels, height, width = penultimate.shape
    penultimate_reshaped = penultimate.transpose(0, 2, 3, 1).reshape(num_samples, -1)
    # penultimate_reshaped shape: (num_samples, 7*7*2048)
    
    print(f"penultimate_reshaped.shape: {penultimate_reshaped.shape}")
    
    # compute complexity measure K for each feature in penultimate layer
    complexity_measure_output_path = output_dir_path / "complexity_measure.jsonl"
    with tqdm(enumerate(penultimate_reshaped.T), total=penultimate_reshaped.shape[1]) as pbar:
        for i, z in pbar:
            # Calculate channel, height, and width indices
            # i = c*h*w + h*w + w
            pos_idx = i
            w_idx = pos_idx % width
            pos_idx = pos_idx // width
            h_idx = pos_idx % height
            c_idx = pos_idx // height
            
            v_infos = OrderedDict()
            for k, feat_np in features_np.items():
                var_z = float(np.var(z, ddof=0))
                r2 = ridge_regression(feat_np, z)
                v_infos[k] = VInformation(var_z, r2)
            complexity_measure_k = ComplexityMeasureK(v_infos)
            pbar.set_postfix(complexity_measure_k=f"{complexity_measure_k.value:.4f}")
            print(f"Complexity measure K of z_({c_idx},{h_idx},{w_idx}) = {complexity_measure_k.value:.4f}")

            with complexity_measure_output_path.open("a", encoding="utf-8") as f:
                out_dict = {
                    "z_index": i,
                    "channel": int(c_idx),
                    "height": int(h_idx),
                    "width": int(w_idx),
                    "complexity_measure": asdict(complexity_measure_k),
                }
                json.dump(out_dict, f, indent=4)
                f.write("\n")
