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
    p.add_argument("--model-name", default="resnetv2_50x1_bit.goog_distilled_in1k")
    p.add_argument(
        "--nodes",
        default="stages.0,stages.1,stages.2,stages.3,head.global_pool",
        help="Comma separated list of FX graph node names.",
    )
    p.add_argument("--weights-path", type=pathlib.Path, default=None)
    p.add_argument("--dataset-dir-path", type=pathlib.Path, default="data/imagenet/val")
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
    dataset = ImageFolder(root=args.dataset_dir_path, transform=transform)
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
    features_np = {k: v.numpy() for k, v in features.items() if k != "head_global_pool"}
    z_all = features["head_global_pool"].numpy()

    # compute complexity measure K
    complexity_measure_output_path = output_dir_path / "complexity_measure.jsonl"
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

            with complexity_measure_output_path.open("a", encoding="utf-8") as f:
                out_dict = {
                    "z_index": i,
                    "complexity_measure": asdict(complexity_measure_k),
                }
                json.dump(out_dict, f, indent=4)
                f.write("\n")
