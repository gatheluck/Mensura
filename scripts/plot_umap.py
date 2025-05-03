import json
import pathlib
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from mensura.v_info.feature import build_feature_extractor, extract_features


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
        # default="layer1,layer2,layer3,layer4,global_pool", # --model-name=resnet50
        help="Comma separated list of FX graph node names.",
    )
    p.add_argument("--weights-path", type=pathlib.Path, default=None)
    p.add_argument("--dataset-dir-path", type=pathlib.Path, default="data/imagenet/val")
    p.add_argument(
        "--output-dir-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/umap"),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-samples", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ridge-alpha", type=float, default=100)
    p.add_argument("--jsonl_path", type=str)
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
    features_np = {k: v.numpy() for k, v in features.items() if k != "global_pool"}
    z_all = features["global_pool"].numpy() # z_all.shape: (20000, 2048)

    jsonl_path = Path(args.jsonl_path)

    # Storage dict
    last_values = {}

    buffer = ''
    brace_level = 0

    with open(jsonl_path, 'r') as f:
        while True:
            c = f.read(1)
            if not c:
                break
            if c == '{':
                if brace_level == 0:
                    buffer = ''  # start new object
                brace_level += 1
            if brace_level > 0:
                buffer += c
            if c == '}':
                brace_level -= 1
                if brace_level == 0:
                    # Completed a JSON object in buffer
                    # Extract z_index
                    idx_match = re.search(r'"z_index"\s*:\s*(\d+)', buffer)
                    if idx_match:
                        idx = int(idx_match.group(1))
                        # find all value occurrences
                        val_matches = list(re.finditer(r'"value"\s*:\s*([-+]?\d*\.\d+|\d+)', buffer))
                        if val_matches:
                            last_val_str = val_matches[-1].group(1)
                            last_val = float(last_val_str)
                            last_values[idx] = last_val
                    buffer = ''  # reset

    # Build list for indices 0..2047
    values_list = [last_values.get(i, float('nan')) for i in range(2048)]

    # UMAPによる次元削減
    reducer = umap.UMAP(n_components=2, n_neighbors=100, min_dist=0.5, random_state=42)
    embedding = reducer.fit_transform(z_all.T)
    # z_all.T: 各z_index（特徴次元）」ごとに、20000サンプル分の値が並んでる状態
    # ここでUMAPに渡してるのは「2048個のベクトル（各z_indexごと）」

    # UMAPで「2048次元→2次元」に圧縮！
    # 2048個の点をプロットしてる！
    # 各点は「z_indexごとの特徴ベクトル」を表してる！

    # UMAP投影結果の可視化（color に values_list を指定）
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=values_list,
        cmap='viridis',
        alpha=0.8,
        vmin=0.0,
        vmax=1.0,
    )
    plt.title("visualatom only pretrain UMAP Projection of Features (colored by values_list)")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.colorbar(scatter, label="values_list")

    # 画像の保存＆表示
    plt.savefig(output_dir_path / "umap_by_values_list.png", dpi=300, bbox_inches='tight')
    plt.show()
