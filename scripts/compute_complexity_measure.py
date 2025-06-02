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
        # default="stages.0,stages.1,stages.2,stages.3,head.global_pool",
        default="layer1,layer2,layer3,layer4,global_pool",
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
    # import torchvision
    # if args.dataset_dir_path == pathlib.Path("CIFAR10"):
    # print("Loading CIFAR100 dataset...")
    # dataset = torchvision.datasets.CIFAR100(
    #     root="./data",
    #     train=False,
    #     download=True,
    #     transform=transform
    # )
    dataset = ImageFolder(root=args.dataset_dir_path, transform=transform)
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
    z_all = features["global_pool"].numpy()
    penultimate = features["layer4_raw"].numpy()
    print(f"penultimate.shape: {penultimate.shape}")

    # # 空間相関ρ̂を推定して動的係数aを計算
    # N, C, H, W = penultimate.shape
    # penultimate_flat = penultimate.reshape(N, C, H * W)  # [N, 2048, 49]
    # # 各チャネルごとに相関係数を計算
    # rho_list = []
    # iu = np.triu_indices(H*W, k=1)
    # for c in range(C):
    #     # サンプル軸でセンタリング
    #     x = penultimate_flat[:, c, :]                     # [N,49]
    #     # 各サンプルごとに平均を引く
    #     x_centered = x - x.mean(axis=0, keepdims=True)
    #     # 列（空間ピクセル位置）ごとの相関行列
    #     corr = np.corrcoef(x_centered, rowvar=False)  # (49,49)

    #     # 上三角のオフダイアゴナル要素を抽出して NaN を除去
    #     vals = corr[iu]
    #     vals = vals[np.isfinite(vals)]
    #     if vals.size > 0:
    #         rho_list.append(vals.mean())
    #     # rho_list.append(corr[iu].mean())
    # rho_hat = np.mean(rho_list)

    # # もし 1 チャネルも計算できなかったら 0 にフォールバック
    # if len(rho_list) == 0:
    #     rho_hat = 0.0
    # else:
    #     rho_hat = float(np.mean(rho_list))

    # with np.errstate(invalid='ignore', divide='ignore'):
    #     a = np.sqrt(H*W / (1 + (H*W - 1) * rho_hat))
    # print(f"rho_hat={rho_hat:.4f}, a={a:.4f}🚀")

    # # NaN になっていないか念のためチェックしてからスケール
    # if np.isfinite(a):
    #     z_all = z_all * a
    # else:
    #     # 異常時はスケールしない or デフォルト値を使う
    #     print("Warning: computed 'a' is NaN or Inf; skipping rescale.")

    # # rho_hatとaをargs.jsonに追記するぞ！
    # with arguments_output_path.open("r", encoding="utf-8") as f:
    #     args_dict = json.load(f)
    # args_dict["rho_hat"] = float(rho_hat)
    # args_dict["dynamic_coefficient_a"] = float(a)
    # with arguments_output_path.open("w", encoding="utf-8") as f:
    #     json.dump(args_dict, f, indent=4, default=path_converter)

    ########################################################################実験中
    # 空間相関 ρ をチャンネルごとに推定して動的係数 a_c を計算
    # N, C, H, W = penultimate.shape
    # penultimate_flat = penultimate.reshape(N, C, H * W)  # [N, C, H*W]
    # iu = np.triu_indices(H * W, k=1)

    # rho_per_channel = []
    # for c in range(C):
    #     x = penultimate_flat[:, c, :]                   # [N, H*W]
    #     # 各ピクセル位置でセンタリング
    #     x_centered = x - x.mean(axis=0, keepdims=True)  # [N, H*W]

    #     # ① 列ごとの標準偏差を計算し、ほぼゼロ分散列を除外
    #     std = x_centered.std(axis=0, ddof=0)            # [H*W]
    #     valid_cols = std > 1e-6                         # 閾値は適宜調整
    #     if valid_cols.sum() < 2:
    #         # 有効な列が少なすぎると相関が取れないので ρ=0 にフォールバック
    #         rho_per_channel.append(0.0)
    #         continue

    #     x_valid = x_centered[:, valid_cols]             # [N, #valid]
    #     corr = np.corrcoef(x_valid, rowvar=False)       # (#valid, #valid)

    #     vals = corr[np.triu_indices(corr.shape[0], k=1)]
    #     vals = vals[np.isfinite(vals)]
    #     rho_per_channel.append(vals.mean() if vals.size > 0 else 0.0)

    # rho_per_channel = np.array(rho_per_channel, dtype=float)  # shape [C]

    # # 各チャンネルに対応する a_c を計算
    # with np.errstate(invalid='ignore', divide='ignore'):
    #     a_per_channel = np.sqrt((H * W) / (1 + (H * W - 1) * rho_per_channel))  # shape [C]

    # # NaN／Inf を 1 にフォールバック
    # a_per_channel = np.where(np.isfinite(a_per_channel), a_per_channel, 1.0)

    # print(f"mean rho: {rho_per_channel.mean():.4f}, a per channel sample: {a_per_channel[:5]}")
    # # z_all にチャンネルごとのスケーリングを適用
    # z_all = z_all * a_per_channel[None, :]

    # # args.json にも rho_per_channel と a_per_channel の統計を追記
    # with arguments_output_path.open("r", encoding="utf-8") as f:
    #     args_dict = json.load(f)
    # args_dict["rho_mean"] = float(rho_per_channel.mean())
    # args_dict["a_mean"]   = float(a_per_channel.mean())
    # with arguments_output_path.open("w", encoding="utf-8") as f:
    #     json.dump(args_dict, f, indent=4, default=path_converter)


    #########################################################################

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
