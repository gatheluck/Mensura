import re
from pathlib import Path
import matplotlib.pyplot as plt

jsonl_path = Path('outputs/compute_complexity_measure/20250528_053749_model=resnet50_weights=weights/cifar10/pt_imagenet_ft_CIFAR10_resnet50_epoch90.pth/complexity_measure.jsonl')

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

# import csv

# # CSVに書き出し
# csv_path = "cifar10_2.csv"
# with open(csv_path, "w", newline="") as f:
#     writer = csv.writer(f)
#     # 1行目のヘッダー
#     # header = ["Value"] + [str(i+1) for i in range(len(values_list))]
#     # writer.writerow(header)
#     # 2行目に値を1行で
#     writer.writerow(["tileDB"] + values_list)

# print(f"やったー！values_listを {csv_path} に保存したよ😆✨")

import numpy as np
# NaNを除外して計算
clean_vals = [v for v in values_list if not np.isnan(v)]
mean_val = np.mean(clean_vals)
median_val = np.median(clean_vals)

print(f"平均値: {mean_val:.4f}")
print(f"中央値: {median_val:.4f}")


