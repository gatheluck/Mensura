# Mensura

![python versions](https://img.shields.io/badge/python-3.12-blue)
[![MIT License](https://img.shields.io/github/license/gatheluck/Mensura?color=green)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Typing: mypy](https://img.shields.io/badge/typing-mypy-blue)](https://github.com/python/mypy)

## What is Mensura?

TBD

## Sample Code

```python
import timm
from mensura.v_information import compute_complexity_k, get_feature_extractor

model = timm.create_model("resnetv2_50x1_bit.goog_distilled_in1k", pretrained=True).eval()
feature_extractor = get_feature_extractor(model)

# Extract features from the model
x = torch.randn(16, 3, 224, 224)
features = feature_extractor(x)

intermediate_features = [v.flatten(1).detach().numpy() for k, v in features.items() if k != "penultimate"]

z = torch.nn.functional.adaptive_avg_pool2d(features["penultimate"], (1, 1)).flatten(1).detach().numpy()

for z_i in z.T:
	# Compute complexity measure K
	complexity_measure = compute_complexity_k(intermediate_features, z_i)
	print(complexity_measure)
```
