[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-fp16-layernorm-tail-guard

A call-site workaround and diagnostic for CPU `float16` LayerNorm on PyTorch builds that produce nonzero output for constant input rows. Without affine bias, those rows should normalize to zero. The observed error can affect a vector tail or an entire row; run the diagnostic on your own build rather than assuming one platform's behavior.

## Install and check

Requires Python 3.9+ and a compatible PyTorch installation (`torch>=2.0` in the optional extra).

```bash
git clone https://github.com/zhuhroscar-tech/torch-fp16-layernorm-tail-guard.git
cd torch-fp16-layernorm-tail-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[torch]"
torch-fp16-layernorm-tail-guard
torch-fp16-layernorm-tail-guard --json
```

If PyTorch is already installed for your environment, use `python -m pip install -e .` instead. The CLI reruns constant-row cases and compares sampled nonconstant inputs with a float64 reference. Its JSON includes the installed torch version, individual cases, `any_bug_present`, and `guard_fully_correct`.

Exit codes describe the **guard check**, not just native bug detection: `0` means the guard passed all diagnostic cases, `1` means a guard check failed, and `2` means torch could not be imported.

## Use in Python

```python
import torch
from torch_fp16_layernorm_tail_guard import safe_layer_norm

x = torch.full((1, 12), 100.0, dtype=torch.float16)
y = safe_layer_norm(x, (12,), eps=1e-5)
```

`safe_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5)` upcasts CPU float16 inputs, weight, and bias to float32, computes normalization, then casts the result back. Other dtype/device combinations delegate unchanged to PyTorch.

## Scope and limitations

- This does **not** monkey-patch `torch.nn.LayerNorm` or protect existing third-party calls. Replace calls explicitly or write your own module wrapper.
- CUDA/MPS inputs pass through unguarded. RMSNorm and other normalization operations are outside scope.
- Extra casts and float32 computation cost memory and time; benchmark your workload.
- Diagnostic samples are not a proof of accuracy for every input. Native behavior depends on the installed build.

## Development

```bash
python -m pip install -e ".[dev,torch]"
python -m pytest -v --cov=torch_fp16_layernorm_tail_guard
```

See [implementation](src/torch_fp16_layernorm_tail_guard/core.py), [tests](tests/test_core.py), and [CI configuration](.github/workflows/ci.yml). Licensed under [MIT](LICENSE).
