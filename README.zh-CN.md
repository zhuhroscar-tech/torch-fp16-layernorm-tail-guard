[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-fp16-layernorm-tail-guard

为 PyTorch 的 CPU `float16` LayerNorm 提供显式调用的规避方案和诊断工具。某些构建在处理常量行时会输出非零值；没有 affine bias 时，这类输入的归一化结果本应为零。误差可能只出现在向量尾部，也可能覆盖整行。请在自己的环境中运行诊断，不要直接套用其他平台的结果。

## 安装与检查

需要 Python 3.9+ 和与当前环境兼容的 PyTorch；可选依赖声明为 `torch>=2.0`。

```bash
git clone https://github.com/zhuhroscar-tech/torch-fp16-layernorm-tail-guard.git
cd torch-fp16-layernorm-tail-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[torch]"
torch-fp16-layernorm-tail-guard
torch-fp16-layernorm-tail-guard --json
```

如果已经安装了适合当前环境的 PyTorch，可改用 `python -m pip install -e .`。CLI 会重新测试常量行，并将抽样的非常量输入与 float64 参考结果比较。JSON 包含 torch 版本、各项测试结果、`any_bug_present` 和 `guard_fully_correct`。

退出码表示的是 **guard 检查结果**，而不是原生实现是否存在问题：`0` 表示所有诊断用例通过，`1` 表示 guard 检查失败，`2` 表示无法导入 torch。

## 在 Python 中使用

```python
import torch
from torch_fp16_layernorm_tail_guard import safe_layer_norm

x = torch.full((1, 12), 100.0, dtype=torch.float16)
y = safe_layer_norm(x, (12,), eps=1e-5)
```

`safe_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5)` 将 CPU float16 的输入、weight 和 bias 转为 float32 后计算，再转回原来的 dtype。其他 dtype/device 组合直接调用 PyTorch 原生实现。

## 适用范围与限制

- **不会** monkey-patch `torch.nn.LayerNorm`，也不会自动保护第三方代码。需要显式替换调用，或自行封装 module。
- CUDA/MPS 输入原样交给 PyTorch；RMSNorm 等其他算子不在检查范围内。
- 类型转换和 float32 计算会增加内存及时间开销，请按实际负载评估。
- 抽样诊断不能证明所有输入的精度；原生实现的表现取决于安装的构建。

## 开发

```bash
python -m pip install -e ".[dev,torch]"
python -m pytest -v --cov=torch_fp16_layernorm_tail_guard
```

参见[实现](src/torch_fp16_layernorm_tail_guard/core.py)、[测试](tests/test_core.py)和 [CI 配置](.github/workflows/ci.yml)。采用 [MIT 许可证](LICENSE)。
