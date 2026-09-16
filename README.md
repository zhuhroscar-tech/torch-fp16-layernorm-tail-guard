# torch-fp16-layernorm-tail-guard

Guards a real, reproduced-from-scratch bug in PyTorch's CPU `float16`
`torch.nn.functional.layer_norm` (and therefore `torch.nn.LayerNorm`):
for an **exact-constant row** (true variance is exactly 0 by
definition) whose normalized length is **not a multiple of 8**, the
last `length % 8` output elements come back **nonzero** instead of the
mathematically guaranteed exact `0.0` -- silently, with no warning,
NaN, or Inf to signal anything went wrong.

```python
import torch

x = torch.full((1, 12), 100.0, dtype=torch.float16)
print(torch.nn.functional.layer_norm(x, (12,), eps=1e-5))
# tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
#          0.0977, 0.0977, 0.0977, 0.0977]])
#                                     ^^^^^^ 4 = 12 % 8 elements wrong;
#                                     every one of them MUST be exactly 0
```

## Why this is a distinct bug, not a duplicate

Several existing, well-known PyTorch issues touch float16 LayerNorm:

- [pytorch/pytorch#66707](https://github.com/pytorch/pytorch/issues/66707)
  ("layer_norm needs to be done in fp32 for fp16 inputs") -- about
  general fp16-accumulation *precision loss*, not an exact-zero-result
  guarantee being violated.
- [pytorch/pytorch#170758](https://github.com/pytorch/pytorch/issues/170758)
  ("bf16 layernorm has wrong answer") -- **bfloat16**, not float16;
  affects the *entire* output, not a `length % 8`-sized tail; open as
  "expected due to reduced-precision accumulation" per a PyTorch
  collaborator's comment, i.e. accepted as unavoidable rounding, not a
  correctness violation the way an exact-zero case is.
- [pytorch/pytorch#173885](https://github.com/pytorch/pytorch/issues/173885)
  -- Inductor-compiled CPU float16 LayerNorm turning `Inf` input into
  `NaN`; a `torch.compile`-specific Welford-accumulator issue, already
  fixed by PR #173989, and about Inf input, not exact-constant input.

None of these describe the specific signature reproduced here: eager
mode (no `torch.compile`), plain `float16` (not `bfloat16`), an
**exact-constant** (zero-variance) input where the mathematically
correct answer is unambiguous (`0.0`, not merely "close to some
value"), and a failure confined precisely to a `length % 8`-sized
tail. A search of the PyTorch issue tracker at the time of writing did
not turn up an existing report matching this exact signature; if one
is found later, this README will be updated to reference it rather
than claim to be first.

## The bug, characterized

Reproduced by brute-force sweep over lengths 1-39 with a constant
`100.0` input:

```
length= 7  nonzero=7   (all wrong -- length < one full vector)
length= 8  nonzero=0   (correct -- exact multiple of 8)
length= 9  nonzero=1   (length % 8 == 1)
length=12  nonzero=4   (length % 8 == 4)
length=16  nonzero=0   (correct -- exact multiple of 8)
length=20  nonzero=4   (length % 8 == 4)
```

The nonzero count exactly equals `length % 8` at every length tested,
and every exact multiple of 8 is correct. This is consistent with the
CPU kernel's SIMD vectorization width (8 float16 lanes on the AVX2/
AVX512 paths PyTorch's `layer_norm_kernel.cpp` dispatches to) mishandling
its scalar remainder/tail loop specifically for the exact-zero-variance
case -- ordinary (non-constant) float16 input at the same lengths
matches a float32-upcast reference to full float16 precision with zero
tail-specific divergence (see `test_nonconstant_input_at_least_as_
accurate_as_native` in the test suite), and float32/float64 LayerNorm
on the identical constant input is correct at every length tested. Not
every constant *value* triggers it (some, like `1.0` or `10.0`, happen
to land on a bit pattern where the buggy path is a no-op), but a wide
majority of realistic activation magnitudes do (confirmed: 25/33
sampled values from 0.1 to 60000 trigger it at length 12).

## Usage

```python
from torch_fp16_layernorm_tail_guard import safe_layer_norm

# Drop-in replacement for torch.nn.functional.layer_norm.
# float16-on-CPU inputs are computed at float32 internally and cast
# back; every other dtype/device delegates to the real function
# unchanged.
y = safe_layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=1e-5)
```

Or run the CLI to check whether the bug reproduces on your installed
torch build and confirm the guard is correct on every case:

```bash
torch-fp16-layernorm-tail-guard
torch-fp16-layernorm-tail-guard --json
```

## Independent verification

- **Bug reproduction**: `tests/test_core.py::TestNativeBugReproduction`
  reruns the exact repro above from scratch against whatever torch
  build is installed (never trusts a cached claim), and asserts the
  precise `nonzero_count == length % 8` signature across lengths 1-24
  -- not just "some elements are wrong somewhere."
- **Guard correctness on the trigger case**: an independent
  mathematical oracle needs no reference implementation --
  `(x - mean)` is exactly `0` for constant input at any dtype, so
  `safe_layer_norm` is asserted to return *exactly* `0.0` (not
  "close to 0") on every constant row from length 1 to 24, at 5
  different magnitudes (`0.1` to `60000.0`).
- **Guard transparency on ordinary input**: for genuinely non-constant
  input, `safe_layer_norm`'s output is compared against an independent
  float64 oracle (`layer_norm` computed on the `.double()`-upcast
  input) and asserted to be at least as accurate as the native
  float16 path at 5 lengths spanning both multiples of 8 (unaffected
  by the bug) and non-multiples (affected) -- proving the fix does not
  regress ordinary usage.
- 21/21 tests pass locally (macOS, torch 2.14.0 CPU) with 100%
  statement coverage of `core.py`; CI reruns the identical suite on
  `ubuntu-latest` (the real target environment for this kind of
  CPU-kernel bug) and `macos-latest` across Python 3.10/3.12.

## Reproducible build & test

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,torch]"
pytest -v --cov=torch_fp16_layernorm_tail_guard --cov-report=term-missing
torch-fp16-layernorm-tail-guard
```

CI builds a wheel and sdist on `ubuntu-latest`, computes
`SHA256SUMS.txt`, and smoke-tests the built wheel (installed into a
clean venv) before a release is published.

## Limitations

- This guards `torch.nn.functional.layer_norm` specifically (and by
  extension `torch.nn.LayerNorm`, which calls it). `RMSNorm` and other
  normalization ops were not checked for the same tail-corruption
  pattern; the same technique (upcast to float32, compute, cast back)
  would generalize trivially if one were found affected.
- Only CPU float16 is guarded (the documented trigger). CUDA/MPS
  float16 LayerNorm was not tested and is not known to share this
  specific bug -- the guard's dtype/device check only intervenes for
  `dtype == torch.float16 and device.type == "cpu"`, so GPU calls
  always pass through unguarded and unchanged.
- The upcast-to-float32 fix trades a small, fixed amount of extra
  compute (one dtype cast in, one out) for correctness; this is the
  same tradeoff real production LayerNorm/RMSNorm implementations
  already make deliberately (see pytorch/pytorch#66707 and HF
  Transformers' `Fp32LayerNorm`), not a novel performance cost.
- This does not patch or monkey-patch `torch.nn.LayerNorm` itself;
  callers must explicitly use `safe_layer_norm` (or wrap their own
  `nn.LayerNorm` subclass around it) rather than getting automatic
  protection from unmodified third-party code that calls the native
  function directly.

## License

MIT
