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
  a PyTorch collaborator's own comment treats it as expected reduced-
  precision-accumulation behavior, i.e. accepted as unavoidable
  rounding, not a correctness violation the way an exact-zero case is.
- [pytorch/pytorch#173885](https://github.com/pytorch/pytorch/issues/173885)
  -- Inductor-compiled CPU float16 LayerNorm turning `Inf` input into
  `NaN`; a `torch.compile`-specific Welford-accumulator issue, already
  fixed by PR #173989, and about Inf input, not exact-constant input.

None of these describe the specific signature reproduced here: eager
mode (no `torch.compile`), plain `float16` (not `bfloat16`), an
**exact-constant** (zero-variance) input where the mathematically
correct answer is unambiguous (`0.0`, not merely "close to some
value"). A search of the PyTorch issue tracker at the time of writing
did not turn up an existing report matching this exact signature; if
one is found later, this README will be updated to reference it
rather than claim to be first.

## The bug, characterized (two distinct shapes observed in this project's own CI)

This package's own CI (`.github/workflows/ci.yml`) confirmed the bug
is real on **both** `ubuntu-latest` and `macos-latest`, but its exact
failure shape differs by platform/build -- an honest finding from
actually running on both, not assumed:

**macOS (this development host, Accelerate-linked torch 2.14.0 CPU
build):** the failure is confined to a `length % 8`-sized tail. Sweep
over lengths 1-39 with constant input `100.0`:

```
length= 7  nonzero=7   (all wrong -- length < one full vector)
length= 8  nonzero=0   (correct -- exact multiple of 8)
length= 9  nonzero=1   (length % 8 == 1)
length=12  nonzero=4   (length % 8 == 4)
length=16  nonzero=0   (correct -- exact multiple of 8)
```

The nonzero count exactly equals `length % 8` at every length tested
on this platform, consistent with the CPU kernel's SIMD vectorization
width (8 float16 lanes) mishandling its scalar remainder/tail loop for
the exact-zero-variance case specifically.

**Linux (GitHub Actions `ubuntu-latest`, `torch==2.14.0+cu130` wheel,
CPU code path since the runner has no GPU):** the failure is instead
uniform across the *entire* row, by a small fixed amount, independent
of length:

```
length=8:  [0.0009765625] * 8
length=12: [0.0009765625] * 12
length=16: [0.0009765625] * 16
```

This is a **different failure shape** from the macOS tail pattern --
this package does not claim a single universal root cause across
platforms/builds, only that both are real, both violate the same
unambiguous correctness standard (an exact-zero-variance row's true
output is exactly `0.0`, not "close to 0"), and both are fixed by the
same guard. `diagnose()`/the CLI record which of the two known shapes
(or neither, which would indicate a third, previously-unseen shape) a
given host exhibits, rather than asserting one platform's pattern
universally. Ordinary (non-constant) float16 input at the same
lengths matches a float32-upcast reference to full float16 precision
on both platforms tested (see `test_nonconstant_input_at_least_as_
accurate_as_native` in the test suite), and float32/float64 LayerNorm
on the identical constant input is correct at every length tested on
both platforms. Not every constant *value* triggers the bug on macOS
(some, like `1.0` or `10.0`, happen to land on a bit pattern where the
buggy path is a no-op), but a wide majority of realistic activation
magnitudes do (confirmed: 25/33 sampled values from 0.1 to 60000
trigger it at length 12 on macOS).

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
