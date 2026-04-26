# QuantumFlow Security & Code Quality Review

**Reviewer:** Marqov
**Date:** 2026-04-25
**Scope:** Full `quantumflow/` package + `examples/` directory at parent commit `f4abeb8`
**Method:** Static analysis of every production `.py` file with verification of each finding by reading the actual code.

## Executive summary

QuantumFlow's core has a notably small attack surface for a library of its size. There is no use of `eval`, `exec`, `pickle`, `marshal`, `yaml.load`, `os.system`, or `shell=True` in production code. The two `subprocess` invocations in production are both in `visualization.py` and call hardcoded executables with list-form arguments. The dynamic-dispatch patterns (`getattr(qkcircuit, name)` in `xqiskit.py`, `getattr(bkcircuit, name)` in `xbraket.py`) are safe — `name` always comes from a hardcoded mapping table.

The findings below are concentrated in the visualization pipeline (real security issue) and a couple of code-quality items elsewhere. Nothing requires emergency action; the LaTeX issue is the only one that warrants prompt attention.

---

## Security findings

### S-1 · LaTeX injection in `visualization.py` → file disclosure (HIGH)

**File:** `quantumflow/visualization.py`
**Lines:** 234, 238, 398, 418, 419, 492–502

`circuit_to_latex()` interpolates several caller-supplied values into a LaTeX document with no escaping, then `latex_to_image()` compiles that document with `pdflatex` (without `-no-shell-escape`).

Injection points:

| Line | Code | Source |
|------|------|--------|
| 234 | `r"\lstick{%s}" % q` | qubit objects from `circ.qubits` (their `__str__`) |
| 238 | `r"\lstick{%s}" % L` | `left_labels` parameter |
| 398 | `r"\rstick{%s}" % L` | `right_labels` parameter |
| 418 | `_QUANTIKZ % (options, ...)` | `options` parameter |
| 419 | `r"\adjustbox{scale=%s}{" % scale` | `scale` parameter |

**Realistic exploit (file disclosure):** A researcher receives a circuit JSON or pickle from a colleague (or downloads supplementary material from a paper, or pulls from a shared S3 bucket). One qubit is named:

```python
"q0}\\input{/Users/victim/.ssh/id_rsa}\\lstick{"
```

When the researcher renders the circuit via `circuit_to_latex(circ)`, `pdflatex` runs `\input{...}` on their SSH private key and embeds the contents into the resulting PDF/PNG. PIL then returns that image to the caller.

**Code-execution exploit (configuration-dependent):** On any pdflatex installation where `shell_escape = t` in `texmf.cnf` (common in Docker images and some CI setups), `\write18{...}` provides full OS command execution. Modern TeX Live defaults to "restricted" shell escape, which whitelists certain commands but is still safer to disable explicitly.

**Recommended fixes:**

1. Add `-no-shell-escape` to the `pdflatex` argument list (line 494 area). One-line change, eliminates the worst case.
2. Escape the 10 LaTeX special characters in any value that comes from caller input. A small helper:

   ```python
   _LATEX_ESCAPES = {
       "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
       "$": r"\$", "&": r"\&", "#": r"\#",
       "_": r"\_", "%": r"\%",
       "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
   }
   def _latex_escape(s: str) -> str:
       return "".join(_LATEX_ESCAPES.get(c, c) for c in s)
   ```

   Apply at lines 234, 238, 398, and to gate names at 273.
3. The `options` parameter is genuinely "dangerous by design" since callers may want to pass real LaTeX options. Either document this clearly ("untrusted callers should not pass `options`") or whitelist a small set of safe option keys.

**Confidence:** 0.95 that file disclosure works on default pdflatex. 0.6 that command execution works depending on installation.

---

## Code-quality findings

### Q-1 · Reference-before-assignment bug in `xbraket.py` (HIGH for correctness, not security)

**File:** `quantumflow/xbraket.py`
**Lines:** 109–122

```python
if isinstance(op, bkAngledGate):
    angle = op.angle
    if op.name in ["XX", "YY", "ZZ"]:
        args = [angle / np.pi] + qubits
    elif name == "XY":
        args = [-0.5 * args[0] / np.pi] + qubits   # <-- bug
    else:
        args = [angle] + qubits
```

In the `XY` branch, `args` is read before being assigned. On the first iteration of the for-loop where this branch is hit, this raises `NameError`. On subsequent iterations it silently uses `args` from the prior gate, producing a wrong (but valid-looking) circuit conversion.

The variable used should clearly be `angle`:

```python
elif name == "XY":
    args = [-0.5 * angle / np.pi] + qubits
```

This bug is not exercised by the existing test suite — `xbraket_test.py:147` only tests `circuit_to_braket` (the output direction) with an XY gate. There is no test for `braket_to_circuit` with an XY-bearing input.

---

### Q-2 · `assert` for input validation in `gates.py` (MEDIUM)

**File:** `quantumflow/gates.py`
**Line:** 235

```python
def __init__(self, target: Gate, controls: Qubits, axes: Optional[str] = None) -> None:
    ...
    if axes is None:
        axes = "Z" * len(controls)
    assert len(axes) == len(controls)        # <-- should be ValueError
```

`axes` is caller-supplied. `assert` is disabled under `python -O` so a length mismatch silently constructs an invalid `ControlGate`. The pattern matches recent fixes elsewhere in the codebase (channel_to_kraus, DAGCircuit traversal) — this one was missed.

Fix:

```python
if len(axes) != len(controls):
    raise ValueError(
        f"axes length ({len(axes)}) must equal number of controls ({len(controls)})"
    )
```

---

### Q-3 · `pdftocairo` invocation lacks `check=True` (LOW)

**File:** `quantumflow/visualization.py`
**Lines:** 504–506

```python
subprocess.run(
    ["pdftocairo", "-singlefile", "-png", "-q", tmppath + ".pdf", tmppath]
)
img = Image.open(tmppath + ".png")
```

Inconsistent with the `pdflatex` call on line 501 (which has `check=True`). If `pdftocairo` fails, `Image.open` raises a `FileNotFoundError` whose message includes the temp directory path. Not a security vulnerability — PIL fails closed rather than returning a corrupted image — but a one-character fix (`check=True`) gives clearer error messages.

---

### Q-4 · Incomplete error handling in `translations.py` (LOW)

**File:** `quantumflow/translate/translations.py`
**Lines:** 37–45

`translation_source_gate()` (line 38) accesses `trans.__annotations__["gate"]` without a `try/except`, in contrast to `translation_target_gates()` immediately below which does. A translation function missing the `gate` annotation produces a bare `KeyError` rather than the helpful `ValueError` used elsewhere.

`translation_target_gates()` itself (line 43) catches only `KeyError`, but `trans.__annotations__["return"].__args__[0]` can also raise `AttributeError` if the return type isn't generic, or `IndexError` if `__args__` is empty.

---

### Q-5 · Gratuitous `shell=True` in test launcher (INFO)

**File:** `examples/examples_test.py`
**Lines:** 17, 22, 27–28, 34, 39–40, 46–47

```python
rval = subprocess.call([os.path.join("examples", "state_prep_w4.py")], shell=True)
```

The path is hardcoded and contains no shell metacharacters, so this isn't exploitable. But the combination of a list argument with `shell=True` is non-idiomatic — Python passes the list joined with spaces to `/bin/sh -c`. The cleanest fix is `subprocess.call([sys.executable, "examples/state_prep_w4.py"])` (drop `shell=True`, run the example explicitly under the current Python interpreter).

---

## Things examined and cleared

The following patterns were searched across the entire codebase and found to be either absent or safely used:

| Pattern | Result |
|---------|--------|
| `eval()` / `exec()` / `compile()` | Only in `stdgates_test.py:60`; restricted namespace, test-only |
| `pickle` / `marshal` / `yaml.load` | Not present |
| `os.system` / `os.popen` | Not present |
| `shell=True` in production code | Not present |
| XML parsing (XXE) | Not present |
| Hardcoded credentials / API keys | Not present |
| `getattr(obj, name)` with untrusted `name` | All 5 sites verified safe — `name` is hardcoded or from internal mapping table |
| HTTP / URL fetching of attacker-controlled targets | Not present |
| `webbrowser.open()` | Used in `xquirk.py:149` — `base_url` is caller-supplied but the realistic threat model (caller passes URL → opens in caller's own browser) doesn't cross a security boundary |
| Path traversal via `open()` | Not present — only `tempfile.TemporaryDirectory()`-derived paths are written |
| Unsafe `tempfile` usage | None — `TemporaryDirectory()` used correctly |

## Strengths worth noting

- 1300+ tests, ~98% line coverage
- Comprehensive `mypy` strict typing
- No use of any of the classic dangerous deserialization or dynamic-execution patterns
- Optional dependencies (`xQiskit`, `xCirq`, etc.) are isolated cleanly so missing backends don't break installs
- Subprocess invocations use list-form arguments (no string concatenation)
- The `pdflatex` call already uses `check=True` and a tempdir

## Suggested triage

| Finding | Severity | Effort | Suggested priority |
|---------|----------|--------|-------------------|
| S-1 LaTeX injection | High | ~30 min | Soon |
| Q-1 `args` reference-before-assignment | High (correctness) | 1 line + a test | Soon |
| Q-2 `assert` → `ValueError` in `gates.py` | Medium | 3 lines | Whenever convenient |
| Q-3 `check=True` on `pdftocairo` | Low | 1 character | Drive-by |
| Q-4 `translations.py` error handling | Low | 5 lines | Drive-by |
| Q-5 `shell=True` in tests | Info | 5-min cleanup | Drive-by |

Marqov is happy to send PRs for any of these in the same focused-PR style as the five fixes already submitted to the repo.
