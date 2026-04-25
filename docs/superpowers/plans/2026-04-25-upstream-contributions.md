# Upstream Contributions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Open 5 focused, polished PRs from marqov-dev/quantumflow against gecrooks/quantumflow covering bug fixes and test improvements.

**Architecture:** Each PR gets its own branch off `upstream/master`. Commits are cherry-picked from our fork's history onto that branch, pushed to `origin`, then a PR is opened against `gecrooks/quantumflow` via `gh pr create --repo gecrooks/quantumflow`. PRs are submitted in order.

**Tech Stack:** git, GitHub CLI (`gh`)

---

## Pre-flight checks

- [ ] **Verify upstream remote is configured**

```bash
git remote -v
```

Expected: `upstream  https://github.com/gecrooks/quantumflow.git` in the output.

- [ ] **Fetch latest upstream**

```bash
git fetch upstream
```

- [ ] **Verify gh is authenticated**

```bash
gh auth status
```

Expected: logged in, no errors.

---

## Task 1: PR — Input validation (assert → proper exceptions)

**Branch:** `upstream-fix/input-validation`
**Commits to cherry-pick:** `c92066b`, `c69bf6d`, `5faf212`, `66af9f5`
**Files touched:** `circuits.py`, `channels.py`, `dagcircuit.py`, `gradients.py`, `gradients_test.py`, `paulialgebra.py`, `paulialgebra_test.py`, `tensors.py`, `tensors_test.py`, `visualization.py`, `visualization_test.py`

- [ ] **Create branch off upstream/master**

```bash
git checkout -b upstream-fix/input-validation upstream/master
```

- [ ] **Cherry-pick the four commits**

```bash
git cherry-pick c92066b c69bf6d 5faf212 66af9f5
```

If any conflict arises, the upstream code still has the old `assert` or missing `f` prefix — keep our version (the one with `raise` / the f-string). Run `git cherry-pick --continue` after resolving.

- [ ] **Run the tests to confirm nothing is broken**

```bash
venv/bin/python -m pytest quantumflow/circuits.py quantumflow/channels.py quantumflow/dagcircuit.py quantumflow/gradients_test.py quantumflow/paulialgebra_test.py quantumflow/tensors_test.py quantumflow/visualization_test.py -x -q 2>/dev/null || uv run pytest quantumflow/ -x -q --ignore=quantumflow/transpile_test.py -q
```

Expected: all tests pass, no failures.

- [ ] **Push branch**

```bash
git push origin upstream-fix/input-validation
```

- [ ] **Open PR against upstream**

```bash
gh pr create --repo gecrooks/quantumflow \
  --title "Replace assert statements with proper exceptions for input validation" \
  --body "$(cat <<'EOF'
## Summary

`assert` statements are disabled when Python runs in optimized mode (`python -O`), making them unreliable for input validation. This PR replaces user-facing assertions with explicit exceptions across four files.

### Changes

**`circuits.py`** — Fix missing `f` prefix on error message string; `{list(qbs)}` was being printed literally instead of interpolated. Also adds a missing space before "but".

**`channels.py` — `channel_to_kraus()`** — Replace `assert` on Choi matrix eigenvalues with `ValueError`. Without this fix, `python -O` silently produces NaN Kraus operators from `sqrt()` of negative numbers. Adds a small tolerance for numerical noise and clamps tiny negatives to zero before `sqrt`.

**`dagcircuit.py` — `next_element()` / `prev_element()`** — Replace `assert False` with `KeyError` containing a descriptive message. `assert False` provides no diagnostic information and is a no-op under `-O`.

**`gradients.py`, `paulialgebra.py`, `tensors.py`, `visualization.py`** — Replace remaining user-input `assert` statements with `ValueError` / `TypeError` with descriptive messages. Internal algorithm invariants (where `assert` is appropriate) are left unchanged.

Tests are added for all new exception paths.

## Testing

All existing tests continue to pass. New tests verify the exception messages and types for each changed code path.
EOF
)"
```

Expected: PR URL printed. Note it down.

---

## Task 2: PR — Compatibility (Qiskit 3.0 + PyQuil 4.0)

**Branch:** `upstream-fix/compatibility-deprecations`
**Commits to cherry-pick:** `c4c9d06`, `f7ccca3`
**Files touched:** `xqiskit.py`, `xforest.py`

- [ ] **Create branch off upstream/master**

```bash
git checkout -b upstream-fix/compatibility-deprecations upstream/master
```

- [ ] **Cherry-pick the two commits**

```bash
git cherry-pick c4c9d06 f7ccca3
```

- [ ] **Push branch**

```bash
git push origin upstream-fix/compatibility-deprecations
```

- [ ] **Open PR against upstream**

```bash
gh pr create --repo gecrooks/quantumflow \
  --title "Fix Qiskit 3.0 and PyQuil 4.0 deprecation warnings" \
  --body "$(cat <<'EOF'
## Summary

Two one-line compatibility fixes to silence deprecation warnings against current backend versions.

### Changes

**`xqiskit.py` — `qiskit_to_circuit()`** — Replace deprecated tuple unpacking of `CircuitInstruction` with named attribute access (`.operation`, `.qubits`, `.clbits`). Tuple unpacking was deprecated in Qiskit 1.2 and removed in Qiskit 3.0. Also removes dead commented-out code referencing `instruction.condition`, which was removed in Qiskit 2.0.

**`xforest.py` — `pyquil_to_circuit()`** — Replace deprecated `.qubits` property with `get_qubit_indices()`. The old property emits 19 deprecation warnings per test run under PyQuil 4.0.

## Testing

Existing tests pass. No behaviour change — purely API alignment with current library versions.
EOF
)"
```

---

## Task 3: PR — Dependency version constraints

**Branch:** `upstream-fix/dependency-constraints`
**Commits to cherry-pick:** `7f41aac`
**Files touched:** `pyproject.toml`

- [ ] **Create branch off upstream/master**

```bash
git checkout -b upstream-fix/dependency-constraints upstream/master
```

- [ ] **Cherry-pick the commit**

```bash
git cherry-pick 7f41aac
```

If there is a conflict on `pyproject.toml` (upstream may have touched it independently), keep upstream's structure but apply our version constraint additions: `numpy >= 1.21, < 2.0`, `scipy >= 1.9, < 2.0`, `networkx >= 2.6, < 4.0`, `matplotlib >= 3.5`, `pillow >= 9.0`.

- [ ] **Push branch**

```bash
git push origin upstream-fix/dependency-constraints
```

- [ ] **Open PR against upstream**

```bash
gh pr create --repo gecrooks/quantumflow \
  --title "Add version constraints to core dependencies" \
  --body "$(cat <<'EOF'
## Summary

Several core dependencies had no version bounds, which can cause silent breakage when a major version introduces incompatible changes. This PR adds explicit lower and upper bounds.

### Changes in `pyproject.toml`

| Dependency | Before | After |
|------------|--------|-------|
| numpy | `< 2.0` | `>= 1.21, < 2.0` |
| scipy | (none) | `>= 1.9, < 2.0` |
| networkx | (none) | `>= 2.6, < 4.0` |
| matplotlib | (none) | `>= 3.5` |
| pillow | (none) | `>= 9.0` |

Lower bounds are set to the oldest versions known to work with the current API usage. Upper bounds cap at the next breaking major where the API has changed.

## Testing

No behaviour change. Existing tests pass.
EOF
)"
```

---

## Task 4: PR — Transpile test coverage

**Branch:** `upstream-test/transpile-coverage`
**Commits to cherry-pick:** `dddf91c`
**Files touched:** `transpile_test.py`, `pyproject.toml`

- [ ] **Create branch off upstream/master**

```bash
git checkout -b upstream-test/transpile-coverage upstream/master
```

- [ ] **Cherry-pick the commit**

```bash
git cherry-pick dddf91c
```

If there is a conflict on `pyproject.toml`, keep the upstream `pyproject.toml` base and manually apply only the `filterwarnings` additions from this commit (the lines that suppress the pyquil internal deprecation warning). Do not re-apply the version constraint changes from Task 3 — those belong in their own PR.

If there is a conflict on `transpile_test.py`, upstream has a minimal version of this file. Keep our full version (the one with 30 additional tests).

- [ ] **Verify the new tests pass**

```bash
venv/bin/python -m pytest quantumflow/transpile_test.py -v 2>/dev/null || uv run pytest quantumflow/transpile_test.py -v
```

Expected: all tests pass. Tests that require `qsimcirq` will be skipped if it is not installed — that is correct behaviour.

- [ ] **Push branch**

```bash
git push origin upstream-test/transpile-coverage
```

- [ ] **Open PR against upstream**

```bash
gh pr create --repo gecrooks/quantumflow \
  --title "Improve transpile.py test coverage from 7% to 92%" \
  --body "$(cat <<'EOF'
## Summary

`transpile_test.py` had 7% coverage because the entire test module was skipped when `qsimcirq` was not installed. This PR restructures the tests so the `qsim`-specific cases are skipped individually while all other format tests run normally.

### Changes

**`transpile_test.py`**
- Remove module-level `pytest.importorskip("qsimcirq")` that skipped everything
- Add `@pytest.mark.skipif` on the two tests that actually require `qsimcirq`
- Add individual tests for each format detection (`detect_format`)
- Add edge case tests: empty circuits, single-qubit circuits, default output format
- Add error handling tests: unknown input format, unknown output format
- Suppress pyquil's own internal deprecation warning (upstream bug: rigetti/pyquil#1720) via `pytest` `filterwarnings` config

**Result:** 30 new tests, coverage 7% → 92%.

## Testing

```
pytest quantumflow/transpile_test.py -v
```

All 30+ tests pass. `qsim`-specific tests are skipped when `qsimcirq` is unavailable.
EOF
)"
```

---

## Task 5: PR — Gatesets test suite

**Branch:** `upstream-test/gatesets`
**Commits to cherry-pick:** `904afaa`
**Files touched:** `gatesets_test.py` (new file)

- [ ] **Create branch off upstream/master**

```bash
git checkout -b upstream-test/gatesets upstream/master
```

- [ ] **Cherry-pick the commit**

```bash
git cherry-pick 904afaa
```

This commit creates a new file (`gatesets_test.py`) — no conflicts expected.

- [ ] **Verify the tests pass**

```bash
venv/bin/python -m pytest quantumflow/gatesets_test.py -v 2>/dev/null || uv run pytest quantumflow/gatesets_test.py -v
```

Expected: all tests pass.

- [ ] **Push branch**

```bash
git push origin upstream-test/gatesets
```

- [ ] **Open PR against upstream**

```bash
gh pr create --repo gecrooks/quantumflow \
  --title "Add test suite for gatesets module" \
  --body "$(cat <<'EOF'
## Summary

`gatesets.py` had no tests. This PR adds `gatesets_test.py` with 100% coverage.

### Tests added

- All gate sets are non-empty
- All entries in each gate set are valid `Gate` / `Operation` subclasses
- Subset relationships are correct (`QSIM_GATES ⊆ CIRQ_GATES`)
- Common gates (H, X, CNOT) are present across all hardware backend sets
- Terminal gate sets contain the expected decomposition targets
- `__all__` exports match the module's public names

## Testing

```
pytest quantumflow/gatesets_test.py -v
```

All tests pass. Coverage for `gatesets.py`: 0% → 100%.
EOF
)"
```

---

## Wrap-up

- [ ] **Return to main branch**

```bash
git checkout main
```

- [ ] **Verify all 5 PRs are open**

```bash
gh pr list --repo gecrooks/quantumflow
```

Expected: 5 open PRs from `marqov-dev`.
