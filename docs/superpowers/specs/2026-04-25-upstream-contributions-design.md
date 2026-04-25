# Upstream Contributions to gecrooks/quantumflow

**Date:** 2026-04-25
**Scope:** Bug fixes and test improvements only (no new features)

## Goal

Submit 5 focused PRs from `marqov-dev/quantumflow` upstream to `gecrooks/quantumflow`. PRs should be polished and easy to review — one clear purpose each, submitted in order.

## PRs

### PR 1 — Input validation (`fix/input-validation`)

**Title:** Replace assert statements with proper exceptions for input validation

**Commits to cherry-pick:**
- `c92066b` Fix missing f-string prefix in Circuit error message
- `c69bf6d` Replace assert with proper validation in channel_to_kraus()
- `5faf212` Replace assert False with KeyError in DAGCircuit methods
- `66af9f5` Replace assert statements with proper exceptions for input validation

**Rationale:** `assert` statements are for internal invariants and are disabled in optimized mode (`python -O`). User-facing errors should raise explicit exceptions with descriptive messages.

---

### PR 2 — Compatibility (`fix/compatibility-deprecations`)

**Title:** Fix Qiskit 3.0 and PyQuil 4.0 deprecation warnings

**Commits to cherry-pick:**
- `c4c9d06` Fix Qiskit 3.0 deprecation warning in qiskit_to_circuit()
- `f7ccca3` Fix PyQuil 4.0 deprecation warning in pyquil_to_circuit()

**Rationale:** Keeps the library clean against current versions of both backends — no behaviour change, just API alignment.

---

### PR 3 — Dependency constraints (`fix/dependency-constraints`)

**Title:** Add version constraints to core dependencies

**Commits to cherry-pick:**
- `7f41aac` Add version constraints to core dependencies

**Rationale:** Unconstrained deps can break silently on upgrades. Explicit bounds make breakage loud and early.

---

### PR 4 — Transpile test coverage (`test/transpile-coverage`)

**Title:** Improve transpile.py test coverage from 7% to 92%

**Commits to cherry-pick:**
- `dddf91c` Improve transpile.py test coverage from 7% to 92%

---

### PR 5 — Gatesets tests (`test/gatesets`)

**Title:** Add test suite for gatesets module

**Commits to cherry-pick:**
- `904afaa` Add test suite for gatesets module

---

## Mechanics

1. Each PR gets its own branch off `upstream/master`
2. Cherry-pick the relevant commits onto that branch
3. Push branch to `marqov-dev/quantumflow` (or a personal fork if preferred)
4. Open PR against `gecrooks/quantumflow` via `gh pr create`
5. Submit in order PR1 → PR5

## What to exclude

- `4ddfbcc` Add code review report — internal marqov document, not appropriate for upstream
- All new features (CircuitMetrics, device noise models, run_and_measure, benchmarking, cross-backend comparison) — to be discussed separately with the maintainer
