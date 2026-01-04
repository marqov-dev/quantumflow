# Copyright 2019-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Unit tests for quantumflow.metrics"""

import pytest

import quantumflow as qf


def test_circuit_metrics_dataclass() -> None:
    """Test CircuitMetrics dataclass basic functionality."""
    metrics = qf.CircuitMetrics(
        depth=5,
        size=10,
        width=3,
        two_qubit_count=4,
        two_qubit_depth=3,
        gate_counts={qf.H: 3, qf.CNot: 4},
    )

    assert metrics.depth == 5
    assert metrics.size == 10
    assert metrics.width == 3
    assert metrics.two_qubit_count == 4
    assert metrics.two_qubit_depth == 3
    assert metrics.gate_counts[qf.H] == 3
    assert metrics.gate_counts[qf.CNot] == 4


def test_analyze_empty_circuit() -> None:
    """Test analyze_circuit with empty circuit."""
    circ = qf.Circuit()
    metrics = qf.analyze_circuit(circ)

    assert metrics.depth == 0
    assert metrics.size == 0
    assert metrics.width == 0
    assert metrics.two_qubit_count == 0
    assert metrics.two_qubit_depth == 0
    assert metrics.gate_counts == {}


def test_analyze_single_gate() -> None:
    """Test analyze_circuit with single-gate circuit."""
    circ = qf.Circuit([qf.H(0)])
    metrics = qf.analyze_circuit(circ)

    assert metrics.depth == 1
    assert metrics.size == 1
    assert metrics.width == 1
    assert metrics.two_qubit_count == 0
    assert metrics.two_qubit_depth == 0
    assert qf.H in metrics.gate_counts


def test_analyze_bell_state() -> None:
    """Test analyze_circuit with Bell state circuit."""
    circ = qf.Circuit([qf.H(0), qf.CNot(0, 1)])
    metrics = qf.analyze_circuit(circ)

    assert metrics.depth == 2
    assert metrics.size == 2
    assert metrics.width == 2
    assert metrics.two_qubit_count == 1
    assert metrics.two_qubit_depth == 1


def test_analyze_ghz_circuit() -> None:
    """Test analyze_circuit with GHZ circuit."""
    N = 5
    circ = qf.ghz_circuit(range(N))
    metrics = qf.analyze_circuit(circ)

    assert metrics.size == N  # 1 H + (N-1) CNots
    assert metrics.width == N
    assert metrics.two_qubit_count == N - 1  # CNots


def test_analyze_parallel_circuit() -> None:
    """Test analyze_circuit with parallel gates."""
    circ = qf.Circuit()
    circ += qf.H(0)
    circ += qf.H(1)
    circ += qf.H(2)

    metrics = qf.analyze_circuit(circ)

    assert metrics.depth == 1  # All parallel
    assert metrics.size == 3
    assert metrics.width == 3
    assert metrics.two_qubit_count == 0
    assert metrics.two_qubit_depth == 0


def test_analyze_mixed_circuit() -> None:
    """Test with circuit containing both 1 and 2 qubit gates."""
    circ = qf.Circuit()
    circ += qf.H(0)
    circ += qf.X(1)
    circ += qf.CNot(0, 1)
    circ += qf.Y(0)
    circ += qf.CZ(0, 1)

    metrics = qf.analyze_circuit(circ)

    assert metrics.size == 5
    assert metrics.width == 2
    assert metrics.two_qubit_count == 2  # CNot + CZ


def test_two_qubit_ratio() -> None:
    """Test two_qubit_ratio property."""
    # Circuit with 2 out of 4 gates being 2-qubit
    circ = qf.Circuit([qf.H(0), qf.H(1), qf.CNot(0, 1), qf.CZ(0, 1)])
    metrics = qf.analyze_circuit(circ)

    assert metrics.two_qubit_ratio == 0.5

    # Empty circuit edge case
    empty_metrics = qf.analyze_circuit(qf.Circuit())
    assert empty_metrics.two_qubit_ratio == 0.0


def test_gate_counts_consistency() -> None:
    """Test that gate_counts matches count_operations."""
    circ = qf.ghz_circuit(range(4))
    metrics = qf.analyze_circuit(circ)

    expected_counts = qf.count_operations(circ)
    assert metrics.gate_counts == expected_counts


def test_analyze_with_swap() -> None:
    """Test circuit with Swap gate (2-qubit)."""
    circ = qf.Circuit([qf.Swap(0, 1)])
    metrics = qf.analyze_circuit(circ)

    assert metrics.two_qubit_count == 1
    assert metrics.size == 1


def test_analyze_with_ccnot() -> None:
    """Test circuit with CCNot gate (3-qubit, not counted as 2-qubit)."""
    circ = qf.Circuit([qf.CCNot(0, 1, 2)])
    metrics = qf.analyze_circuit(circ)

    assert metrics.two_qubit_count == 0  # CCNot is 3-qubit, not 2-qubit
    assert metrics.size == 1
    assert metrics.width == 3


# --- BackendReport and TranspilationReport tests ---


def test_backend_report_overhead_calculations() -> None:
    """Test BackendReport overhead property calculations."""
    original = qf.CircuitMetrics(
        depth=10,
        size=20,
        width=2,
        two_qubit_count=5,
        two_qubit_depth=5,
        gate_counts={},
    )
    transpiled = qf.CircuitMetrics(
        depth=15,
        size=30,
        width=2,
        two_qubit_count=8,
        two_qubit_depth=8,
        gate_counts={},
    )

    report = qf.BackendReport(backend="test", original=original, transpiled=transpiled)

    assert report.depth_overhead == 1.5
    assert report.size_overhead == 1.5
    assert report.two_qubit_overhead == 1.6


def test_backend_report_edge_cases() -> None:
    """Test BackendReport with zero original values."""
    original = qf.CircuitMetrics(
        depth=0,
        size=0,
        width=0,
        two_qubit_count=0,
        two_qubit_depth=0,
        gate_counts={},
    )
    transpiled = qf.CircuitMetrics(
        depth=0,
        size=0,
        width=0,
        two_qubit_count=0,
        two_qubit_depth=0,
        gate_counts={},
    )

    report = qf.BackendReport(backend="test", original=original, transpiled=transpiled)

    assert report.depth_overhead == 0.0
    assert report.size_overhead == 0.0
    assert report.two_qubit_overhead == 0.0


def test_compare_backends_basic() -> None:
    """Test compare_backends with a simple circuit."""
    circ = qf.Circuit([qf.H(0), qf.CNot(0, 1)])
    report = qf.compare_backends(circ, backends=["qiskit", "cirq"])

    assert "qiskit" in report.backends
    assert "cirq" in report.backends
    assert report.original.size == 2


def test_compare_backends_with_translation() -> None:
    """Test that Can gate expands differently per backend."""
    # Can gate is not native to most backends, should expand
    circ = qf.Circuit([qf.Can(0.1, 0.2, 0.3, 0, 1)])
    report = qf.compare_backends(circ)

    # Different backends should have different transpiled sizes
    sizes = {b: r.transpiled.size for b, r in report.backends.items()}
    # At least one backend should have expanded the circuit
    assert any(s > 1 for s in sizes.values())


def test_transpilation_report_best_methods() -> None:
    """Test best_for_depth and best_for_two_qubit methods."""
    circ = qf.Circuit([qf.Can(0.1, 0.2, 0.3, 0, 1)])
    report = qf.compare_backends(circ)

    # Should return valid backend names
    assert report.best_for_depth() in report.backends
    assert report.best_for_two_qubit() in report.backends


def test_transpilation_report_summary() -> None:
    """Test summary table generation."""
    circ = qf.ghz_circuit(range(3))
    report = qf.compare_backends(circ, backends=["qiskit", "braket"])

    summary = report.summary()
    assert "Backend" in summary
    assert "qiskit" in summary
    assert "braket" in summary
    assert "Depth" in summary
    assert "2Q Gates" in summary


def test_compare_backends_empty_circuit() -> None:
    """Test compare_backends handles empty circuits."""
    circ = qf.Circuit()
    report = qf.compare_backends(circ, backends=["qiskit"])

    assert report.original.size == 0
    assert report.backends["qiskit"].transpiled.size == 0


def test_compare_backends_unknown_backend() -> None:
    """Test that unknown backend raises error."""
    circ = qf.Circuit([qf.H(0)])
    with pytest.raises(ValueError, match="Unknown backend"):
        qf.compare_backends(circ, backends=["unknown_backend"])


def test_compare_backends_all_supported() -> None:
    """Test compare_backends with all supported backends."""
    circ = qf.Circuit([qf.H(0), qf.CNot(0, 1)])
    report = qf.compare_backends(circ)  # No backends specified = all

    # Should have all supported backends
    assert "braket" in report.backends
    assert "cirq" in report.backends
    assert "qiskit" in report.backends
    assert "pyquil" in report.backends
    assert "qsim" in report.backends


def test_supported_backends_constant() -> None:
    """Test SUPPORTED_BACKENDS is properly exported."""
    assert "braket" in qf.SUPPORTED_BACKENDS
    assert "cirq" in qf.SUPPORTED_BACKENDS
    assert "qiskit" in qf.SUPPORTED_BACKENDS


# fin
