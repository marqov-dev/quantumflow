# Copyright 2019-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Unit tests for quantumflow.benchmark"""

import quantumflow as qf


def test_benchmark_circuit_dataclass() -> None:
    """Test BenchmarkCircuit dataclass."""
    circ = qf.Circuit([qf.H(0), qf.CNot(0, 1)])
    bc = qf.BenchmarkCircuit(name="test", circuit=circ, category="entanglement")

    assert bc.name == "test"
    assert bc.category == "entanglement"
    assert bc.circuit.size() == 2


def test_benchmark_result_dataclass() -> None:
    """Test BenchmarkResult dataclass and properties."""
    original = qf.CircuitMetrics(
        depth=2, size=4, width=2, two_qubit_count=2, two_qubit_depth=2, gate_counts={}
    )
    transpiled = qf.CircuitMetrics(
        depth=4, size=8, width=2, two_qubit_count=4, two_qubit_depth=4, gate_counts={}
    )

    result = qf.BenchmarkResult(
        circuit_name="test",
        category="test_cat",
        backend="qiskit",
        original=original,
        transpiled=transpiled,
        time_ms=1.5,
    )

    assert result.depth_ratio == 2.0
    assert result.size_ratio == 2.0
    assert result.two_qubit_ratio == 2.0
    assert result.time_ms == 1.5


def test_benchmark_result_to_dict() -> None:
    """Test BenchmarkResult.to_dict() method."""
    original = qf.CircuitMetrics(
        depth=2, size=4, width=2, two_qubit_count=2, two_qubit_depth=2, gate_counts={}
    )
    transpiled = qf.CircuitMetrics(
        depth=4, size=8, width=2, two_qubit_count=4, two_qubit_depth=4, gate_counts={}
    )

    result = qf.BenchmarkResult(
        circuit_name="test",
        category="cat",
        backend="cirq",
        original=original,
        transpiled=transpiled,
        time_ms=2.0,
    )

    d = result.to_dict()
    assert d["circuit_name"] == "test"
    assert d["backend"] == "cirq"
    assert d["depth_before"] == 2
    assert d["depth_after"] == 4
    assert d["depth_ratio"] == 2.0


def test_standard_suite_creation() -> None:
    """Test standard_suite creates expected circuits."""
    bench = qf.TranspileBenchmark.standard_suite(n_qubits=4)

    assert len(bench.circuits) == 6

    names = [c.name for c in bench.circuits]
    assert "bell" in names
    assert "ghz_4" in names
    assert "qft_4" in names
    assert "vqe_4" in names
    assert "qaoa_4" in names
    assert "mixed_4" in names


def test_standard_suite_different_sizes() -> None:
    """Test standard_suite with different qubit counts."""
    bench3 = qf.TranspileBenchmark.standard_suite(n_qubits=3)
    bench5 = qf.TranspileBenchmark.standard_suite(n_qubits=5)

    # All circuits should have the expected naming
    names3 = [c.name for c in bench3.circuits]
    names5 = [c.name for c in bench5.circuits]

    assert "ghz_3" in names3
    assert "ghz_5" in names5


def test_transpile_benchmark_run() -> None:
    """Test TranspileBenchmark.run() method."""
    bench = qf.TranspileBenchmark.standard_suite(n_qubits=3)
    results = bench.run(backends=["qiskit", "cirq"])

    # Should have results for all circuits x backends
    assert len(results) == 6 * 2  # 6 circuits x 2 backends

    # Check that results have expected structure
    for r in results:
        assert r.circuit_name is not None
        assert r.backend in ["qiskit", "cirq"]
        assert r.original.size > 0 or r.circuit_name == "bell"
        assert r.time_ms >= 0


def test_transpile_benchmark_summary() -> None:
    """Test TranspileBenchmark.summary() method."""
    bench = qf.TranspileBenchmark.standard_suite(n_qubits=3)
    bench.run(backends=["qiskit"])

    summary = bench.summary()

    assert "Benchmark Summary" in summary
    assert "qiskit" in summary
    assert "Avg Depth Ratio" in summary


def test_transpile_benchmark_summary_before_run() -> None:
    """Test summary() before run() returns appropriate message."""
    bench = qf.TranspileBenchmark.standard_suite(n_qubits=3)
    summary = bench.summary()

    assert "No results" in summary


def test_transpile_benchmark_summary_by_category() -> None:
    """Test summary_by_category() method."""
    bench = qf.TranspileBenchmark.standard_suite(n_qubits=3)
    bench.run(backends=["qiskit"])

    summary = bench.summary_by_category()

    assert "Summary by Category" in summary
    assert "ENTANGLEMENT" in summary or "entanglement" in summary.lower()


def test_custom_benchmark_circuits() -> None:
    """Test TranspileBenchmark with custom circuits."""
    custom_circuits = [
        qf.BenchmarkCircuit(
            name="custom1",
            circuit=qf.Circuit([qf.H(0), qf.H(1), qf.CZ(0, 1)]),
            category="custom",
        ),
        qf.BenchmarkCircuit(
            name="custom2",
            circuit=qf.Circuit([qf.X(0), qf.CNot(0, 1), qf.Y(1)]),
            category="custom",
        ),
    ]

    bench = qf.TranspileBenchmark(custom_circuits)
    results = bench.run(backends=["braket"])

    assert len(results) == 2
    assert results[0].circuit_name == "custom1"
    assert results[1].circuit_name == "custom2"


def test_benchmark_result_edge_cases() -> None:
    """Test BenchmarkResult with edge case values."""
    # Zero original depth
    original = qf.CircuitMetrics(
        depth=0, size=0, width=0, two_qubit_count=0, two_qubit_depth=0, gate_counts={}
    )
    transpiled = qf.CircuitMetrics(
        depth=0, size=0, width=0, two_qubit_count=0, two_qubit_depth=0, gate_counts={}
    )

    result = qf.BenchmarkResult(
        circuit_name="empty",
        category="edge",
        backend="test",
        original=original,
        transpiled=transpiled,
        time_ms=0.1,
    )

    assert result.depth_ratio == 0.0
    assert result.size_ratio == 0.0
    assert result.two_qubit_ratio == 0.0


def test_benchmark_all_backends() -> None:
    """Test benchmark with all supported backends."""
    bench = qf.TranspileBenchmark.standard_suite(n_qubits=3)
    results = bench.run()  # No backends specified = all

    # Should have results for all backends
    backends_found = set(r.backend for r in results)
    assert "braket" in backends_found
    assert "cirq" in backends_found
    assert "qiskit" in backends_found
    assert "pyquil" in backends_found
    assert "qsim" in backends_found


# fin
