# Copyright 2019-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
=====================
Benchmarking Framework
=====================

Framework for benchmarking transpilation quality across backends and circuit types.

.. contents:: :local:
.. currentmodule:: quantumflow

.. autoclass:: BenchmarkCircuit
.. autoclass:: TranspileBenchmark
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .circuits import Circuit, ghz_circuit
from .gates import QFTGate
from .metrics import SUPPORTED_BACKENDS, CircuitMetrics, analyze_circuit, compare_backends
from .stdgates import CNot, CZ, H, Rx, Ry, Rz, Rzz, X
from .translate import circuit_translate

__all__ = [
    "BenchmarkCircuit",
    "BenchmarkResult",
    "TranspileBenchmark",
]


@dataclass
class BenchmarkCircuit:
    """A circuit for benchmarking with metadata.

    Attributes:
        name: Descriptive name for the circuit.
        circuit: The quantum circuit.
        category: Category for grouping (e.g., "entanglement", "transform").
    """

    name: str
    circuit: Circuit
    category: str


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single circuit on a single backend.

    Attributes:
        circuit_name: Name of the benchmark circuit.
        category: Category of the circuit.
        backend: Target backend name.
        original: Metrics before transpilation.
        transpiled: Metrics after transpilation.
        time_ms: Time taken for transpilation in milliseconds.
    """

    circuit_name: str
    category: str
    backend: str
    original: CircuitMetrics
    transpiled: CircuitMetrics
    time_ms: float

    @property
    def depth_ratio(self) -> float:
        """Ratio of transpiled depth to original depth."""
        if self.original.depth == 0:
            return 0.0
        return self.transpiled.depth / self.original.depth

    @property
    def size_ratio(self) -> float:
        """Ratio of transpiled size to original size."""
        if self.original.size == 0:
            return 0.0
        return self.transpiled.size / self.original.size

    @property
    def two_qubit_ratio(self) -> float:
        """Ratio of transpiled 2Q gates to original 2Q gates."""
        if self.original.two_qubit_count == 0:
            return 0.0 if self.transpiled.two_qubit_count == 0 else float("inf")
        return self.transpiled.two_qubit_count / self.original.two_qubit_count

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "circuit_name": self.circuit_name,
            "category": self.category,
            "backend": self.backend,
            "depth_before": self.original.depth,
            "depth_after": self.transpiled.depth,
            "depth_ratio": self.depth_ratio,
            "size_before": self.original.size,
            "size_after": self.transpiled.size,
            "size_ratio": self.size_ratio,
            "two_qubit_before": self.original.two_qubit_count,
            "two_qubit_after": self.transpiled.two_qubit_count,
            "two_qubit_ratio": self.two_qubit_ratio,
            "time_ms": self.time_ms,
        }


class TranspileBenchmark:
    """Framework for benchmarking transpilation across backends.

    Examples:
        >>> import quantumflow as qf
        >>> bench = qf.TranspileBenchmark.standard_suite(n_qubits=4)
        >>> results = bench.run(backends=["qiskit", "cirq"])
        >>> print(bench.summary())
    """

    def __init__(self, circuits: List[BenchmarkCircuit]):
        """Initialize with a list of benchmark circuits.

        Args:
            circuits: List of BenchmarkCircuit instances to benchmark.
        """
        self.circuits = circuits
        self.results: List[BenchmarkResult] = []

    def run(
        self,
        backends: Optional[List[str]] = None,
    ) -> List[BenchmarkResult]:
        """Run benchmarks and return results.

        Args:
            backends: List of backend names to test. Defaults to all supported.

        Returns:
            List of BenchmarkResult instances.
        """
        if backends is None:
            backends = list(SUPPORTED_BACKENDS.keys())

        self.results = []

        for bench_circ in self.circuits:
            original_metrics = analyze_circuit(bench_circ.circuit)

            for backend_name in backends:
                gateset = SUPPORTED_BACKENDS[backend_name]

                start_time = time.perf_counter()
                transpiled_circ = circuit_translate(bench_circ.circuit, targets=gateset)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                transpiled_metrics = analyze_circuit(transpiled_circ)

                result = BenchmarkResult(
                    circuit_name=bench_circ.name,
                    category=bench_circ.category,
                    backend=backend_name,
                    original=original_metrics,
                    transpiled=transpiled_metrics,
                    time_ms=elapsed_ms,
                )
                self.results.append(result)

        return self.results

    def to_dataframe(self):
        """Convert results to pandas DataFrame.

        Returns:
            pandas DataFrame with benchmark results.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame output")

        rows = [r.to_dict() for r in self.results]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Return formatted summary of benchmark results.

        Returns:
            Multi-line string with summary table.
        """
        if not self.results:
            return "No results. Run benchmark first."

        lines = [
            "Benchmark Summary",
            "=" * 80,
            f"{'Backend':<12} | {'Avg Depth Ratio':>15} | {'Avg Size Ratio':>14} | "
            f"{'Avg 2Q Ratio':>12} | {'Avg Time (ms)':>13}",
            "-" * 80,
        ]

        # Group by backend
        backend_stats: Dict[str, Dict[str, List[float]]] = {}
        for r in self.results:
            if r.backend not in backend_stats:
                backend_stats[r.backend] = {
                    "depth_ratio": [],
                    "size_ratio": [],
                    "two_qubit_ratio": [],
                    "time_ms": [],
                }
            backend_stats[r.backend]["depth_ratio"].append(r.depth_ratio)
            backend_stats[r.backend]["size_ratio"].append(r.size_ratio)
            # Skip inf values for 2Q ratio average
            if r.two_qubit_ratio != float("inf"):
                backend_stats[r.backend]["two_qubit_ratio"].append(r.two_qubit_ratio)
            backend_stats[r.backend]["time_ms"].append(r.time_ms)

        for backend in sorted(backend_stats.keys()):
            stats = backend_stats[backend]
            avg_depth = sum(stats["depth_ratio"]) / len(stats["depth_ratio"])
            avg_size = sum(stats["size_ratio"]) / len(stats["size_ratio"])
            avg_2q = (
                sum(stats["two_qubit_ratio"]) / len(stats["two_qubit_ratio"])
                if stats["two_qubit_ratio"]
                else 0.0
            )
            avg_time = sum(stats["time_ms"]) / len(stats["time_ms"])

            lines.append(
                f"{backend:<12} | {avg_depth:>15.2f}x | {avg_size:>14.2f}x | "
                f"{avg_2q:>12.2f}x | {avg_time:>13.3f}"
            )

        lines.append("-" * 80)
        lines.append(f"Total circuits: {len(self.circuits)}")
        lines.append(f"Total benchmarks: {len(self.results)}")

        return "\n".join(lines)

    def summary_by_category(self) -> str:
        """Return summary grouped by circuit category.

        Returns:
            Multi-line string with per-category summary.
        """
        if not self.results:
            return "No results. Run benchmark first."

        lines = [
            "Summary by Category",
            "=" * 80,
        ]

        # Group by category, then backend
        categories: Dict[str, Dict[str, List[BenchmarkResult]]] = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {}
            if r.backend not in categories[r.category]:
                categories[r.category][r.backend] = []
            categories[r.category][r.backend].append(r)

        for category in sorted(categories.keys()):
            lines.append(f"\n{category.upper()}")
            lines.append("-" * 40)
            for backend in sorted(categories[category].keys()):
                results = categories[category][backend]
                avg_size = sum(r.size_ratio for r in results) / len(results)
                avg_2q = sum(
                    r.two_qubit_ratio
                    for r in results
                    if r.two_qubit_ratio != float("inf")
                )
                count = sum(
                    1 for r in results if r.two_qubit_ratio != float("inf")
                )
                avg_2q = avg_2q / count if count > 0 else 0.0
                lines.append(f"  {backend:<10}: size {avg_size:.2f}x, 2Q {avg_2q:.2f}x")

        return "\n".join(lines)

    @classmethod
    def standard_suite(cls, n_qubits: int = 4) -> "TranspileBenchmark":
        """Create benchmark suite with standard circuits.

        Args:
            n_qubits: Number of qubits for scalable circuits.

        Returns:
            TranspileBenchmark with standard circuit suite.

        The standard suite includes:
            - Bell state (2 qubits)
            - GHZ state (n qubits)
            - QFT circuit (n qubits)
            - Hardware-efficient VQE ansatz (n qubits)
            - QAOA-style circuit (n qubits)
            - Random-ish circuit (n qubits)
        """
        qubits = list(range(n_qubits))
        circuits = []

        # 1. Bell state (baseline, 2 qubits)
        bell = Circuit([H(0), CNot(0, 1)])
        circuits.append(BenchmarkCircuit("bell", bell, "entanglement"))

        # 2. GHZ state (n qubits)
        ghz = ghz_circuit(qubits)
        circuits.append(BenchmarkCircuit(f"ghz_{n_qubits}", ghz, "entanglement"))

        # 3. QFT circuit (n qubits)
        qft_gate = QFTGate(qubits)
        qft_circ = Circuit(qft_gate.decompose())
        circuits.append(BenchmarkCircuit(f"qft_{n_qubits}", qft_circ, "transform"))

        # 4. Hardware-efficient VQE ansatz (Ry + CNot layers)
        vqe = Circuit()
        for q in qubits:
            vqe += Ry(0.5, q)
        for i in range(n_qubits - 1):
            vqe += CNot(i, i + 1)
        for q in qubits:
            vqe += Ry(0.3, q)
        circuits.append(BenchmarkCircuit(f"vqe_{n_qubits}", vqe, "variational"))

        # 5. QAOA-style circuit (Rzz + Rx layers)
        qaoa = Circuit()
        # Initial layer
        for q in qubits:
            qaoa += H(q)
        # Cost layer (Rzz on edges)
        for i in range(n_qubits - 1):
            qaoa += Rzz(0.5, i, i + 1)
        # Mixer layer
        for q in qubits:
            qaoa += Rx(0.3, q)
        circuits.append(BenchmarkCircuit(f"qaoa_{n_qubits}", qaoa, "variational"))

        # 6. Mixed circuit with various gates
        mixed = Circuit()
        for q in qubits:
            mixed += H(q)
        for i in range(n_qubits - 1):
            mixed += CZ(i, i + 1)
        for q in qubits:
            mixed += Rz(0.25, q)
        for i in range(0, n_qubits - 1, 2):
            mixed += CNot(i, i + 1)
        circuits.append(BenchmarkCircuit(f"mixed_{n_qubits}", mixed, "mixed"))

        return cls(circuits)
