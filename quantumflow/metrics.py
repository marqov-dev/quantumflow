# Copyright 2019-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
==============
Circuit Metrics
==============

Metrics and analysis functions for quantum circuits.

.. contents:: :local:
.. currentmodule:: quantumflow

.. autoclass:: CircuitMetrics
.. autoclass:: BackendReport
.. autoclass:: TranspilationReport
.. autofunction:: analyze_circuit
.. autofunction:: compare_backends
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Type

from .circuits import Circuit, count_operations
from .dagcircuit import DAGCircuit
from .gatesets import (
    BRAKET_GATES,
    CIRQ_GATES,
    QISKIT_GATES,
    QSIM_GATES,
    QUIL_GATES,
)
from .ops import Gate, Operation
from .translate import circuit_translate

__all__ = [
    "CircuitMetrics",
    "analyze_circuit",
    "BackendReport",
    "TranspilationReport",
    "compare_backends",
    "SUPPORTED_BACKENDS",
]

SUPPORTED_BACKENDS: Dict[str, Set[Type[Gate]]] = {
    "braket": BRAKET_GATES,
    "cirq": CIRQ_GATES,
    "qiskit": QISKIT_GATES,
    "pyquil": QUIL_GATES,
    "qsim": QSIM_GATES,
}


@dataclass
class CircuitMetrics:
    """Comprehensive circuit analysis metrics.

    Attributes:
        depth: Total circuit depth (including single-qubit gates).
        size: Total number of operations in the circuit.
        width: Number of qubits in the circuit.
        two_qubit_count: Number of two-qubit gates.
        two_qubit_depth: Depth considering only multi-qubit gates.
        gate_counts: Dictionary mapping gate types to their counts.
    """

    depth: int
    size: int
    width: int
    two_qubit_count: int
    two_qubit_depth: int
    gate_counts: Dict[Type[Operation], int]

    @property
    def two_qubit_ratio(self) -> float:
        """Ratio of 2-qubit gates to total gates."""
        return self.two_qubit_count / self.size if self.size > 0 else 0.0


def analyze_circuit(circ: Circuit) -> CircuitMetrics:
    """Analyze a quantum circuit and return comprehensive metrics.

    This function efficiently computes various metrics about a circuit,
    performing DAG conversion only once for depth calculations.

    Args:
        circ: The quantum circuit to analyze.

    Returns:
        CircuitMetrics containing depth, size, width, and gate statistics.

    Examples:
        >>> import quantumflow as qf
        >>> circ = qf.ghz_circuit(range(4))
        >>> metrics = qf.analyze_circuit(circ)
        >>> metrics.depth
        4
        >>> metrics.two_qubit_count
        3
    """
    # Handle empty circuit edge case
    if len(circ) == 0:
        return CircuitMetrics(
            depth=0,
            size=0,
            width=circ.qubit_nb,
            two_qubit_count=0,
            two_qubit_depth=0,
            gate_counts={},
        )

    # Convert to DAG once for all depth calculations
    dag = DAGCircuit(circ)

    # Get gate counts using existing function
    gate_counts = count_operations(circ)

    # Count 2-qubit gates by checking qubit_nb on each operation
    two_qubit_count = sum(1 for op in circ if op.qubit_nb == 2)

    # Get two-qubit depth; DAGCircuit.depth(local=False) returns -1 when
    # there are no multi-qubit gates, so we clamp to 0
    two_qubit_depth = dag.depth(local=False)
    if two_qubit_depth < 0:
        two_qubit_depth = 0

    return CircuitMetrics(
        depth=dag.depth(local=True),
        size=circ.size(),
        width=circ.qubit_nb,
        two_qubit_count=two_qubit_count,
        two_qubit_depth=two_qubit_depth,
        gate_counts=gate_counts,
    )


@dataclass
class BackendReport:
    """Metrics for a single backend transpilation.

    Attributes:
        backend: Name of the backend (e.g., "qiskit", "cirq").
        original: Metrics of the original circuit before transpilation.
        transpiled: Metrics of the circuit after transpilation to this backend.
    """

    backend: str
    original: CircuitMetrics
    transpiled: CircuitMetrics

    @property
    def depth_overhead(self) -> float:
        """Ratio of transpiled depth to original depth."""
        if self.original.depth == 0:
            return 0.0
        return self.transpiled.depth / self.original.depth

    @property
    def size_overhead(self) -> float:
        """Ratio of transpiled size to original size."""
        if self.original.size == 0:
            return 0.0
        return self.transpiled.size / self.original.size

    @property
    def two_qubit_overhead(self) -> float:
        """Ratio of transpiled 2-qubit gates to original 2-qubit gates."""
        if self.original.two_qubit_count == 0:
            return 0.0 if self.transpiled.two_qubit_count == 0 else float("inf")
        return self.transpiled.two_qubit_count / self.original.two_qubit_count


@dataclass
class TranspilationReport:
    """Cross-backend transpilation comparison report.

    Attributes:
        original: Metrics of the original circuit.
        backends: Dictionary mapping backend names to their transpilation reports.
    """

    original: CircuitMetrics
    backends: Dict[str, BackendReport]

    def best_for_depth(self) -> str:
        """Return the backend name with the lowest transpiled depth."""
        return min(self.backends, key=lambda b: self.backends[b].transpiled.depth)

    def best_for_two_qubit(self) -> str:
        """Return the backend name with the fewest 2-qubit gates after transpilation."""
        return min(
            self.backends, key=lambda b: self.backends[b].transpiled.two_qubit_count
        )

    def summary(self) -> str:
        """Return a formatted comparison table.

        Returns:
            A multi-line string with a table comparing backends.
        """
        lines = ["Backend     | Depth | Size | 2Q Gates | Depth OH | Size OH"]
        lines.append("-" * 60)
        for name, report in sorted(self.backends.items()):
            t = report.transpiled
            lines.append(
                f"{name:11} | {t.depth:5} | {t.size:4} | {t.two_qubit_count:8} | "
                f"{report.depth_overhead:7.2f}x | {report.size_overhead:.2f}x"
            )
        return "\n".join(lines)


def compare_backends(
    circ: Circuit, backends: Optional[List[str]] = None
) -> TranspilationReport:
    """Compare circuit transpilation across multiple backends.

    This function translates a circuit to the gate sets supported by
    different quantum computing backends and compares the resulting
    circuit metrics.

    Args:
        circ: The quantum circuit to analyze.
        backends: List of backend names to compare. If None, compares all
                  supported backends: braket, cirq, qiskit, pyquil, qsim.

    Returns:
        TranspilationReport with per-backend metrics and comparison methods.

    Raises:
        ValueError: If an unknown backend name is provided.

    Examples:
        >>> import quantumflow as qf
        >>> circ = qf.Circuit([qf.Can(0.1, 0.2, 0.3, 0, 1)])
        >>> report = qf.compare_backends(circ)
        >>> print(report.summary())
        >>> report.best_for_depth()
        'cirq'
    """
    if backends is None:
        backends = list(SUPPORTED_BACKENDS.keys())

    original_metrics = analyze_circuit(circ)
    backend_reports: Dict[str, BackendReport] = {}

    for backend_name in backends:
        if backend_name not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unknown backend: {backend_name}")

        gateset = SUPPORTED_BACKENDS[backend_name]
        transpiled_circ = circuit_translate(circ, targets=gateset)
        transpiled_metrics = analyze_circuit(transpiled_circ)

        backend_reports[backend_name] = BackendReport(
            backend=backend_name,
            original=original_metrics,
            transpiled=transpiled_metrics,
        )

    return TranspilationReport(
        original=original_metrics,
        backends=backend_reports,
    )
