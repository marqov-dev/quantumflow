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
.. autofunction:: analyze_circuit
"""

from dataclasses import dataclass
from typing import Dict, Type

from .circuits import Circuit, count_operations
from .dagcircuit import DAGCircuit
from .ops import Operation

__all__ = ["CircuitMetrics", "analyze_circuit"]


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
