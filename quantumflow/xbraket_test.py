# Copyright 2019-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Unit tests for quantumflow.xbraket
"""

import numpy as np
import pytest

import quantumflow as qf
from quantumflow.xbraket import (
    BraketSimulator,
    braket_to_circuit,
    circuit_to_braket,
)

pytest.importorskip("braket")


def test_braket_to_circuit() -> None:
    from braket.circuits import Circuit as bkCircuit

    bkcirc = bkCircuit().h(0).cnot(0, 1)

    circ = braket_to_circuit(bkcirc)
    print(circ)

    bkcirc = bkcirc.rx(1, 0.2)
    bkcirc = bkcirc.xx(0, 1, np.pi * 0.5)
    bkcirc = bkcirc.xy(0, 2, np.pi * 0.5)
    circ = braket_to_circuit(bkcirc)
    print(circ)


def test_circuit_to_qiskit() -> None:
    circ = qf.Circuit([qf.CNot(0, 1), qf.Rz(0.2, 1)])
    bkcirc = circuit_to_braket(circ)
    print(bkcirc)


def test_braket_unitary_roundtrip_2qubit() -> None:
    """Test round-trip conversion of 2-qubit UnitaryGate through Braket."""
    # Use RandomGate which is a UnitaryGate subclass
    gate0 = qf.RandomGate([0, 1])
    circ0 = qf.Circuit([gate0])

    # Convert QF → Braket → QF
    bkcirc = circuit_to_braket(circ0)
    circ1 = braket_to_circuit(bkcirc)

    assert qf.circuits_close(circ0, circ1)


def test_braket_unitary_roundtrip_1qubit() -> None:
    """Test round-trip conversion of 1-qubit UnitaryGate through Braket."""
    gate0 = qf.RandomGate([0])
    circ0 = qf.Circuit([gate0])

    # Convert QF → Braket → QF
    bkcirc = circuit_to_braket(circ0)
    circ1 = braket_to_circuit(bkcirc)

    assert qf.circuits_close(circ0, circ1)


def test_braket_unitary_single_qubit() -> None:
    """Test Braket Unitary gate conversion for single qubit."""
    from braket.circuits import Circuit as bkCircuit

    # Create Braket circuit with unitary (X gate as matrix)
    bkcirc = bkCircuit()
    U = np.array([[0, 1], [1, 0]])
    bkcirc = bkcirc.unitary(matrix=U, targets=[0])

    # Convert to QuantumFlow
    circ = braket_to_circuit(bkcirc)

    # Verify it's equivalent to X gate
    assert qf.gates_close(circ[0], qf.X(0))


def test_braket_unitary_2qubit_cnot() -> None:
    """Test Braket 2-qubit Unitary gate conversion (CNOT matrix)."""
    from braket.circuits import Circuit as bkCircuit

    # CNOT matrix
    cnot_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])
    bkcirc = bkCircuit()
    bkcirc = bkcirc.unitary(matrix=cnot_matrix, targets=[0, 1])

    # Convert to QuantumFlow
    circ = braket_to_circuit(bkcirc)

    # Verify it's equivalent to CNOT gate
    assert qf.gates_close(circ[0], qf.CNot(0, 1))


def test_braket_unitary_mixed_circuit() -> None:
    """Test circuit with both standard gates and UnitaryGate."""
    from braket.circuits import Circuit as bkCircuit

    # Create QF circuit with mixed gates
    circ0 = qf.Circuit()
    circ0 += qf.H(0)
    circ0 += qf.RandomGate([1])  # UnitaryGate
    circ0 += qf.CNot(0, 1)

    # Convert QF → Braket → QF
    bkcirc = circuit_to_braket(circ0)
    circ1 = braket_to_circuit(bkcirc)

    assert qf.circuits_close(circ0, circ1)


def test_braket_unitary_complex_phase() -> None:
    """Test Unitary with complex phases (S gate = phase gate)."""
    from braket.circuits import Circuit as bkCircuit

    # S gate matrix (has complex entries)
    s_matrix = np.array([[1, 0], [0, 1j]])
    bkcirc = bkCircuit()
    bkcirc = bkcirc.unitary(matrix=s_matrix, targets=[0])

    # Convert to QuantumFlow
    circ = braket_to_circuit(bkcirc)

    # Verify it's equivalent to S gate
    assert qf.gates_close(circ[0], qf.S(0))


def test_braketsimulator() -> None:
    circ = qf.Circuit()
    circ += qf.Rx(0.4, 0)
    circ += qf.X(0)
    circ += qf.H(1)
    circ += qf.Y(2)
    circ += qf.Rx(0.3, 0)
    circ += qf.XX(0.2, 0, 1)
    circ += qf.XY(0.3, 0, 1)
    circ += qf.ZZ(0.4, 0, 1)

    circ += qf.Can(0.1, 0.2, 0.2, 0, 1)
    circ += qf.V(0)
    circ += qf.CV(2, 3)
    circ += qf.CPhase01(2, 3)

    sim = BraketSimulator(circ)
    assert qf.states_close(circ.run(), sim.run())


# fin
