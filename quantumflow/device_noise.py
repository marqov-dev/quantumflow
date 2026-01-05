# Copyright 2019-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
==================
Device Noise Models
==================

Device-specific noise parameters for realistic fidelity estimation.

.. contents:: :local:
.. currentmodule:: quantumflow

.. autoclass:: DeviceNoiseModel
.. autofunction:: estimate_circuit_fidelity

Pre-configured device models:

.. data:: IONQ_ARIA
.. data:: IQM_GARNET
.. data:: RIGETTI_ANKAA
.. data:: DEVICE_MODELS
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np

from .circuits import Circuit
from .ops import Gate

__all__ = [
    "DeviceNoiseModel",
    "IONQ_ARIA",
    "IQM_GARNET",
    "RIGETTI_ANKAA",
    "DEVICE_MODELS",
    "estimate_circuit_fidelity",
]


@dataclass
class DeviceNoiseModel:
    """Device-specific noise parameters for realistic simulation.

    This dataclass captures the key noise characteristics of quantum hardware,
    enabling fidelity estimation without full density matrix simulation.

    Attributes:
        name: Device identifier (e.g., "ionq_aria").
        num_qubits: Number of qubits on the device.
        t1: T1 relaxation time per qubit in seconds.
        t2: T2 dephasing time per qubit in seconds.
        single_qubit_gate_time: Duration of 1Q gates in seconds.
        two_qubit_gate_time: Duration of 2Q gates in seconds.
        single_qubit_gate_error: Error probability for 1Q gates.
        two_qubit_gate_error: Error probability for 2Q gates.
        readout_error: Tuple of (p(1|0), p(0|1)) readout errors.

    Examples:
        >>> import quantumflow as qf
        >>> model = qf.DeviceNoiseModel(name="custom", num_qubits=5)
        >>> circ = qf.Circuit([qf.H(0), qf.CNot(0, 1)])
        >>> fid = model.circuit_fidelity(circ)
    """

    name: str
    num_qubits: int = 20
    t1: float = 1e-3  # Default 1ms
    t2: float = 500e-6  # Default 500µs
    single_qubit_gate_time: float = 50e-9  # 50ns
    two_qubit_gate_time: float = 200e-9  # 200ns
    single_qubit_gate_error: float = 0.001  # 0.1%
    two_qubit_gate_error: float = 0.01  # 1%
    readout_error: Tuple[float, float] = (0.01, 0.01)  # 1% each direction

    def gate_fidelity(self, gate: Gate) -> float:
        """Estimate fidelity of a single gate on this device.

        Combines depolarizing error based on gate error rate with
        T1/T2 decoherence based on gate duration.

        Args:
            gate: The gate to estimate fidelity for.

        Returns:
            Estimated gate fidelity (0.0 to 1.0).
        """
        n_qubits = gate.qubit_nb

        if n_qubits == 1:
            gate_error = self.single_qubit_gate_error
            gate_time = self.single_qubit_gate_time
        else:
            gate_error = self.two_qubit_gate_error
            gate_time = self.two_qubit_gate_time

        # Depolarizing contribution
        f_depol = 1.0 - gate_error

        # T1/T2 contribution (per qubit)
        p_t1 = 1.0 - np.exp(-gate_time / self.t1)
        p_t2 = 1.0 - np.exp(-gate_time / self.t2)
        f_coherence = (1.0 - p_t1) * (1.0 - p_t2 / 2)

        # Combined fidelity (product for independent errors)
        return f_depol * (f_coherence**n_qubits)

    def circuit_fidelity(self, circ: Circuit) -> float:
        """Estimate fidelity of entire circuit on this device.

        Assumes independent gate errors, so fidelity is the product
        of individual gate fidelities.

        Args:
            circ: The quantum circuit to analyze.

        Returns:
            Estimated circuit fidelity (0.0 to 1.0).

        Examples:
            >>> import quantumflow as qf
            >>> model = qf.IONQ_ARIA
            >>> circ = qf.ghz_circuit(range(4))
            >>> fid = model.circuit_fidelity(circ)
            >>> fid > 0.9
            True
        """
        fidelity = 1.0
        for op in circ:
            if hasattr(op, "qubit_nb"):  # It's a gate
                fidelity *= self.gate_fidelity(op)
        return fidelity


# Pre-configured device models based on published specifications


IONQ_ARIA = DeviceNoiseModel(
    name="ionq_aria",
    num_qubits=25,
    t1=10.0,  # ~10s for trapped ions (IonQ specs)
    t2=1.0,  # ~1s (IonQ specs)
    single_qubit_gate_time=135e-6,  # 135µs (IonQ specs)
    two_qubit_gate_time=600e-6,  # 600µs (IonQ specs)
    single_qubit_gate_error=0.0006,  # 0.06% (AWS Braket, IonQ practical performance)
    two_qubit_gate_error=0.006,  # 0.6% (IonQ specs)
    readout_error=(0.0039, 0.0039),  # 0.39% SPAM error (AWS Braket)
)
"""IonQ Aria trapped-ion quantum computer.

Sources:
    - https://ionq.com/quantum-systems/aria
    - https://ionq.com/resources/ionq-aria-practical-performance
    - https://aws.amazon.com/blogs/quantum-computing/amazon-braket-launches-ionq-aria-with-built-in-error-mitigation/

Characteristics:
    - Trapped-ion technology with long coherence times (T1 ~10s, T2 ~1s)
    - All-to-all connectivity (no SWAP overhead)
    - High-fidelity gates but slower gate times (135µs 1Q, 600µs 2Q)
    - Best for deep circuits requiring high fidelity
    - Native gates: GPI, GPI2, MS (Molmer-Sorenson)
"""

IQM_GARNET = DeviceNoiseModel(
    name="iqm_garnet",
    num_qubits=20,
    t1=40e-6,  # ~40µs (IQM whitepaper)
    t2=60e-6,  # ~60µs (estimated from T1, simulation papers)
    single_qubit_gate_time=30e-9,  # 30ns (AWS Braket, 20-40ns range)
    two_qubit_gate_time=30e-9,  # 30ns CZ (AWS Braket, 20-40ns range)
    single_qubit_gate_error=0.0008,  # 0.08% median (arXiv:2408.12433)
    two_qubit_gate_error=0.0049,  # 0.49% median (arXiv:2408.12433)
    readout_error=(0.01, 0.01),  # ~1% (estimated)
)
"""IQM Garnet superconducting quantum computer.

Sources:
    - https://meetiqm.com/wp-content/uploads/2025/04/IQM-Garnet-20Q-Whitepaper-2024.pdf
    - https://arxiv.org/abs/2408.12433 (Technology and Performance Benchmarks)
    - https://aws.amazon.com/braket/quantum-computers/iqm/

Characteristics:
    - Superconducting transmon qubits with tunable couplers
    - Very fast gate times (20-40ns) enabled by tunable coupler technology
    - Square lattice topology with 20 computational qubits + 30 couplers
    - Quantum Volume 32 (2^5)
    - Native gates: PRx (phased rotation), CZ
    - Good for variational algorithms with shallow circuits
"""

RIGETTI_ANKAA = DeviceNoiseModel(
    name="rigetti_ankaa",
    num_qubits=84,
    t1=34e-6,  # 34µs median (Ankaa-3, Rigetti press Dec 2024)
    t2=20e-6,  # 20µs median (Ankaa-3, Rigetti press Dec 2024)
    single_qubit_gate_time=40e-9,  # ~40ns (AWS Braket)
    two_qubit_gate_time=72e-9,  # 72ns iSWAP median (Rigetti specs)
    single_qubit_gate_error=0.0009,  # 0.09% (99.91% fidelity, Rigetti specs)
    two_qubit_gate_error=0.01,  # 1.0% iSWAP (99.0% fidelity, Rigetti specs)
    readout_error=(0.053, 0.053),  # 5.3% (94.7% fidelity, arXiv:2410.05202)
)
"""Rigetti Ankaa-3 superconducting quantum computer.

Sources:
    - https://investors.rigetti.com/news-releases/news-release-details/rigetti-computing-launches-84-qubit-ankaatm-3-system-achieves
    - https://arxiv.org/abs/2410.05202 (Real-time QEC on Ankaa-2)
    - https://qcs.rigetti.com/qpus

Characteristics:
    - Superconducting transmon qubits with tunable couplers
    - 84 qubits (largest of the three devices)
    - Native gates: RX, iSWAP, fSim, CZ
    - fSim gates achieve 99.5% fidelity (56ns) for specialized algorithms
    - Good for exploring scaling behavior and NISQ algorithms
    - Note: Readout fidelity lower than gate fidelity (common for superconducting)
"""

DEVICE_MODELS: Dict[str, DeviceNoiseModel] = {
    "ionq_aria": IONQ_ARIA,
    "iqm_garnet": IQM_GARNET,
    "rigetti_ankaa": RIGETTI_ANKAA,
}
"""Dictionary mapping device names to their noise models."""


def estimate_circuit_fidelity(
    circ: Circuit,
    device: Union[str, DeviceNoiseModel],
) -> float:
    """Estimate circuit fidelity on a specific device.

    Convenience function that accepts either a device name or
    DeviceNoiseModel instance.

    Args:
        circ: The quantum circuit to analyze.
        device: Device name (e.g., "ionq_aria") or DeviceNoiseModel instance.

    Returns:
        Estimated fidelity (0.0 to 1.0).

    Raises:
        ValueError: If device name is not recognized.

    Examples:
        >>> import quantumflow as qf
        >>> circ = qf.Circuit([qf.H(0), qf.CNot(0, 1)])
        >>> fid = qf.estimate_circuit_fidelity(circ, "ionq_aria")
        >>> fid > 0.99
        True
    """
    if isinstance(device, str):
        if device not in DEVICE_MODELS:
            raise ValueError(f"Unknown device: {device}")
        device = DEVICE_MODELS[device]

    return device.circuit_fidelity(circ)
