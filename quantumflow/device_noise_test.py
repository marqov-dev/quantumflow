# Copyright 2019-, Gavin E. Crooks and contributors
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Unit tests for quantumflow.device_noise"""

import pytest

import quantumflow as qf


def test_device_noise_model_creation() -> None:
    """Test DeviceNoiseModel dataclass creation."""
    model = qf.DeviceNoiseModel(name="test", num_qubits=10)
    assert model.name == "test"
    assert model.num_qubits == 10
    # Check defaults
    assert model.t1 == 1e-3
    assert model.t2 == 500e-6
    assert model.single_qubit_gate_error == 0.001
    assert model.two_qubit_gate_error == 0.01


def test_device_noise_model_custom_params() -> None:
    """Test DeviceNoiseModel with custom parameters."""
    model = qf.DeviceNoiseModel(
        name="custom",
        num_qubits=5,
        t1=100e-6,
        t2=50e-6,
        single_qubit_gate_time=30e-9,
        two_qubit_gate_time=80e-9,
        single_qubit_gate_error=0.002,
        two_qubit_gate_error=0.015,
        readout_error=(0.03, 0.04),
    )
    assert model.t1 == 100e-6
    assert model.two_qubit_gate_error == 0.015
    assert model.readout_error == (0.03, 0.04)


def test_gate_fidelity_single_qubit() -> None:
    """Test gate_fidelity for single-qubit gates."""
    model = qf.IONQ_ARIA
    fid = model.gate_fidelity(qf.H(0))
    # IonQ has very high fidelity
    assert 0.99 < fid < 1.0


def test_gate_fidelity_two_qubit() -> None:
    """Test gate_fidelity for two-qubit gates."""
    model = qf.IONQ_ARIA
    fid = model.gate_fidelity(qf.CNot(0, 1))
    # 2Q gates have lower fidelity than 1Q gates
    assert 0.99 < fid < 1.0


def test_gate_fidelity_lower_for_two_qubit() -> None:
    """Test that 2Q gate fidelity is lower than 1Q gate fidelity."""
    model = qf.IQM_GARNET
    fid_1q = model.gate_fidelity(qf.H(0))
    fid_2q = model.gate_fidelity(qf.CNot(0, 1))
    # 2Q gates should have lower fidelity due to higher error rate
    assert fid_2q < fid_1q


def test_circuit_fidelity_bell_state() -> None:
    """Test circuit_fidelity for Bell state circuit."""
    model = qf.IONQ_ARIA
    circ = qf.Circuit([qf.H(0), qf.CNot(0, 1)])
    fid = model.circuit_fidelity(circ)
    # 2 gates, very high fidelity expected
    assert 0.99 < fid < 1.0


def test_circuit_fidelity_ghz() -> None:
    """Test circuit_fidelity for GHZ circuit."""
    model = qf.IQM_GARNET
    circ = qf.ghz_circuit(range(4))
    fid = model.circuit_fidelity(circ)
    # 4 gates (1 H + 3 CNots), should still be reasonable
    assert 0.9 < fid < 1.0


def test_circuit_fidelity_empty() -> None:
    """Test circuit_fidelity for empty circuit."""
    model = qf.IONQ_ARIA
    circ = qf.Circuit()
    fid = model.circuit_fidelity(circ)
    # Empty circuit should have perfect fidelity
    assert fid == 1.0


def test_circuit_fidelity_deep_circuit() -> None:
    """Test that fidelity decreases with circuit depth."""
    model = qf.IQM_GARNET
    shallow = qf.ghz_circuit(range(3))  # 3 gates
    deep = qf.ghz_circuit(range(10))  # 10 gates

    fid_shallow = model.circuit_fidelity(shallow)
    fid_deep = model.circuit_fidelity(deep)

    # Deeper circuits should have lower fidelity
    assert fid_deep < fid_shallow


def test_ionq_better_than_superconducting_deep() -> None:
    """Test that IonQ has higher fidelity for deep circuits."""
    circ = qf.ghz_circuit(range(10))  # Deep circuit

    ionq_fid = qf.IONQ_ARIA.circuit_fidelity(circ)
    iqm_fid = qf.IQM_GARNET.circuit_fidelity(circ)

    # IonQ should have higher fidelity for deep circuits
    # due to longer coherence times
    assert ionq_fid > iqm_fid


def test_estimate_circuit_fidelity_by_name() -> None:
    """Test estimate_circuit_fidelity with device name."""
    circ = qf.Circuit([qf.H(0), qf.CNot(0, 1)])
    fid = qf.estimate_circuit_fidelity(circ, "ionq_aria")
    assert 0.99 < fid < 1.0


def test_estimate_circuit_fidelity_by_model() -> None:
    """Test estimate_circuit_fidelity with DeviceNoiseModel instance."""
    circ = qf.Circuit([qf.H(0), qf.CNot(0, 1)])
    fid = qf.estimate_circuit_fidelity(circ, qf.IQM_GARNET)
    assert 0.9 < fid < 1.0


def test_estimate_circuit_fidelity_unknown_device() -> None:
    """Test that unknown device name raises ValueError."""
    circ = qf.Circuit([qf.H(0)])
    with pytest.raises(ValueError, match="Unknown device"):
        qf.estimate_circuit_fidelity(circ, "unknown_device")


def test_pre_configured_ionq_aria() -> None:
    """Test IONQ_ARIA pre-configured model."""
    assert qf.IONQ_ARIA.name == "ionq_aria"
    assert qf.IONQ_ARIA.num_qubits == 25
    assert qf.IONQ_ARIA.t1 == 10.0  # Long coherence time
    assert qf.IONQ_ARIA.single_qubit_gate_error == 0.0006  # 0.06% (AWS Braket)


def test_pre_configured_iqm_garnet() -> None:
    """Test IQM_GARNET pre-configured model."""
    assert qf.IQM_GARNET.name == "iqm_garnet"
    assert qf.IQM_GARNET.num_qubits == 20
    assert qf.IQM_GARNET.t1 == 40e-6  # ~40µs (IQM whitepaper)
    assert qf.IQM_GARNET.two_qubit_gate_time == 30e-9  # Fast CZ gates


def test_pre_configured_rigetti_ankaa() -> None:
    """Test RIGETTI_ANKAA pre-configured model."""
    assert qf.RIGETTI_ANKAA.name == "rigetti_ankaa"
    assert qf.RIGETTI_ANKAA.num_qubits == 84  # Larger device
    assert qf.RIGETTI_ANKAA.two_qubit_gate_error == 0.01  # 1.0% iSWAP (Ankaa-3)


def test_device_models_dict() -> None:
    """Test DEVICE_MODELS dictionary."""
    assert "ionq_aria" in qf.DEVICE_MODELS
    assert "iqm_garnet" in qf.DEVICE_MODELS
    assert "rigetti_ankaa" in qf.DEVICE_MODELS
    assert qf.DEVICE_MODELS["ionq_aria"] is qf.IONQ_ARIA


def test_fidelity_product_of_gates() -> None:
    """Test that circuit fidelity equals product of gate fidelities."""
    model = qf.DeviceNoiseModel(
        name="test",
        t1=1.0,  # Long coherence to minimize T1/T2 effects
        t2=1.0,
        single_qubit_gate_error=0.01,
        two_qubit_gate_error=0.02,
    )

    circ = qf.Circuit([qf.H(0), qf.H(1)])

    # Calculate expected fidelity
    fid_h = model.gate_fidelity(qf.H(0))
    expected_fid = fid_h * fid_h

    actual_fid = model.circuit_fidelity(circ)
    assert abs(actual_fid - expected_fid) < 1e-10


def test_coherence_affects_fidelity() -> None:
    """Test that shorter coherence times reduce fidelity."""
    long_coherence = qf.DeviceNoiseModel(
        name="long",
        t1=1.0,
        t2=0.5,
        single_qubit_gate_error=0.001,
        two_qubit_gate_error=0.01,
    )
    short_coherence = qf.DeviceNoiseModel(
        name="short",
        t1=10e-6,
        t2=5e-6,
        single_qubit_gate_error=0.001,
        two_qubit_gate_error=0.01,
    )

    circ = qf.Circuit([qf.H(0), qf.CNot(0, 1)])

    fid_long = long_coherence.circuit_fidelity(circ)
    fid_short = short_coherence.circuit_fidelity(circ)

    # Shorter coherence should give lower fidelity
    assert fid_short < fid_long


# fin
