# Realistic Device Noise Models: What We Learned Researching QPU Specifications

*January 4, 2025*

When building a quantum circuit orchestration platform like Marqov, one question keeps coming up: **how do you estimate circuit fidelity before running on real hardware?**

The answer involves device noise models - mathematical representations of how real quantum computers introduce errors. We recently added device noise modeling to QuantumFlow and learned some interesting things along the way.

## The Problem: Guessing vs. Knowing

Our initial implementation used "reasonable estimates" for device parameters:

```python
IONQ_ARIA = DeviceNoiseModel(
    name="ionq_aria",
    t1=10.0,                    # 10 seconds - seemed right for trapped ions
    single_qubit_gate_time=10e-6,  # 10 microseconds - a guess
    two_qubit_gate_error=0.005,    # 0.5% - optimistic
)
```

These values *felt* reasonable based on general knowledge of trapped-ion vs. superconducting technologies. But were they accurate?

We decided to find out.

## The Research Process

We launched parallel research efforts for three devices:
- **IonQ Aria** (trapped-ion)
- **IQM Garnet** (superconducting)
- **Rigetti Ankaa-3** (superconducting)

Our sources included:
- Official vendor documentation and spec sheets
- AWS Braket device property pages
- arXiv papers from vendor research teams
- Academic benchmarking studies
- Open-source implementations in Qiskit, Cirq, and Braket

## What We Found: The Gaps Were Significant

### IonQ Aria

Our biggest surprise was gate times. We estimated 10 microseconds for single-qubit gates; the actual value is **135 microseconds** - over 13x slower than we thought.

| Parameter | Our Estimate | Actual Value | Source |
|-----------|-------------|--------------|--------|
| 1Q gate time | 10 µs | **135 µs** | IonQ specs |
| 2Q gate time | 200 µs | **600 µs** | IonQ specs |
| 1Q error | 0.03% | **0.06%** | AWS Braket |
| 2Q error | 0.5% | **0.6%** | IonQ specs |

Why does this matter? Gate time directly affects decoherence. Longer gates mean more T1/T2 decay, which our fidelity model now correctly captures.

### IQM Garnet

IQM's superconducting qubits are *fast* - we actually overestimated their two-qubit gate time:

| Parameter | Our Estimate | Actual Value | Source |
|-----------|-------------|--------------|--------|
| T1 | 30 µs | **40 µs** | IQM whitepaper |
| 2Q gate time | 60 ns | **30 ns** | AWS Braket |
| 2Q error | 1.0% | **0.49%** | arXiv 2408.12433 |

The 2Q error rate being half what we estimated is a pleasant surprise - IQM's tunable coupler architecture is paying off.

### Rigetti Ankaa-3

Rigetti recently launched Ankaa-3 with improved specs over Ankaa-2:

| Parameter | Our Estimate | Actual Value | Source |
|-----------|-------------|--------------|--------|
| T1 | 20 µs | **34 µs** | Rigetti press |
| 1Q error | 0.2% | **0.09%** | Official specs |
| 2Q error | 2.0% | **1.0%** | Official specs |
| Readout error | 2-3% | **5.3%** | arXiv paper |

The readout error was higher than expected - this is a known challenge for superconducting qubits and something circuit designers should account for.

## The Technology Divide

One pattern became clear: **trapped-ion and superconducting systems have fundamentally different trade-offs**.

### Trapped-Ion (IonQ)
- **Coherence**: T1 ~ 10 seconds, T2 ~ 1 second
- **Gate speed**: Slow (100s of microseconds)
- **Connectivity**: All-to-all
- **Best for**: Deep circuits, algorithms needing high fidelity

### Superconducting (IQM, Rigetti)
- **Coherence**: T1 ~ 20-40 microseconds
- **Gate speed**: Fast (20-100 nanoseconds)
- **Connectivity**: Fixed topology
- **Best for**: Wide circuits, NISQ algorithms, rapid iteration

This isn't news to quantum computing experts, but having concrete numbers makes the trade-offs tangible.

## How Other Frameworks Handle This

We also researched how Qiskit, Cirq, and Amazon Braket approach device noise:

**Qiskit Aer** offers `NoiseModel.from_backend()` which pulls live calibration data from IBM backends. This is powerful but requires authentication and network access.

**Amazon Braket** exposes device properties via their SDK:
```python
device = AwsDevice("arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3")
t1 = device.properties.standardized.oneQubitProperties['0'].T1
```

**Cirq** has `GoogleNoiseProperties` for Google's processors, with similar dynamic fetching.

The pattern is clear: vendors prefer dynamic APIs that return current calibration data, since quantum hardware characteristics drift over time.

## Our Approach: Static with Sources

For QuantumFlow, we chose **static pre-configured models with clear documentation**:

```python
IONQ_ARIA = DeviceNoiseModel(
    name="ionq_aria",
    num_qubits=25,
    t1=10.0,                        # 10s (IonQ specs)
    t2=1.0,                         # 1s (IonQ specs)
    single_qubit_gate_time=135e-6,  # 135µs (IonQ specs)
    two_qubit_gate_time=600e-6,     # 600µs (IonQ specs)
    single_qubit_gate_error=0.0006, # 0.06% (AWS Braket)
    two_qubit_gate_error=0.006,     # 0.6% (IonQ specs)
    readout_error=(0.0039, 0.0039), # 0.39% (AWS Braket)
)
```

Why static values?
1. **No authentication required** - works offline
2. **Reproducible** - same values for benchmarking
3. **Simple** - no network dependencies
4. **Documented** - sources cited in code

We may add optional dynamic fetching later, but the static approach serves most use cases.

## Using Device Noise Models

With accurate parameters, fidelity estimation becomes meaningful:

```python
import quantumflow as qf

# Create a 10-qubit GHZ circuit
circ = qf.ghz_circuit(range(10))

# Estimate fidelity on different devices
ionq_fid = qf.estimate_circuit_fidelity(circ, "ionq_aria")
iqm_fid = qf.estimate_circuit_fidelity(circ, "iqm_garnet")
rigetti_fid = qf.estimate_circuit_fidelity(circ, "rigetti_ankaa")

print(f"IonQ Aria:     {ionq_fid:.2%}")
print(f"IQM Garnet:    {iqm_fid:.2%}")
print(f"Rigetti Ankaa: {rigetti_fid:.2%}")
```

For deep circuits, IonQ's long coherence times win. For shallow, wide circuits, the superconducting speed advantage may dominate.

## Lessons Learned

1. **Don't guess - research**. Our initial estimates were off by 10x for some parameters.

2. **Cite your sources**. Device parameters change; knowing where values came from helps with updates.

3. **Understand the trade-offs**. Trapped-ion vs. superconducting isn't just about fidelity - it's about which *kind* of circuit you're running.

4. **Static is fine for most uses**. Dynamic APIs are great but add complexity. Start simple.

5. **Vendors publish more than you'd expect**. Between press releases, arXiv papers, and cloud provider docs, most parameters are findable.

## What's Next

We've opened [Issue #27](https://github.com/marqov-dev/quantumflow/issues/27) to track ongoing device parameter research. As vendors release new systems and update existing ones, we'll keep the models current.

For Marqov, these noise models feed into our backend selection logic. When a user submits a circuit, we can now estimate expected fidelity across backends and route accordingly.

---

*The device noise modeling work is part of [Epic #22: QuantumFlow Transpilation & Benchmarking Framework](https://github.com/marqov-dev/quantumflow/issues/22). See the [QuantumFlow repository](https://github.com/marqov-dev/quantumflow) for the implementation.*

## Sources

### IonQ Aria
- [IonQ Aria System Specifications](https://ionq.com/quantum-systems/aria)
- [IonQ Aria: Practical Performance](https://ionq.com/resources/ionq-aria-practical-performance)
- [AWS Braket - IonQ Aria Launch](https://aws.amazon.com/blogs/quantum-computing/amazon-braket-launches-ionq-aria-with-built-in-error-mitigation/)

### IQM Garnet
- [IQM Garnet 20Q Whitepaper 2024](https://meetiqm.com/wp-content/uploads/2025/04/IQM-Garnet-20Q-Whitepaper-2024.pdf)
- [arXiv:2408.12433 - Technology and Performance Benchmarks of IQM's 20-Qubit Quantum Computer](https://arxiv.org/abs/2408.12433)
- [AWS Braket - IQM](https://aws.amazon.com/braket/quantum-computers/iqm/)

### Rigetti Ankaa
- [Rigetti Ankaa-3 Launch Press Release](https://investors.rigetti.com/news-releases/news-release-details/rigetti-computing-launches-84-qubit-ankaatm-3-system-achieves)
- [arXiv:2410.05202 - Real-time Quantum Error Correction on Ankaa-2](https://arxiv.org/abs/2410.05202)
- [Rigetti QCS Documentation](https://qcs.rigetti.com/qpus)

### Cross-Platform Benchmarking
- [arXiv:2502.06471 - Evaluating QPU Performance at Large Width and Depth](https://arxiv.org/html/2502.06471v2)
