import numpy as np
from random import choices, sample


from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import PassManager
from qiskit.quantum_info import Clifford, Pauli

"""
Classes and functions to run accreditation.
P.S. "Accreditation" is a really ugly name. Can we change it? :p
"""

# The u3 angles for all the 24 one-qubit gate Clifford gates
CLIFFORD_U3_ANGLES = [
    [0.0, 3.141592653589793, 3.141592653589793],
    [3.141592653589793, 3.141592653589793, 3.141592653589793],
    [0.0, 6.283185307179586, 0.0],
    [-1.5707963267948966, 6.283185307179586, 1.5707963267948966],
    [-1.5707963267948966, 4.71238898038469, 4.71238898038469],
    [1.5707963267948966, 4.71238898038469, 4.71238898038469],
    [1.5707963267948966, 4.71238898038469, 1.5707963267948966],
    [-1.5707963267948966, 4.71238898038469, 1.5707963267948966],
    [-3.141592653589793, 4.71238898038469, 4.71238898038469],
    [0.0, 4.71238898038469, 4.71238898038469],
    [3.141592653589793, 4.71238898038469, 1.5707963267948966],
    [0.0, 4.71238898038469, 1.5707963267948966],
    [3.141592653589793, 4.71238898038469, 3.141592653589793],
    [0.0, 4.71238898038469, 3.141592653589793],
    [-3.141592653589793, 4.71238898038469, 6.283185307179586],
    [0.0, 4.71238898038469, 0.0],
    [1.5707963267948966, 4.71238898038469, 3.141592653589793],
    [-1.5707963267948966, 4.71238898038469, 3.141592653589793],
    [-1.5707963267948966, 4.71238898038469, 6.283185307179586],
    [-4.71238898038469, 4.71238898038469, 6.283185307179586],
    [1.5707963267948966, 3.141592653589793, 3.141592653589793],
    [-1.5707963267948966, 3.141592653589793, 3.141592653589793],
    [-1.5707963267948966, 6.283185307179586, 6.283185307179586],
    [-3.141592653589793, 6.283185307179586, 7.853981633974483],
]


class ConvertCircuitToTrap(TransformationPass):
    r"""
    A pass that converts a quantum circuit to a template for a trap.

    It scans through the circuit and replaces every one-qubit gate with parametrized U3 gate.
    Each U3 gate owns three parameters named "p[0]_i", "p[1]_i", and "p[2]_i" that represent
    "theta", "phi", and "lam" respectively, where "i" is the one-qubit gate index.

    .. note ::

        This pass make no attempts to minimize the number of U3 gates in the returned circuit. So
        for example, it does not skip identities, and it does not merge adjacent one-qubit gates.
    """

    def run(self, dag: DAGCircuit, seed=None) -> DAGCircuit:
        idx = 0
        for node in dag.op_nodes():
            op = node.op
            if op.name == "barrier" or op.num_qubits != 1 or op.num_clbits != 0:
                continue

            param_gate = QuantumCircuit(1)
            param = ParameterVector(f"p_{idx}", 3)
            param_gate.u(theta=param[0], phi=param[1], lam=param[2], qubit=0)
            dag.substitute_node_with_dag(node, circuit_to_dag(param_gate))

            idx += 1

        return dag


class Accreditation:
    r"""
    A class to perform pre-processing and post-processing for accreditation.
    """

    def __init__(self, target_circuit: QuantumCircuit) -> None:
        self._target_circuit = target_circuit

    @property
    def target_circuit(self) -> QuantumCircuit:
        r"""The target circuit for this accreditation run."""
        return self._target_circuit

    def make_pub_elements(
        self, num_traps: int = 10
    ) -> tuple[QuantumCircuit, list[Pauli], list[list[float]]]:
        r"""
        Generates the elements of the PUBs for running ``num_traps`` trap circuits, and namely:

        * One "template" trap circuit containing parametric U3 gates.
        * A set of ``num_traps`` observables to estimate the expectation value of (one per
          trap circuit).
        * A list of ``num_traps`` parameter sets (one per trap circuit) to turn the trap template
          into a concrete, non-parametric trap circuit.

        When running an estimator job with the given PUBs, this will send down (and compile) a
        single circuit, and it will "zip" observables and parameters.

        .. note ::

            Depending on the target circuit's layout, the returned circuit and observables may need
            to be transpiled before running them on a backend.
        """
        # Create a template for every trap circuit
        pm = PassManager([ConvertCircuitToTrap()])
        trap_template = pm.run(self.target_circuit)

        # Initialize lists to store the observables and the parameter set
        observables = []
        params_set = []

        # TODO: This can probably be parallelized if speedup is needed
        for _ in range(num_traps):
            # Create a copy of the template
            trap = trap_template.copy()

            # Choose a triplet of U3 angles for every parametric and assign the parameters
            chosen_u3_cliff_angles = choices(
                CLIFFORD_U3_ANGLES, k=trap.num_parameters // 3
            )
            for idx, (theta, phi, lam) in enumerate(chosen_u3_cliff_angles):
                trap.assign_parameters(
                    {f"p_{idx}[0]": theta, f"p_{idx}[1]": phi, f"p_{idx}[2]": lam},
                    inplace=True,
                )

            # Generate an n-qubit Pauli-Z by sampling I with prob. 1/4 and Z with prob. 3/4
            z = choices([True, False], [0.75, 0.25], k=trap.num_qubits)
            pauli_in = Pauli((z, [False] * trap.num_qubits, [0] * trap.num_qubits))

            # Evolve ``Pauli_in`` to get the output observable
            obs = pauli_in.evolve(Clifford(trap), frame="s")

            params_set.append(list(np.array(chosen_u3_cliff_angles).flatten()))
            observables.append(obs)

        # Return the template trap, the whole set of observables, and the whole set of parameters.
        # When running an estimator, this will send down (and compile) a single circuit.
        return trap_template, observables, params_set
