"""
"""

import warnings

from qiskit.aqua.utils.backend_utils import is_ibmq_provider

from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit, IBMQBackend


def _call_compile(
    pytket_backend,
    pytket_circuit,
    optimisation_level,
    auto_regress,
):
    """ """
    try:
        pytket_backend.compile_circuit(
            pytket_circuit, optimisation_level=optimisation_level)
    except:
        warnings.warn(
            'pytket compile at optimisation_level='
            + f'{optimisation_level}' + ' failed'
        )
        if auto_regress and optimisation_level > 0:
            _call_compile(
                pytket_backend,
                pytket_circuit,
                optimisation_level-1,
                auto_regress,
            )
        else:
            raise


def compile_for_backend(
    backend,
    circuit,
    optimisation_level=2,
    auto_regress=False,
):
    """
    Use pytket to compile single circuit or list of circuits for a IBMQ
    backend, preserves circuit names.

    Parameters
    ----------
    backend : qiskit backend
        IBMQ backend to compile circuits for
    circuit : qiskit.QuantumCircuit, list(qiskit.QuantumCircuit)
        Circuit or list circuits to compile
    optimisation_level : int, optional
        Optimisation level argument passed to pytket compiler
    auto_regress : boolean, optional
        If set to True and compilation at optimisation_level fails, function
        will automatically decrease optimisation_level (downwards in steps of
        1) and try again to compile at lower optimisation_level

    Returns
    -------
    qiskit.QuantumCircuit, list(qiskit.QuantumCircuit)
        return matches format of arg
    """
    if not is_ibmq_provider(backend):
        return circuit

    pytket_backend = IBMQBackend(backend.name(),
                                 hub=backend.hub,
                                 group=backend.group,
                                 project=backend.project,)

    single_circ = False
    if not isinstance(circuit, list):
        single_circ = True
        circuit = [circuit]

    transpiled_circuits = []
    for circ in circuit:
        pytket_circuit = qiskit_to_tk(circ)
        _call_compile(
            pytket_backend,
            pytket_circuit,
            optimisation_level,
            auto_regress,
        )
        transpiled_circuits.append(tk_to_qiskit(pytket_circuit))

        # preserve exact parameter objs
        # NOTE: this may be made redundant in a future pytket update
        updates = {
            p: next(x for x in circ.parameters if x.name == p.name)
            for p in transpiled_circuits[-1].parameters
        }
        transpiled_circuits[-1].assign_parameters(updates, inplace=True)

        # preserve circuit name
        transpiled_circuits[-1].name = circ.name

    if single_circ:
        return transpiled_circuits[0]
    return transpiled_circuits
