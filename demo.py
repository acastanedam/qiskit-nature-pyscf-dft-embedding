# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.drivers import MethodType, PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.settings import settings
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import SPSA
from qiskit_aer.primitives import Estimator as aer_estimator
from qiskit_aer import AerSimulator
from qiskit.algorithms.minimum_eigensolvers import VQE

from dft_embedding_solver import DFTEmbeddingSolver

from dft_embedding_solver.dft_embedding_solver import logging, LOGGER
from qiskit_nature import logging as nature_logging

settings.tensor_unwrapping = False
settings.use_pauli_sum_op = False
settings.use_symmetry_reduced_integrals = True

LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL)

nature_logging.set_levels_for_names(
    {"qiskit_nature": LOG_LEVEL, "qiskit": LOG_LEVEL})

LOGGER.setLevel(LOG_LEVEL)

callback_file = f"embd_callback.data"
intermediate_info = {"nfev": [], "parameters": [], "energy": []}

def callback(nfev, parameters, energy):
    intermediate_info["nfev"].append(nfev)
    intermediate_info["parameters"].append(parameters)
    intermediate_info["energy"].append(energy)
    output_list = [nfev, energy]
    [output_list.append(x) for x in parameters]
    with open(callback_file, "a") as interfile:
        print(
            *output_list,
            sep=",",
            file=interfile,
            flush=True,
        )

def _main():
    omega = 1.0

    # setup driver
    driver = PySCFDriver(
        atom="O 0.0 0.0 0.115; H 0.0 0.754 -0.459; H 0.0 -0.754 -0.459",
        basis="sto3g",
        method=MethodType.RKS,
        xc_functional=f"ldaerf + lr_hf({omega})",
        xcf_library="xcfun",
    )

    # specify active space
    num_particles = (2, 2)
    num_electrons = np.sum(num_particles)
    num_spatial_orbitals = 4
    active_space = ActiveSpaceTransformer(num_electrons=num_electrons, num_spatial_orbitals=num_spatial_orbitals)

    # setup solver
    mapper = ParityMapper(num_particles=num_particles)

    solver_type = "numpy"
    if solver_type == "numpy":
        solver = NumPyMinimumEigensolver()
        solver.filter_criterion = lambda state, val, aux: np.isclose(
            aux["ParticleNumber"][0], num_electrons
        )
    elif solver_type == "vqe":
        # setup the initial state for the ansatz
        init_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)

        # setup the ansatz for VQE
        num_qubits = 2 * num_spatial_orbitals
        ansatz = EfficientSU2(num_qubits, reps=1, insert_barriers=True)

        # add the initial stateb
        ansatz.compose(init_state, front=True, inplace=True)

        # setup and run VQE
        backend = AerSimulator(method="automatic")
        estimator = aer_estimator(
            backend_options=backend.options._fields,
        )

        # setup optimizer
        optimizer = SPSA(maxiter=10)

        solver = VQE(estimator, ansatz, optimizer)

    algo = GroundStateEigensolver(mapper, solver)

    dft_solver = DFTEmbeddingSolver(active_space, algo, callback=callback)

    # NOTE: By default, no mixing will be applied to the active density.
    # Uncomment any of the following to apply the given mixing method.
    # (1) density mixing using the last `history_length` number of densities
    # history_length = 10
    # dft_solver.damp_density = lambda history: np.mean(history[-history_length:])
    # (2) density mixing using a constant damping parameter `_alpha`
    alpha = 0.5
    dft_solver.damp_density = (
        lambda history: alpha * history[-2] + (1.0 - alpha) * history[-1]
        if len(history) > 1
        else history[-1]
    )

    result = dft_solver.solve(driver, omega)
    print(result)


if __name__ == "__main__":
    _main()
