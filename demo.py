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

from dft_embedding_solver import DFTEmbeddingSolver

settings.tensor_unwrapping = False
settings.use_pauli_sum_op = False
settings.use_symmetry_reduced_integrals = True


from dft_embedding_solver.dft_embedding_solver import logging, LOGGER
from qiskit_nature import logging as nature_logging

LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL)

nature_logging.set_levels_for_names(
    {"qiskit_nature": LOG_LEVEL, "qiskit": LOG_LEVEL})

LOGGER.setLevel(LOG_LEVEL)

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
    active_space = ActiveSpaceTransformer(4, 4)

    # setup solver
    mapper = ParityMapper(num_particles=(2, 2))
    solver = NumPyMinimumEigensolver()
    solver.filter_criterion = lambda state, val, aux: np.isclose(
        aux["ParticleNumber"][0], 4.0
    )
    algo = GroundStateEigensolver(mapper, solver)

    dft_solver = DFTEmbeddingSolver(active_space, algo)

    # NOTE: By default, no mixing will be applied to the active density.
    # Uncomment any of the following to apply the given mixing method.
    # (1) density mixing using the last `history_length` number of densities
    # history_length = 10
    # dft_solver.damp_density = lambda history: np.mean(history[-history_length:])
    # (2) density mixing using a constant damping parameter `_alpha`
    # alpha = 0.5
    # dft_solver.damp_density = (
    #     lambda history: alpha * history[-2] + (1.0 - alpha) * history[-1]
    #     if len(history) > 1
    #     else history[-1]
    # )

    result = dft_solver.solve(driver, omega)
    print(result)


if __name__ == "__main__":
    _main()
