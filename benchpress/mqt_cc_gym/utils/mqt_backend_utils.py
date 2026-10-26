# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# Modified in 2026 to share backend discovery with the Qiskit gym.
"""Backend helpers for the mqt-cc gym."""

from benchpress.qiskit_gym.utils.qiskit_backend_utils import (
    get_qiskit_bench_backend as get_mqt_bench_backend,
)

__all__ = ["get_mqt_bench_backend"]
