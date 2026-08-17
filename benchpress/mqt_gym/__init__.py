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

try:
    import mqt.core.mlir  # noqa: F401
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "benchpress.mqt_gym requires an MLIR-enabled mqt.core build "
        "(mqt.core.mlir). Install the compatible revision pinned in "
        "requirements-mqt.txt with MLIR_DIR set; the stable PyPI wheel does "
        "not ship MLIR yet."
    ) from exc

from benchpress.config import Configuration

Configuration.gym_name = "mqt"
