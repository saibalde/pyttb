# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

"""Incremental singular value decomposition."""

from __future__ import annotations

import numpy as np


class isvd:
    r"""Incremental singular value decomposition.

    Initialize an incremental SVD factorization from either a data matrix
    :math:`A` or its truncated SVD approximation :math:`A \approx U S V^\top`
    satisfying the error bound

    .. math::

        \| A - U S V^\top \|_F \leq \tau \| A \|

    where :math:`\tau` is the prescribed relative error tolerance. If the
    initial data is a SVD tuple :math:`(U, S, V)`, then the usual properties
    for the factorization and the corresponding approximation error
    :math:`E = A - U S V^\top` are assumed to be true:

    .. math::

        U^\top U = I_m, \; V^\top V = I_n, \; U^\top E = 0, \; E V = 0

    If initial data norm :math:`\| A \|_F` and the actual error
    :math:`\| A - U S V^\top \|_F` are known, then error tracking is activated
    (these inputs quantify the slack in the error bound and subsequent
    streaming updates attempt to reduce this slack as much as possible,
    potentially reducing the rank of the final factorization). When a raw
    matrix :math:`A` is used as the initial input, then these quantities are
    automatically computed.

    Parameters
    ----------
    data:
        Initialization data for the incremental factorization, either provided
        in raw form as a data matrix :math:`A`, or as the initial truncated
        factorization as a SVD tuple :math:`(U, S, V)` (the format is assumed
        to mimic the output of `numpy.linalg.svd`, i.e. the middle element is a
        1D array of singular values rather than a diagonal matrix, and the last
        element is transposed).
    tolerance:
        Relative error tolerance :math:`\tau` of the initial factorization (in
        Frobenius norm). This tolerance is maintaind throughout subsequent
        incremental updates.
    frobenius_norm:
        Norm of the original data. If initial `data` is a raw matrix, this
        argument is ignored (the data norm is computed internally). To indicate
        data norm is unknown when initial data is factorized, pass any negative
        value.
    frobenius_error:
        Norm of the original approximation error. If initial `data` is a raw
        matrix, this argument is ignored (the error norm is computed
        internally). To indicate error norm is unknown when initial data is
        factorized, pass any negative value.
    """

    def __init__(
        self,
        data: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray],
        tolerance: float,
        frobenius_norm: float = -1.0,
        frobenius_error: float = -1.0,
    ):
        def is_matrix(A):
            return (
                isinstance(A, np.ndarray)
                and A.ndim == 2
                and np.issubdtype(A.dtype, np.floating)
            )

        def is_svd_tuple(data):
            if not isinstance(data, tuple) or len(data) != 3:
                return False
            U, s, Vt = data
            return (
                isinstance(U, np.ndarray)
                and U.ndim == 2
                and np.issubdtype(U.dtype, np.floating)
                and isinstance(s, np.ndarray)
                and s.ndim == 1
                and np.issubdtype(s.dtype, np.floating)
                and isinstance(Vt, np.ndarray)
                and Vt.ndim == 2
                and np.issubdtype(Vt.dtype, np.floating)
                and U.shape[1] == s.shape[0] == Vt.shape[0]
            )

        if is_matrix(data):
            U, s, Vt = np.linalg.svd(data, full_matrices=False)

            norm_squared = np.sum(s**2)
            squared_error_tolerance = tolerance**2 * norm_squared

            rank = s.size
            error_squared = 0.0
            while rank > 1:
                new_error_squared = error_squared + s[rank - 1] ** 2
                if new_error_squared > squared_error_tolerance:
                    break
                rank -= 1
                error_squared = new_error_squared

            self._num_row = data.shape[0]
            self._num_col = data.shape[1]
            self._U = U[:, :rank].copy()
            self._s = s[:rank].copy()
            self._Vt = Vt[:rank, :].copy()
            self._tolerance = tolerance
            self._rank = rank
            self._track_error = True
            self._frobenius_norm = np.sqrt(norm_squared)
            self._frobenius_error = np.sqrt(error_squared)
        elif is_svd_tuple(data):
            U, s, Vt = data
            self._num_row = U.shape[0]
            self._num_col = Vt.shape[1]
            self._U = U.copy()
            self._s = s.copy()
            self._Vt = Vt.copy()
            self._tolerance = tolerance
            self._rank = s.shape[0]
            self._track_error = frobenius_norm > 0.0 and frobenius_error > 0.0
            self._frobenius_norm = frobenius_norm
            self._frobenius_error = frobenius_error
        else:
            raise ValueError("initial data must be a 2D array or a SVD tuple")
