# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from __future__ import annotations

import numpy as np

import pyttb as ttb


class TestISVD:
    def test_init_from_raw(self):
        rng = np.random.default_rng(seed=0)
        m = 64
        n = 64
        A = rng.normal(size=(m, n))
        tol = 1.0e-03
        isvd = ttb.isvd(A, tol)
        assert True

    def test_init_from_svd(self):
        rng = np.random.default_rng(seed=0)
        m = 64
        n = 64
        A = rng.normal(size=(m, n))
        U, s, Vt = np.linalg.svd(A)
        tol = 1.0e-03
        isvd = ttb.isvd((U, s, Vt), tol)
        assert True
