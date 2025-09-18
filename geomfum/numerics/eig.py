import gs.backend as gs
import scipy

import geomfum.backend as xgs


class ScipyEigsh:
    def __init__(
        self,
        spectrum_size=6,
        sigma=None,
        which="LM",
    ):
        self.spectrum_size = spectrum_size
        self.sigma = sigma
        self.which = which

    def __call__(self, A, M=None):
        vals, vecs = scipy.sparse.linalg.eigsh(
            xgs.sparse.to_scipy_csc(A),
            k=self.spectrum_size,
            M=xgs.sparse.to_scipy_dia(M),
            sigma=self.sigma,
            which=self.which,
        )
        return gs.from_numpy(vals), gs.from_numpy(vecs)
