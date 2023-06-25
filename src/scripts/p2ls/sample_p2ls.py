"""Simulate the P2LS data on a grid of spll, sprp."""

import sys

import asdf
import numpy as np
from showyourwork.paths import user as user_paths

paths = user_paths()

sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.src.prob_2ls import P3D_Distribution

##############################################################################

try:
    snkmkp: dict[str, int] = snakemake.params  # type: ignore[name-defined]
except NameError:
    snkmkp = {"seed": 42}

rng = np.random.default_rng(snkmkp["seed"])

# -----------------------------------------------------------------
# Load data

with asdf.open(paths.data / "P2LS.asdf", lazy_load=False, copy_arrays=True) as af:
    spll = af["spll"]
    sprp = af["sprp"]
    Pspllsprp = af["Pspllsprp"]

# -----------------------------------------------------------------
# Sample

distribution = P3D_Distribution.from_Pspllsprp(
    spll, sprp, Pspllsprp, spll_b=2.9, pdf_cutoff=1e-4
)

samples = distribution.rvs(size=int(1e7), rng=rng)
samples = samples[~np.isnan(samples).any(axis=1)]


##############################################################################

with (paths.data / "p2ls_samples.npy").open("wb") as f:
    np.save(f, samples)
