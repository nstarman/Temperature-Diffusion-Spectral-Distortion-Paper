"""Simulate the P2LS data on a grid of spll, sprp."""

import sys

import asdf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from showyourwork.paths import user as user_paths

paths = user_paths()

sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.src.distances import r_distance_between_s

##############################################################################

rng = np.random.default_rng(0)

# -----------------------------------------------------------------
# Load

with asdf.open(paths.data / "variables.asdf") as af:
    Leq = af["Leq"]

with (paths.data / "p2ls_samples.npy").open("rb") as f:
    samples = np.load(f)

# -----------------------------------------------------------------
# Plot

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Samples
axs[0].scatter(
    samples[:, 0], samples[:, 1], s=0.1, c="k", label="samples", rasterized=True
)
axs[0].set_ylim(-0.005, None)
axs[0].set_xlabel(r"$s_{||}$", fontsize=14)
axs[0].set_ylabel(r"$s_{\perp}$", fontsize=14)

# Logarithmic samples
axs[1].hexbin(samples[:, 0], np.log10(samples[:, 1]), norm=LogNorm())
axs[1].set_xlabel(r"$s_{||}$", fontsize=14)
axs[1].set_ylabel(r"$\log_{10}s_{\perp}$", fontsize=14)

# Sample separation
num_points = 10_000_000
indx1, indx2 = rng.integers(0, len(samples), size=(2, num_points))
dists = r_distance_between_s(samples[indx1], samples[indx2], Leq=1.0)
axs[2].hist(
    np.log10(dists),
    bins=100,
    range=(-4, 1),
    color="k",
    histtype="step",
    label="histogram",
    log=True,
)
axs[2].set_xlabel(r"$\Delta{s}$", fontsize=14)
axs[2].set_ylabel(r"frequency", fontsize=14)

fig.tight_layout()
fig.savefig(str(paths.figures / "P2LS_samples.pdf"))
