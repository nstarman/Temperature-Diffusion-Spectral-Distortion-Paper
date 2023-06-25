"""Plot the P2LS data."""

import asdf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from showyourwork.paths import user as user_paths

paths = user_paths()

# -----------------------------------------------------------------
# Load data

with asdf.open(paths.data / "variables.asdf", lazy_load=False, copy_arrays=True) as vs:
    s_recombination = vs["s_recombination"]

with asdf.open(paths.data / "P2LS.asdf", lazy_load=False, copy_arrays=True) as af:
    spll = af["spll"]
    sprp = af["sprp"]
    Pspllsprp = af["Pspllsprp"]


# -----------------------------------------------------------------
# Plot

fig, ax = plt.subplots(figsize=(6, 3))


Spll, Sprp = np.meshgrid(spll, sprp, indexing="ij")
im = ax.scatter(
    Spll, np.log10(Sprp), c=Pspllsprp, s=1, norm=LogNorm(vmin=1e-2), rasterized=True
)
ax.axvline(s_recombination, c="k", ls="--")

ax.set_xlabel(r"$s_{||}$", fontsize=15)
ax.set_ylabel(r"$\log_{10}s_{\perp}$", fontsize=15)

cbar = plt.colorbar(im, ax=ax)
cbar.ax.set_ylabel(r"$\log_{10}\left(2\pi\mathcal{P}(s_{||}, s_{\perp}, \phi)\right)$")
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{np.log10(float(y)):g}"))

ax.set_xlim(1.5, np.nanmax(Spll))
ax.set_ylim(np.log10(np.nanmin(Sprp)), np.log10(np.nanmax(Sprp)))

fig.tight_layout()

plt.savefig(paths.figures / "P2LS.pdf")
