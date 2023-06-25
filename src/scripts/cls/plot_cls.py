"""Plot the P2LS data."""

from math import pi

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from showyourwork.paths import user as user_paths

paths = user_paths()

##############################################################################

npzfile = np.load(paths.data / "cls.npy")

lmax = 1500

ls = npzfile["ls"][: lmax + 1]
cls_EE = npzfile["cls_EE"][: lmax + 1]
cls_EL = npzfile["cls_EL"][: lmax + 1]
cls_LL = npzfile["cls_LL"][: lmax + 1]

##############################################################################

fig = plt.figure(figsize=(5, 4))
gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1])
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])

ax0.plot(
    ls,
    ls * (ls + 1) * cls_LL / 2 / pi,
    label=r"$(\Delta T)^2 (\Delta T)^2$",
    c="tab:blue",
    lw=2,
)
ax0.plot(
    ls,
    5e1 * ls * (ls + 1) * cls_EL / 2 / pi,
    label=r"$50 \mathcal{Y}(\Delta T)^2$",
    c="tab:purple",
    lw=2,
)
ax0.plot(
    ls,
    1e3 * ls * (ls + 1) * cls_EE / 2 / pi,
    label=r"$10^{3} \mathcal{Y}\mathcal{Y}$",
    c="tab:red",
    lw=2,
)
ax0.set(ylabel=r"$A_s^{-2} \, \ell(\ell+1)C_\ell/2\pi$")
ax0.legend()
ax0.grid(visible=True, alpha=0.3)
ax0.set_axisbelow(b=True)
ax0.set_xticklabels([], visible=False)

ax1.plot(
    ls,
    np.log10(cls_EL / cls_LL),
    label=r"$\frac{\mathcal{Y}{(\Delta T)^2}}{ (\Delta T)^2}$",
    c="tab:purple",
    lw=2,
)
ax1.plot(
    ls,
    np.log10(cls_EE / cls_LL),
    label=r"$\frac{\mathcal{Y}\mathcal{Y}}{ (\Delta T)^2}$",
    c="tab:red",
    lw=2,
)
ax1.set(xlabel=r"$\ell$", ylabel=r"$\log_{10}C_{\ell,1}/C_{\ell,2}$")
ax1.legend()
ax1.grid(visible=True, alpha=0.3)
ax1.set_axisbelow(b=True)

fig.tight_layout()

fig.savefig(paths.figures / "cls.pdf")
plt.close(fig)
