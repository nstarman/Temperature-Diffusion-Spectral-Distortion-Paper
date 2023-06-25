"""Plot the P2LS data."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from showyourwork.paths import user as user_paths

paths = user_paths()

##############################################################################

# Load data
chis_EE, C_EE_corr = np.load(paths.data / "C_EE_corr.npy")
chis_EL, C_EL_corr = np.load(paths.data / "C_EL_corr.npy")
chis_LL, C_LL_corr = np.load(paths.data / "C_LL_corr.npy")

# Convert to degrees
chis_EE *= 180 / np.pi
chis_EL *= 180 / np.pi
chis_LL *= 180 / np.pi

##############################################################################

fig = plt.figure(figsize=(5, 4))
gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1])
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])


ax0.plot(
    chis_LL,
    C_LL_corr,
    label=r"$C^{(\Delta T)^2(\Delta T)^2}(\theta) / A_s^2$",
    c="tab:blue",
    lw=2,
)
ax0.plot(
    chis_EL,
    C_EL_corr,
    label=r"$C^{\mathcal{Y}(\Delta T)^2}(\theta) / A_s^2$",
    c="tab:purple",
    lw=2,
)
ax0.plot(
    chis_EL,
    C_EE_corr,
    label=r"$C^{\mathcal{Y}\mathcal{Y}}(\theta) / A_s^2$",
    c="tab:red",
    lw=2,
)
ax0.set_yscale("log")
ax0.set_ylabel(r"$\log_{10}$ Reduced 4-Point $C^{XX}$")
ax0.legend()
ax0.grid(visible=True, alpha=0.3)
ax0.set_axisbelow(b=True)
ax0.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{np.log10(float(y)):g}"))
ax0.set_xticklabels([], visible=False)


ax1.plot(
    chis_EL,
    C_EL_corr / C_EE_corr,
    label=r"$\frac{C^{\mathcal{Y}(\Delta T)^2}}{ C^{\mathcal{Y}\mathcal{Y}}}(\theta)$",  # noqa: E501
    c="tab:red",
    lw=2,
)
ax1.plot(chis_EL, np.ones_like(chis_EL), c="tab:purple", ls="--")
ax1.plot(
    chis_EL,
    C_EL_corr / C_LL_corr,
    label=r"$\frac{C^{\mathcal{Y}(\Delta T)^2}}{ C^{(\Delta T)^2(\Delta T)^2}}(\theta)$",  # noqa: E501
    c="tab:blue",
    lw=2,
)
ax1.set_yscale("log")
ax1.set_xlabel(r"$\theta$ [deg]")
ax1.set_ylabel(r"$\log_{10}$ ratio")
ax1.legend(loc="lower right", ncols=1)
ax1.grid(visible=True, alpha=0.3)
ax1.set_axisbelow(b=True)

ax1.yaxis.set_ticks(np.geomspace(0.01, 100, 3))
ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{np.log10(float(y)):g}"))

fig.tight_layout()

fig.savefig(paths.figures / "correlation.pdf")
plt.close(fig)
