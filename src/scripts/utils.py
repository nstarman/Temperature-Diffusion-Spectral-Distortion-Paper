"""Simulate the P2LS data on a grid of spll, sprp."""

import numpy as np


def coeff_exp(x: float) -> tuple[float, float]:
    """Format a float in exponential notation."""
    exponent = int(np.floor(np.log10(abs(x))))
    coeff = np.round(x / float(10**exponent), 2)
    return coeff, exponent


def scientific_format(x: float, /, decimals: int = 0) -> str:
    """Format a float in scientific notation."""
    coeff, exponent = coeff_exp(x)
    return rf"{coeff:.{decimals}f} \times 10^{{{exponent:d}}}"
