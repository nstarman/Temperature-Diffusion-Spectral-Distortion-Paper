"""Utilities for :mod:`classy`."""

from __future__ import annotations

__all__: list[str] = []

import configparser
from typing import TYPE_CHECKING, Any
from typing import Annotated as Ann

from classy import Class

from .wrapper import StandardCosmologyWrapper

if TYPE_CHECKING:
    from os import PathLike

    from typing_extensions import Doc

##############################################################################
# CODE
##############################################################################


def _flatten_dict(
    d: Ann[
        dict[str, Any | dict[str, Any] | dict[str, dict[str, Any]]],
        Doc("Dictionary to flatten"),
    ],
    /,
) -> Ann[dict[str, Any], Doc("Flattened dictionary")]:
    """Recursively flatten nested dictionary.

    Examples
    --------
    >>> _flatten_dict({"a": 1, "b": 2, "c": 3})
    {'a': 1, 'b': 2, 'c': 3}

    >>> _flatten_dict({"a": 1, "_": {"c": 2, "d": 3}})
    {'a': 1, 'c': 2, 'd': 3}

    """
    out: dict[str, Any] = {}
    for key, val in d.items():
        out.update(_flatten_dict(val) if isinstance(val, dict) else {key: val})
    return out


class CLASSConfigParser(configparser.ConfigParser):
    """Parser for CLASS config files.

    This is a subclass of :class:`configparser.ConfigParser` that overrides
    :meth:`optionxform` to aalways return the string form of the input. This
    is necessary because CLASS config files should stay as strings.

    Examples
    --------
    An example file::

        [background parameters]

        # Hubble parameter:
        h =0.674
        # Photon density:
        T_cmb = 2.7255
        # Baryon density:
        omega_b = 0.0224
        # Ultra-relativistic species / massless neutrino density:
        N_ur = 3.046
        # Density of cdm (cold dark matter):
        omega_cdm = 0.12
        # Curvature:
        Omega_k = 0.
        # Scale factor today 'a_today'
        a_today = 1.

        ...

    """

    def optionxform(self, optionstr: str) -> str:
        """Return string as-is."""
        return str(optionstr)


def read_params_from_ini(
    filename: Ann[str | bytes | PathLike[str], Doc("Path to INI file.")], /
) -> Ann[dict[str, Any], Doc("Dictionary of parameters. Nested dict are flattened.")]:
    """Read parameters from INI file."""
    config = CLASSConfigParser()
    config.read(str(filename))
    return _flatten_dict({k: dict(config[k]) for k in config.sections()})


def cosmo_from_params(
    params: Ann[dict[str, Any], Doc("Dictionary of Paramters")], /
) -> Ann[StandardCosmologyWrapper, Doc("CLASS wrapper")]:
    """Get a CLASS instance from a parameter dictionary."""
    # Create an instance of the CLASS wrapper
    cosmo = Class()
    # set parameters
    cosmo.set(params)
    # Run the whole code.
    cosmo.compute()
    return StandardCosmologyWrapper(cosmo)
