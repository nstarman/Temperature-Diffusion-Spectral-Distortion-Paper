"""The cosmology."""

import sys

from showyourwork.paths import user as user_paths

paths = user_paths()

sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.src.qclassy.io import cosmo_from_params, read_params_from_ini

##############################################################################

filename = str(paths.static / "planck18_parameters.ini")

params = read_params_from_ini(filename)

cosmo = cosmo_from_params(params)
