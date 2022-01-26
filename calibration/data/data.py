from functools import lru_cache
from importlib.resources import open_binary

from astropy.io import ascii
from astropy.table import Table

from calibration import data


@lru_cache(1)
def narayan_etal2019_t1() -> Table:
    """Slightly modified Table 1 from Narayan et al. 2019

    https://ui.adsabs.harvard.edu/abs/2019ApJS..241...20N/abstract
    """
    with open_binary(data, 'WDphot_ILAPHv5_abmag.tex') as fh:
        table = ascii.read(fh, format='latex')
    return table