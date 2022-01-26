from functools import lru_cache
from importlib.resources import open_binary

from astropy.io import ascii

from calibration import data


@lru_cache(1)
def narayan_etal2019_t1():
    with open_binary(data, 'WDphot_ILAPHv5_abmag.tex') as fh:
        table = ascii.read(fh, format='latex')
    return table