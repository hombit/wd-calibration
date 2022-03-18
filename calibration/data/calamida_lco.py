from functools import lru_cache
from importlib.resources import open_binary

from astropy.io import ascii
from astropy.table import Table

from calibration import data

__all__ = 'calamida_lco_t2',


@lru_cache(1)
def calamida_lco_t2() -> Table:
    """Table 2 from Future Calamida at al LCO WD paper"""
    with open_binary(data, 'calamida_lco_table2.tex') as fh:
        table = ascii.read(fh, format='latex')
    return table


