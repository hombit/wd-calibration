from functools import lru_cache
from importlib.resources import open_binary

import numpy as np
from astropy.io import ascii
from astropy.table import Table

from calibration import data

__all__ = 'calamida_lco_t2',


EXCLUDE = [
    'SDSSJ20372.169-051302.964',
    'SDSSJ041053.632-063027.580',
    'WD0554-165',
    'SDSSJ172135.97+294016.0',
    'WD0757-606',
    'WD0418-534',
]


@lru_cache(1)
def calamida_lco_t2() -> Table:
    """Table 2 from Future Calamida at al LCO WD paper"""
    with open_binary(data, 'calamida_lco_table2.tex') as fh:
        table = ascii.read(fh, format='latex')
    table = table[~np.isin(table['Orig. name'], EXCLUDE)]
    return table


