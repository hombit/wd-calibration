from pathlib import Path
from typing import Tuple, Union

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from calibration.data import narayan_etal2019_t1


__all__ = 'wd_coords',


def wd_coords(with_bright: bool = True) -> Tuple[np.ndarray, SkyCoord]:
    table = narayan_etal2019_t1()
    if not with_bright:
        table = table[3:]
    coords = SkyCoord(ra=table['R.A.'], dec=table[r'$\delta$'], unit=('hour', 'deg'))
    return table['Object'], coords


def coord_to_ipac(coord: SkyCoord, path: Union[str, Path]) -> None:
    table = Table(dict(ra=coord.ra, dec=coord.dec))
    table.write(path, format='ipac')