from collections import defaultdict
from functools import lru_cache
from importlib.resources import contents, open_binary
from io import BytesIO
from pathlib import Path
from typing import Iterable, Union

import numpy as np
from astropy import units
from astropy.io import ascii, fits
from astropy.table import QTable

from calibration.data import ck04models
from calibration.units import FLAM


__all__ = ('get_castelli_kurucz_model', 'castelli_kurucz_locus',)


def _parse_metallicity(s: str) -> float:
    assert len(s) == 3
    abs_value = float(s[1:]) / 10.0
    match s[0]:
        case 'p':
            return abs_value
        case 'm':
            return -abs_value
        case _:
            raise ValueError(f'unknown prefix {s[0]}')


# It loads everything into RAM, not very optimal!
@lru_cache(maxsize=1)
def castelli_kurucz_tables() -> dict[float, dict[float, dict[float, QTable]]]:
    filenames = [fname for fname in contents(ck04models) if fname.endswith('.fits.gz') and fname.startswith('ck')]
    tables = defaultdict(lambda: defaultdict(dict))
    for fname in filenames:
        with open_binary(ck04models, fname) as fh:
            data = fits.getdata(fh, 1)

        basename = fname.removesuffix('.fits.gz')
        assert basename != fname
        assert basename[:2] == 'ck'
        metallicity = _parse_metallicity(basename[2:5])
        temperature = float(basename[6:])

        wavelength = data['WAVELENGTH'] * units.angstrom

        for column in data.dtype.names:
            if column == 'WAVELENGTH':
                continue

            assert column.startswith('g')
            assert len(column) == 3
            lg_g = float(column[1:]) / 10.0

            flux = data[column]
            # Zero flux means no data
            if np.all(flux == 0.0):
                continue
            flux = flux * FLAM

            tables[metallicity][temperature][lg_g] = QTable(data=dict(wave=wavelength, flux=flux), copy=False)
    return tables


def _nearest(it: Iterable[float], value: float) -> float:
    """O(n) approach to find the nearest match in the collection"""
    array = np.fromiter(it, dtype=float)
    idx = np.argmin(np.abs(array - value))
    return array[idx]


def get_castelli_kurucz_model(*, metallicity: float, temperature: float, lg_g: float) -> QTable:
    """Castelli–Kurucz model for parameters closest to the given"""
    tables = castelli_kurucz_tables()
    tables = tables[_nearest(tables, metallicity)]
    tables = tables[_nearest(tables, temperature)]
    table = tables[_nearest(tables, lg_g)]
    return table


CASTELLI_KURUCZ_LOCUS_TEMP_LGG = (
    (45000, 4.5),
    (41000, 4.5),
    (40000, 4.0),
    (39000, 4.0),
    (38000, 4.0),
    (37000, 4.0),
    (36000, 4.0),
    (35000, 4.0),
    (34000, 4.0),
    (33000, 4.0),
    (32000, 4.0),
    (30000, 4.0),
    (25000, 4.0),
    (19000, 4.0),
    (15000, 4.0),
    (12000, 4.0),
    (9500, 4.0),
    (9250, 4.0),
    (8750, 4.0),
    (8250, 4.0),
    (7250, 4.0),
    (7000, 4.0),
    (6500, 4.0),
    (6250, 4.0),
    (6000, 4.5),
    (5750, 4.5),
    (5500, 4.5),
    (5250, 4.5),
    (4750, 4.5),
    (4500, 4.5),
    (4250, 4.5),
    (4000, 4.5),
    (3750, 4.5),
    (3500, 4.5),
    (3500, 5.0),
    (29000, 3.5),
    (15000, 3.5),
    (5750, 3.0),
    (5250, 2.5),
    (4750, 2.0),
    (4000, 1.5),
    (3750, 1.5),
    (26000, 3.0),
    (14000, 2.5),
    (9750, 2.0),
    (8500, 2.0),
    (7750, 2.0),
    (7000, 1.5),
    (5500, 1.0),
    (4750, 1.0),
    (4500, 1.0),
    (3750, 0.0),
    (3750, 0.0),
    (3500, 0.0),
)


def castelli_kurucz_locus(metallicity: float = 0.0) -> list[QTable]:
    """Locus from AA_README, but with flexible metallicity"""
    return [get_castelli_kurucz_model(metallicity=metallicity, temperature=temperature, lg_g=lg_g)
            for temperature, lg_g in CASTELLI_KURUCZ_LOCUS_TEMP_LGG]
