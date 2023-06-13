from functools import lru_cache
from importlib.resources import open_binary

import numpy as np
from astropy import units
from astropy.io import ascii
from astropy.table import QTable
from scipy.interpolate import UnivariateSpline

from calibration.data import filters


__all__ = 'get_passband',


# 2022.03.09
FILTER_A_AV = {
    'CTIO/DECam.u': 1.49,
    'CTIO/DECam.g': 1.20,
    'CTIO/DECam.r': 0.842,
    'CTIO/DECam.i': 0.636,
    'CTIO/DECam.z': 0.495,
    'CTIO/DECam.Y': 0.436,
    'GAIA/GAIA3.Gbp': 1.10,
    'GAIA/GAIA3.G': 0.87,
    'GAIA/GAIA3.Grp': 0.636,
    'Palomar/ZTF.g': 1.21,
    'Palomar/ZTF.r': 0.848,
    'Palomar/ZTF.i': 0.622,
    'PAN-STARRS/PS1.g': 1.18,
    'PAN-STARRS/PS1.r': 0.881,
    'PAN-STARRS/PS1.i': 0.667,
    'PAN-STARRS/PS1.z': 0.534,
    'PAN-STARRS/PS1.y': 0.457,
    'SLOAN/SDSS.u': 1.58,
    'SLOAN/SDSS.g': 1.22,
    'SLOAN/SDSS.r': 0.884,
    'SLOAN/SDSS.i': 0.673,
    'SLOAN/SDSS.z': 0.514,
    'WISE/WISE.W1': 0.0726,
    'WISE/WISE.W2': 0.0509,
    'WISE/WISE.W3': 0.0572,
    'WISE/WISE.W4': 0.02,
}

FILTER_DETECTOR = {
    'CTIO/DECam.u': 'photon',
    'CTIO/DECam.g': 'photon',
    'CTIO/DECam.r': 'photon',
    'CTIO/DECam.i': 'photon',
    'CTIO/DECam.z': 'photon',
    'CTIO/DECam.Y': 'photon',
    'GAIA/GAIA3.Gbp': 'photon',
    'GAIA/GAIA3.G': 'photon',
    'GAIA/GAIA3.Grp': 'photon',
    'Palomar/ZTF.g': 'energy',
    'Palomar/ZTF.r': 'energy',
    'Palomar/ZTF.i': 'energy',
    'PAN-STARRS/PS1.g': 'photon',
    'PAN-STARRS/PS1.r': 'photon',
    'PAN-STARRS/PS1.i': 'photon',
    'PAN-STARRS/PS1.z': 'photon',
    'PAN-STARRS/PS1.y': 'photon',
    'SLOAN/SDSS.u': 'photon',
    'SLOAN/SDSS.g': 'photon',
    'SLOAN/SDSS.r': 'photon',
    'SLOAN/SDSS.i': 'photon',
    'SLOAN/SDSS.z': 'photon',
    'WISE/WISE.W1': 'energy',
    'WISE/WISE.W2': 'energy',
    'WISE/WISE.W3': 'energy',
    'WISE/WISE.W4': 'energy',
}

assert set(FILTER_A_AV) == set(FILTER_DETECTOR)
assert set(FILTER_DETECTOR.values()).issubset({'photon', 'energy'})


@lru_cache
def get_passband(name: str) -> QTable:
    """Passband transmission from Filter Profile Service name

    Returns
    -------
    QTable
        A table with two columns: 'wavelength' in angstroms and 'transmission'
        The `meta` attribute contains a dictionary with:
            - 'spline': function returns interpolated transmission value,
                the argument is a dimensional wavelength
            - 'A_AV': extinction
            - 'detector': detector type, one of "photon", "energy"
    """
    fname = name.replace('/', '_')
    with open_binary(filters, f'{fname}.dat') as fh:
        table = ascii.read(
            fh,
            format='basic',
            delimiter=' ',
            names=['wavelength', 'transmission'],
            converters={'*': [ascii.convert_numpy(np.float64)]},  # we don't need integer wavelengths
        )

    table = QTable(table, copy=False)
    table['wavelength'].unit = units.angstrom

    spline = UnivariateSpline(x=table['wavelength'], y=table['transmission'], k=3, ext='zeros')
    table.meta['spline'] = lambda lmbd: spline(lmbd.to_value(units.angstrom))

    table.meta['A_AV'] = FILTER_A_AV[name]
    table.meta['detector'] = FILTER_DETECTOR[name]

    return table