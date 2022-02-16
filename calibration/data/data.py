from functools import lru_cache, partial
from importlib.resources import open_binary

import numpy as np
from astropy import units
from astropy.io import ascii
from astropy.table import Table
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline

from calibration import data
from calibration.data import filters


__all__ = 'narayan_etal2019_t1', 'passband'


@lru_cache(1)
def narayan_etal2019_t1() -> Table:
    """Slightly modified Table 1 from Narayan et al. 2019

    https://ui.adsabs.harvard.edu/abs/2019ApJS..241...20N/abstract
    """
    with open_binary(data, 'WDphot_ILAPHv5_abmag.tex') as fh:
        table = ascii.read(fh, format='latex')
    return table


@lru_cache
def passband(name: str) -> Table:
    """Passband transmission from Filter Profile Service name

    Returns
    -------
    Table
        A table with two columns: 'wavelength' in angstroms and 'transmission'
        The `meta` attribute contains a dictionary with:
            - 'simpson_integral': transmission integral in angstroms
            - 'spline': function returns interpolated transmission value,
                the argument is a dimensional wavelength
            - 'spline_integral': transmission integral in angstroms
    """
    name = name.replace('/', '_')
    with open_binary(filters, f'{name}.dat') as fh:
        table = ascii.read(
            fh,
            format='basic',
            delimiter=' ',
            names=['wavelength', 'transmission'],
            converters={'*': [ascii.convert_numpy(np.float)]},  # we don't need integer wavelengths
        )

    table.meta['simpson_integral'] = simpson(x=table['wavelength'], y=table['transmission']) * units.angstrom
    spline = UnivariateSpline(x=table['wavelength'], y=table['transmission'], k=3, ext='zeros')
    table.meta['spline'] = lambda lmbd: spline(lmbd.to_value(units.angstrom))
    # Spline integral could be a bit different from simpson integral
    table.meta['spline_integral'] = (
            spline.integral(table['wavelength'].min(), table['wavelength'].max()) * units.angstrom
    )

    table['wavelength'].unit = units.angstrom

    return table
