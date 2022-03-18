from typing import Union

import numpy as np
from astropy import constants, units
from astropy.table import QTable
from scipy.integrate import simpson

from calibration.const import F_AB
from calibration.data import get_spectrum, get_passband
from calibration.units import CGS_F_LMBD


__all__ = ('syn_mag',)


def syn_mag(sed: Union[str, QTable], band: str) -> float:
    """Integrate a spectrum with a passband transmission

    Parameters
    ----------
    specname : str or QTable
        SED in for of `QTable` w/ dimensional columns "wave" and "flux",
        or a spectroscopy file name exists in `calibration.data.spectroscopy`
    band : str
        Passband name exists in `calibration.data.band`

    Returns
    -------
    flux : float
    flux_err : float
    """

    fltr = get_passband(band)
    fltr_spline = fltr.meta['spline']
    fltr_lmbd_cm = fltr['wavelength'].to_value(units.cm)

    if isinstance(sed, str):
        sed = get_spectrum(sed)

    lmbd = sed['wave']
    lmbd_cm = lmbd.to_value(units.cm)

    flmbd_cgs = sed['flux'].to_value(CGS_F_LMBD)

    match fltr.meta['detector']:
        case 'photon':
            flux_integral = simpson(x=lmbd_cm, y=lmbd_cm * fltr_spline(lmbd) * flmbd_cgs) * CGS_F_LMBD * units.cm**2
            FAB_integral = F_AB * constants.c * simpson(x=fltr_lmbd_cm, y=fltr['transmission'] / fltr_lmbd_cm)
        case 'energy':
            flux_integral = simpson(x=lmbd_cm, y=fltr_spline(lmbd) * flmbd_cgs) * CGS_F_LMBD * units.cm**2
            FAB_integral = F_AB * constants.c * simpson(x=fltr_lmbd_cm, y=fltr['transmission'] / fltr_lmbd_cm**2)
        case detector:
            raise ValueError(f'detector type "{detector}" unknown')

    mag = -2.5 * np.log10(flux_integral / FAB_integral)

    return mag.to_value(units.dimensionless_unscaled)
