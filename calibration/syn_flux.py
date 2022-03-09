import numpy as np
from astropy import units
from scipy.integrate import simpson

from calibration.data import get_spectrum, get_passband


def syn_flux(specname: str, band: str) -> tuple[float, float]:
    """Integrate a spectrum with a passband transmission

    Parameters
    ----------
    specname : str
        Spectroscopy file name exists in `calibration.data.spectroscopy`
    band : str
        Passband name exists in `calibration.data.band`

    Returns
    -------
    flux : float
    flux_err : float
    """

    fltr = get_passband(band)
    fltr_spline = fltr.meta['spline']
    fltr_integral_a = fltr.meta['spline_integral'].to_value(units.angstrom)

    sed = get_spectrum(specname)

    lmbd = sed['wave']
    lmbd_a = lmbd.to_value(units.angstrom)
    flux = simpson(x=lmbd_a, y=lmbd_a * fltr_spline(lmbd) * sed['flux']) / fltr_integral_a
    flux_err = np.sqrt(simpson(x=lmbd_a, y=lmbd_a * fltr_spline(lmbd) * np.square(sed['flux_err'])) / fltr_integral_a)

    return flux, flux_err
