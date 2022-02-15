from typing import Optional, Literal

import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord, Angle
from lightkurve import search_targetpixelfile, KeplerLightCurve


__all__ = 'kepler_k2', 'tess',


def kepler_tess(mission: Literal['kepler', 'K2', 'tess'], coord: SkyCoord,
                *, cache_path: Optional[str] = None, plot: bool = False) -> Optional[npt.NDArray]:
    search_result = search_targetpixelfile(coord, mission=mission, radius=Angle(5, 'arcsec'))
    if not search_result:
        return None
    pixels = search_result.download_all(download_dir=cache_path, quality_bitmask='hardest')

    flux = []
    for pixel in pixels:
        # aperture or PRF or PLD?
        # PRF is marked experimental
        lc = pixel.to_lightcurve(method='aperture', aperture_mask='pipeline')
        # lc = pixel.to_lightcurve(method='prf')
        # lc = pixel.to_lightcurve(method='pld')

        flux.append(lc.flux)

        if plot:
            import matplotlib.pyplot as plt

            flatten_lc, trend_lc = lc.flatten(return_trend=True)
            flatten_lc = flatten_lc.remove_outliers()
            plt.figure()
            plt.xlabel('MJD')
            plt.ylabel(f'{mission} flux')
            plt.errorbar(flatten_lc.time.mjd, flatten_lc.flux.data, flatten_lc.flux_err.data, ls='')
            # plt.errorbar(lc.time.mjd[:100], lc.flux.data[:100], lc.flux_err.data[:100], ls='')
            # periodogram = lc.normalize().to_periodogram()
            # plt.plot(periodogram.frequency, periodogram.power)
            # print(periodogram.period_at_max_power)
            plt.show()

    flux = np.concatenate(flux)

    return flux


# Do we need K1? Probably no, no object is found there
def kepler_k2(coord: SkyCoord, *, cache_path: Optional[str] = None, plot: bool = False) -> Optional[npt.NDArray]:
    return kepler_tess('K2', coord, cache_path=cache_path, plot=plot)


def tess(coord: SkyCoord, *, cache_path: Optional[str] = None, plot: bool = False) -> Optional[npt.NDArray]:
    return kepler_tess('tess', coord, cache_path=cache_path, plot=plot)
