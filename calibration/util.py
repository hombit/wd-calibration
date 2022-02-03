from astropy.coordinates import SkyCoord

from calibration.data import narayan_etal2019_t1


__all__ = 'wd_coords',


def wd_coords():
    table = narayan_etal2019_t1()
    coords = SkyCoord(ra=table['R.A.'], dec=table['$\delta$'], unit=('hour', 'deg'))
    coords.info.name = table['Object']
    return coords