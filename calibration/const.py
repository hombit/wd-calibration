from astropy import units

from calibration.units import CGS_F_NU

__all__ = ('F_AB',)


F_AB = (0 * units.ABmag).to(CGS_F_NU)