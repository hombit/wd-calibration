from astropy import units


__all__ = ('CGS_F_NU', 'CGS_F_LMBD', 'FLAM')


CGS_F_LMBD = units.erg / units.cm**2 / units.s / units.cm
FLAM = units.erg / units.cm**2 / units.s / units.angstrom
CGS_F_NU = units.erg / units.cm**2 / units.s / units.Hz