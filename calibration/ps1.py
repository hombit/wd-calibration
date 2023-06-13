# https://github.com/spacetelescope/notebooks/blob/master/notebooks/MAST/PanSTARRS/PS1_DR2_TAP/PS1_DR2_TAP.ipynb

from pathlib import Path
from typing import Literal, TypeVar, Union

import numpy as np
import pyvo
from astropy.coordinates import SkyCoord
from astropy.table import Table

from calibration.errors import NotFound, FoundMultiple


class Ps1:
    radius_deg = 1.0 / 3600.0
    bands = tuple('grizy')
    colors = {band: f'C{i}' for i, band in enumerate(bands)}

    __default_dr = 2

    @property
    def tap_url(self):
        return f'http://vao.stsci.edu/PS1DR{self.dr}/tapservice.aspx'

    def __init__(self, dr: int =__default_dr):
        self.dr = dr
        self.tap_service = pyvo.dal.TAPService(self.tap_url)

    def mean_objects(self, coord: SkyCoord) -> Table:
        result = self.tap_service.run_sync(f'''
            SELECT objID, RAMean, DecMean, nDetections, ng, nr, ni, nz, ny, gMeanPSFMag, gMeanPSFMagErr, rMeanPSFMag,
            rMeanPSFMagErr, iMeanPSFMag, iMeanPSFMagErr, zMeanPSFMag, zMeanPSFMagErr, yMeanPSFMag, yMeanPSFMagErr,
            gMeanApMag, gMeanApMagErr, rMeanApMag, rMeanApMagErr, iMeanApMag, iMeanApMagErr, zMeanApMag, zMeanApMagErr,
            yMeanApMag, yMeanApMagErr, gQfPerfect, rQfPerfect, iQfPerfect, zQfPerfect, yQfPerfect
            FROM dbo.MeanObjectView
            WHERE CONTAINS(
                    POINT('ICRS', RAMean, DecMean), CIRCLE('ICRS',{coord.ra.deg},{coord.dec.deg},{self.radius_deg})
                ) = 1
--                 AND ng >= 2 AND nr >= 2 AND ni >= 2 AND nz >= 2 AND ny >= 2
--                 AND gMeanApMag != -999 AND rMeanApMag != -999 AND iMeanApMag != -999 AND zMeanApMag != -999 AND yMeanApMag != -999
--                 AND gQfPerfect > 0.9 AND rQfPerfect > 0.9 AND iQfPerfect > 0.9 AND zQfPerfect > 0.9 AND yQfPerfect > 0.9
            ORDER BY nDetections DESC
        ''')
        table = result.to_table()
        return table

    def stacked_objects(self, coord: SkyCoord) -> Table:
        result = self.tap_service.run_sync(f'''
                    SELECT objID, raStack, decStack, nDetections, ng, nr, ni, nz, ny, gPSFMag, gPSFMagErr, rPSFMag,
                    rPSFMagErr, iPSFMag, iPSFMagErr, zPSFMag, zPSFMagErr, yPSFMag, yPSFMagErr, gApMag, gApMagErr,
                    rApMag, rApMagErr, iApMag, iApMagErr, zApMag, zApMagErr, yApMag, yApMagErr, gpsfQfPerfect,
                    rpsfQfPerfect, ipsfQfPerfect, zpsfQfPerfect, ypsfQfPerfect
                    FROM dbo.StackObjectView
                    WHERE CONTAINS(
                            POINT('ICRS', raStack, decStack), CIRCLE('ICRS',{coord.ra.deg},{coord.dec.deg},{self.radius_deg})
                        ) = 1 AND primaryDetection = 1
                    ORDER BY nDetections DESC
                ''')
        table = result.to_table()
        return table

    def light_curve(self, coord: SkyCoord) -> Table:
        mean_obj = self.mean_objects(coord)
        if len(mean_obj) == 0:
            raise NotFound
        if len(mean_obj) > 1:
            raise FoundMultiple
        mean_obj = mean_obj[0]

        result = self.tap_service.run_sync(f'''
            SELECT objID, detectID, imageID, skyCellID, DetectionObjectView.filterID as filterID, Filter.filterType,
                obsTime, ra, dec, psfFlux, psfFluxErr, psfMajorFWHM, psfMinorFWHM, psfQfPerfect, apFlux, apFluxErr,
                apFillF, apRadius, infoFlag, infoFlag2, infoFlag3
            FROM DetectionObjectView
            NATURAL JOIN Filter
            WHERE objID={mean_obj['objID']}
            ORDER BY obsTime, filterID
        ''')
        result = result.to_table()
        result.meta['mean'] = mean_obj
        return result

    _known_cuts = ('PHOTOM_PSF', 'PHOTOM_APER', 'good', 'ok', 'no')
    CUT = TypeVar('CUT', *(Literal[cut] for cut in _known_cuts))

    @staticmethod
    def cut_light_curve(lc: Table, cut: CUT) -> Table:
        def ok():
            return (lc['psfQfPerfect'] > 0.9) & (lc['apFlux'] > 0.0)

        match cut:
            case 'no':
                return lc.copy()
            case 'ok':
                idx = ok()
            case 'good':
                # https://outerspace.stsci.edu/display/PANSTARRS/PS1+Detection+Flags
                good_info_flags = 1
                bad_info_flags = (2 + 8 + 16 + 32 + 128 + 256 + 1024 + 2048 + 4096 + 65536 + 131072 + 536870912
                                  + 1073741824 + 2147483648)
                bad_info_flags2 = 8 + 16 + 32 + 64 + 512 + 1024 + 4194304
                bad_info_flags3 = 2 + 4 + 8 + 16 + 512 + 8192 + 16384 + 33554432
                idx = (
                    ok()
                    & (np.bitwise_and(lc['infoFlag'], good_info_flags) == good_info_flags)
                    & (np.bitwise_and(lc['infoFlag'], bad_info_flags) == 0)
                    & (np.bitwise_and(lc['infoFlag2'], bad_info_flags2) == 0)
                    & (np.bitwise_and(lc['infoFlag3'], bad_info_flags3) == 0)
                )
            case 'PHOTOM_APER':
                good_info_flags3 = 2097152
                idx = (np.bitwise_and(lc['infoFlag3'], good_info_flags3) == good_info_flags3)
            case 'PHOTOM_PSF':
                good_info_flags3 = 1048576
                idx = (np.bitwise_and(lc['infoFlag3'], good_info_flags3) == good_info_flags3)
        return lc[idx]

    _flux_markers = {'ap': 's', 'psf': 'o'}
    _known_flux_types = tuple(_flux_markers) + ('both',)
    FLUX = TypeVar('FLUX', *(Literal[flux] for flux in _known_flux_types))

    def plot(self, light_curve: Table, *, flux: FLUX = 'both', name: str, path: Union[str, Path], cut: CUT):
        import matplotlib.pyplot as plt

        if flux == 'both':
            flux = tuple(self._flux_markers)
        else:
            flux = (flux,)

        if ('ap' in flux and cut == 'PHOTOM_PSF') or ('psf' in flux and cut == 'PHOTOM_APER'):
            raise ValueError(f'{flux = } {cut = }')

        path = Path(path)
        if path.is_dir():
            path = path.joinpath(f'{name}.pdf')
        path.parent.mkdir(exist_ok=True)

        lc = self.cut_light_curve(light_curve, cut)

        def plot_flux(flux: str) -> None:
            flux_column = f'{flux}Flux'
            flux_err_column = f'{flux_column}Err'
            for band in self.colors:
                idx = lc['filterType'] == band

                n_lc = np.sum(idx)
                if n_lc == 0:
                    continue

                time = lc['obsTime'][idx]
                fl = lc[flux_column][idx]
                fl_err = lc[flux_err_column][idx]

                mag = -2.5 * np.log10(fl / 3631)
                mag_err = 2.5 / np.log(10) * fl_err / lc[flux_column][idx]

                # We follow the DAWD LCO paper and compute chi2 in mags
                mag_weights = mag_err**-2
                mean_mag = np.average(mag, weights=mag_weights)
                chi2 = np.sum(np.square(mag - mean_mag) * mag_weights)

                plt.errorbar(time, mag, mag_err, ls='', marker=self._flux_markers[flux], markerfacecolor='none',
                             label=rf'{flux} {band}, $\chi^2/\mathrm{{dof}} = {chi2:.1f}/{n_lc - 1} = {chi2/(n_lc-1):.2f}$')

        plt.figure(constrained_layout=True)
        plt.xlabel('MJD')
        plt.ylabel(f'mag')
        plt.gca().invert_yaxis()
        plt.title(name)
        for fl in flux:
            plot_flux(fl)
        plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.4), ncol=2)
        plt.savefig(path)
        plt.close()
