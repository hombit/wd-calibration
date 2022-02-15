# https://github.com/spacetelescope/notebooks/blob/master/notebooks/MAST/PanSTARRS/PS1_DR2_TAP/PS1_DR2_TAP.ipynb

import pyvo
from astropy.coordinates import SkyCoord
from astropy.table import Table


class Ps1:
    tap_url = "http://vao.stsci.edu/PS1DR2/tapservice.aspx"
    radius_deg = 3.0 / 3600.0
    bands = tuple('grizy')

    def __init__(self):
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

    def light_curve(self, coord: SkyCoord) -> Table:
        mean_obj = self.mean_objects(coord)[0]
        result = self.tap_service.run_sync(f'''
            SELECT objID, detectID, imageID, Detection.filterID as filterID, Filter.filterType, obsTime, ra, dec,
                psfFlux, psfFluxErr, psfMajorFWHM, psfMinorFWHM, psfQfPerfect, apFlux, apFluxErr, infoFlag, infoFlag2,
                infoFlag3
            FROM Detection
            NATURAL JOIN Filter
            WHERE objID={mean_obj['objID']}
            ORDER BY obsTime, filterID
        ''')
        result = result.to_table()
        result.meta['mean'] = mean_obj
        return result
