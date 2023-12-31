import numpy as np
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table
from astroquery.vizier import Vizier


__all__ = 'gaia_edr3',


def gaia_edr3(coord: SkyCoord) -> Table:
    """

    https://ui.adsabs.harvard.edu/abs/2021A%26A...649A...3R/abstract
    https://ui.adsabs.harvard.edu/abs/2021A%26A...649A...4L/abstract
    """

    # Note (G2)  : Note on Calibration corrected values.
    #
    # G-band magnitude correction for sources with 6-parameter astrometric solutions.
    #
    # The paper Gaia Early Data Release 3: Photometric content and validation by
    # Riello et al. (2020) explains that for sources with 6-parameter astrometric solutions the G-band magnitude should be corrected and a formula to do so is provided. The corresponding Python code to do this is presented in Gaia Early Data Release 3: Summary of the contents and survey properties (Gaia Collaboration et al., 2020). The source code can be found as a Jupyter notebook in this repository:
    #
    # https://github.com/agabrown/gaiaedr3-6p-gband-correction
    #
    # Corrected flux excess factor.
    #
    # The paper Gaia Early Data Release 3: Photometric content and validation by
    # Riello et al. (2020) presents a corrected version of the photometric flux excess factor as published in the Gaia EDR3 catalogue. The corrected version acounts for the average variation of the flux excess for 'normal' sources. A formula for calculating the corrected excess factor is provided. The corresponding Python code to do this is presented in Gaia Early Data Release 3: Summary of the contents and survey properties (Gaia Collaboration et al., 2020). The source code can be found as a Jupyter notebook in this repository:
    #
    # https://github.com/agabrown/gaiaedr3-flux-excess-correction
    #
    # See also:
    # https://www.cosmos.esa.int/web/gaia/edr3-code

    vizier = Vizier(
        catalog='I/350/gaiaedr3',
        columns=['_q', '_r', '_RAJ2000', '_DEJ2000', 'Source', 'PM', 'pmRA', 'pmDE', 'RUWE', 'Dup', 'Mode', 'FG',
                 'e_FG', 'Gmag', 'e_Gmag', 'o_Gmag', 'FBP', 'e_FBP', 'BPmag', 'e_BPmag', 'o_BPmag', 'FRP', 'e_FRP',
                 'RPmag', 'e_RPmag', 'p_RPmag', 'PS1', 'SDSSDR13', 'SkyMapper2'],
    )
    table, = vizier.query_region(coord, radius=Angle(5, 'arcsec'))
    np.testing.assert_array_equal(table['_q'], np.arange(1, coord.size + 1), "Vizier cross-matches are not one-to-one")
    assert np.all(table['RUWE'] < 1.4),\
           "Some of the returned GAIA sources do not have good astrometric solution: RUWE >= 1.4"
    np.testing.assert_array_equal(table['Dup'], 0, "Some of returned GAIA sources have Dup != 0")
    np.testing.assert_array_equal(table['Mode'], 0, "Some of returned GAIA sources have Mode != 0")
    return table