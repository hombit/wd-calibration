from pathlib import Path
from typing import Union
from urllib.parse import urljoin

import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.table import Table

from calibration.errors import NotFound


class ZtfDr:
    base_url = "http://db.ztf.snad.space/api/v3/data/"
    dr = 'dr8'

    @property
    def url(self):
        return urljoin(self.base_url, f'{self.dr}/circle/full/json')

    radius_arcsec = 1.0
    bands = ('zg', 'zr')
    colors = {band: f'C{i}' for i, band in enumerate(bands)}
    meta_columns = ('filter',)

    def __init__(self):
        self.session = requests.Session

    def cone_search(self, coord: SkyCoord) -> dict:
        with requests.get(self.url, params=dict(ra=coord.ra.deg, dec=coord.dec.deg, radius_arcsec=self.radius_arcsec)) as response:
            response.raise_for_status()
            data = response.json()
        if len(data) == 0:
            raise NotFound
        return data

    def light_curve(self, coord: SkyCoord) -> Table:
        data = self.cone_search(coord)
        lc = []
        for obj_id, obj in data.items():
            meta = {column: obj['meta'][column] for column in self.meta_columns}
            for obs in obj['lc']:
                obs = obs | meta
                lc.append(obs)
        table = Table(rows=lc)
        return table

    def plot(self, light_curve: Table, *, name: str, path: Union[str, Path]):
        import matplotlib.pyplot as plt

        path = Path(path)
        if path.is_dir():
            path = path.joinpath(f'{name}.pdf')
        path.parent.mkdir(exist_ok=True)

        plt.figure(constrained_layout=True)
        plt.xlabel('MJD')
        plt.ylabel(f'mag')
        plt.gca().invert_yaxis()
        plt.title(name)

        for band in self.colors:
            idx = light_curve['filter'] == band

            n_lc = np.sum(idx)
            if n_lc == 0:
                continue

            time = light_curve['mjd'][idx]
            mag = light_curve['mag'][idx]
            magerr = light_curve['magerr'][idx]

            weights = magerr**-2
            mean_mag = np.average(mag, weights=weights)
            chi2 = np.sum(np.square(mag - mean_mag) * weights)

            plt.errorbar(time, mag, magerr, ls='', marker='x', markerfacecolor='none',
                         label=rf'{band}, $\chi^2/\mathrm{{dof}} = {chi2:.1f}/{n_lc - 1}$')

        plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.4), ncol=2)
        plt.savefig(path)
        plt.close()
