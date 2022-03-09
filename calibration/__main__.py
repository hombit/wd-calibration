from pathlib import Path

from astropy.table import Table

from calibration.catwise import catwise2020
from calibration.gaia import gaia_edr3
from calibration.kepler_tess import kepler_k2, tess
from calibration.ps1 import Ps1
from calibration.util import wd_coords


def main():
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)

    names, coords = wd_coords(with_bright=False)

    catwise = catwise2020(coords)
    catwise.write(results_dir.joinpath('catwise2020.csv'), overwrite=True)

    ps1 = Ps1()
    ps1_mean = []
    results_ps1 = results_dir.joinpath('ps1')
    results_ps1.mkdir(exist_ok=True)
    for name, coord in zip(names, coords):
        ps1_lc = ps1.light_curve(coord)
        ps1_lc.write(results_ps1.joinpath(f'{name}.csv'), overwrite=True)
        ps1_mean.append(ps1_lc.meta['mean'])
        ps1.plot(ps1_lc, flux='psf', name=name, path='./figures', cut='PHOTOM_PSF')
    ps1_mean = Table(rows=ps1_mean, names=ps1_mean[0].columns)
    ps1_mean.write(results_ps1.joinpath('mean.csv'), overwrite=True)

    gaia_table = gaia_edr3(coords)
    gaia_table.write(results_dir.joinpath('gaiaedr3.csv'), overwrite=True)

    print(kepler_k2(coords[-2]))  # the only object found in 5"

    # for coord in coords:
    #     print(tess(coord))


if __name__ == '__main__':
    main()