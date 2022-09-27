import sys
from pathlib import Path
from subprocess import check_call
from typing import Iterable

from astropy.coordinates import SkyCoord

from calibration.data import calamida_lco_t2
from calibration.errors import NotFound
from calibration.ps1 import Ps1
from calibration.ztf import ZtfDr


def ps1(names: Iterable[str], coords: Iterable[SkyCoord]):
    fig_path = Path('./calamida_lco_ps1')
    fig_path.mkdir(exist_ok=True)

    ps1 = Ps1()
    for name, coord in zip(names, coords):
        try:
            ps1_lc = ps1.light_curve(coord)
        except NotFound:
            print(f'No PS1 matches fof {name} {coord}', file=sys.stdout)
            continue
        ps1.plot(ps1_lc, flux='psf', name=name, path=fig_path, cut='good')


def ztf(names: Iterable[str], coords: Iterable[SkyCoord]):
    fig_path = Path('./calamida_lco_ztf')
    fig_path.mkdir(exist_ok=True)

    ztf = ZtfDr()
    for name, coord in zip(names, coords):
        try:
            lc = ztf.light_curve(coord)
        except NotFound:
            print(f'No ZTF DR matches fof {name} {coord}', file=sys.stdout)
            continue
        ztf.plot(lc, name=name, path=fig_path)


def casjobs(names, coords, exec=False):
    for name, coord in zip(names, coords):
        table_name, *_ = name.split('.')
        query = f'''
select sov.objID, sov.RAMean, sov.DecMean, sov.nDetections, sov.ng, sov.nr, sov.ni, sov.nz, sov.ny,
                sov.gPSFMag, sov.gPSFMagErr, sov.rPSFMag, sov.rPSFMagErr, sov.iPSFMag, sov.iPSFMagErr,
                sov.zPSFMag, sov.zPSFMagErr, sov.yPSFMag, sov.yPSFMagErr,
                sov.gApMag, sov.gApMagErr, sov.rApMag, sov.rApMagErr, sov.iApMag, sov.iApMagErr,
                sov.zApMag, sov.zApMagErr, sov.yApMag, sov.yApMagErr,
                sov.gpsfQf, sov.rpsfQf, sov.ipsfQf, sov.zpsfQf, sov.ypsfQf,
                sov.gpsfQfPerfect, sov.rpsfQfPerfect, sov.ipsfQfPerfect, sov.zpsfQfPerfect, sov.ypsfQfPerfect,
                sov.objInfoFlag, sov.qualityFlag
into mydb.{table_name}
from fGetNearbyObjEq({coord.ra.deg},{coord.dec.deg},60.0) nb
inner join StackObjectView sov on sov.objid=nb.objid
WHERE (sov.qualityFlag & 0x00000010 != 0) AND (sov.primaryDetection = 1)'''
        print(query)
        if exec:
            check_call([
                'java',
                '-jar', 'casjobs.jar',
                'submit',
                '-n', table_name,
                query,
            ])


def main():
    table = calamida_lco_t2()
    names = table['Orig. name']
    coords = SkyCoord(ra=table['RA'], dec=table['DEC'], unit=('hour', 'deg'))

    casjobs(names, coords, exec=True)

    ps1(names, coords)
    ztf(names, coords)


if __name__ == '__main__':
    main()