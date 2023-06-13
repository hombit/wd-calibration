from pathlib import Path

from astropy.coordinates import SkyCoord

from calibration.data import calamida_lco_t2
from calibration.des import Des


def main():
    objects = calamida_lco_t2()
    result_dir = Path('./des_1deg_cone')
    result_dir.mkdir(exist_ok=True)
    des = Des(dr=2)
    for name, ra, dec in objects[['Star', 'RA', 'DEC']]:
        print(name)
        coord = SkyCoord(ra, dec, unit='deg')
        table_str = des.cone_search_raw(coord, radius_arcsec=3600)
        with open(result_dir / f'{name}.csv', 'w') as fh:
            fh.write(table_str)


if __name__ == '__main__':
    main()
