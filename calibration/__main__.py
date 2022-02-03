from calibration.gaia import gaia_edr3
from calibration.kepler_tess import kepler_k2, tess
from calibration.util import wd_coords


def main():
    coords = wd_coords()

    gaia_table = gaia_edr3(coords)
    gaia_table.pprint_all()
    gaia_table.write('gaiaedr3.csv', overwrite=True)

    print(kepler_k2(coords[-2]))  # the only object found in 5"

    # for coord in coords:
    #     print(tess(coord))
    #     break


if __name__ == '__main__':
    main()