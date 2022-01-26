from astropy.coordinates import SkyCoord

from calibration.data import narayan_etal2019_t1

def wd_coords():
    table = narayan_etal2019_t1()
    coords = SkyCoord(ra=table['R.A.'], dec=table['$\delta$'], unit=('hour', 'deg'))
    return coords


def main():
    print(wd_coords())


if __name__ == '__main__':
    main()