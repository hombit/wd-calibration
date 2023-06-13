#!/usr/bin/env python3

from itertools import product

from calibration.des import Des


def main():
    des = Des()
    bands = 'griz'  # we skip y-band because it is not precise
    constraines = [
        ('wavg_mag_psf', '> 15'),
        ('wavg_mag_psf', '< 26'),
        ('wavg_magerr_psf', '< 0.02'),
        ('flags', '<= 1'),
        ('spread_model', '< 0.01'),
    ]
    where = ' AND '.join(f'{op}_{band} {value}' for band, (op, value) in product(bands, constraines))
    query = f"""
    SELECT * FROM {des.table_name}
    WHERE ebv_sfd98 < 0.02
        AND {where}
    """
    print(query)
    # des.query_and_cache(query, directory='.', prefix='des_constrained', compress=True)


if __name__ == '__main__':
    main()
