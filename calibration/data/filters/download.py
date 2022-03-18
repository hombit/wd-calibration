import logging
from pathlib import Path

import requests


FILTER_IDS = (
    'GAIA/GAIA3.Gbp',
    'GAIA/GAIA3.G',
    'GAIA/GAIA3.Grp',
    'Palomar/ZTF.g',
    'Palomar/ZTF.r',
    'Palomar/ZTF.i',
    'PAN-STARRS/PS1.g',
    'PAN-STARRS/PS1.r',
    'PAN-STARRS/PS1.i',
    'PAN-STARRS/PS1.z',
    'PAN-STARRS/PS1.y',
    'SLOAN/SDSS.u',
    'SLOAN/SDSS.g',
    'SLOAN/SDSS.r',
    'SLOAN/SDSS.i',
    'SLOAN/SDSS.z',
    'WISE/WISE.W1',
    'WISE/WISE.W2',
    'WISE/WISE.W3',
    'WISE/WISE.W4',
)


BASE_URL = 'http://svo2.cab.inta-csic.es/theory/fps3/getdata.php'


def download_filters() -> None:
    """Update data-files inside calibration.data.filters

    Data from Filter Profile Service
    """
    directory = Path(__file__).parent

    assert len(FILTER_IDS) == len(set(FILTER_IDS)), 'FILTER_IDS has duplicates'

    for filter_id in FILTER_IDS:
        logging.info(f'Downloading {filter_id}')

        # Mimic FPS ASCII file names
        filter_name = filter_id.replace('/', '_')
        path = directory.joinpath(f'{filter_name}.dat')

        with requests.get(BASE_URL, params=dict(format='ascii', id=filter_id), stream=True) as response:
            response.raise_for_status()
            with open(path, 'wb') as fh:
                for chunk in response.iter_content(chunk_size=1 << 13):
                    fh.write(chunk)