from functools import lru_cache
from importlib.resources import contents, open_binary

import h5py
from astropy import units
from astropy.table import Table, QTable

from calibration.data import spectroscopy


__all__ = 'get_spectrum',


@lru_cache(1)
def sed_files() -> dict[str, str]:
    names_to_files = dict()
    for filename in contents(spectroscopy):
        if not filename.endswith('.hdf5'):
            continue
        name, _ = filename.split('-', maxsplit=1)
        names_to_files[name] = filename
    return names_to_files


@lru_cache
def get_spectrum(name: str) -> QTable:
    filename = sed_files()[name]
    with open_binary(spectroscopy, filename) as fh, h5py.File(fh) as file:
        dataset = file['model']
        table = Table(dataset, copy=True)
    table = QTable(table)
    table['wave'].unit = units.angstrom
    raise NotImplemented
    table['flux'].unit = table['flux_err'].unit = ...
    return table
