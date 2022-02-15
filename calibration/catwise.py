from astropy.coordinates import Angle, SkyCoord
from astropy.table import Row, Table
from astroquery.ipac.irsa import Irsa


def catwise2020(coords: SkyCoord) -> Table:
    radius = Angle(4.0, unit='arcsec')
    rows = []
    for coord in coords:
        result = Irsa.query_region(coord, catalog='catwise_2020', radius=radius)
        if len(result) > 0:
            rows.append(result[0])
            columns = result.columns
        else:
            rows.append(None)
    rows = [[None] * len(columns) if row is None else row for row in rows]
    table = Table(rows=rows, names=columns)
    return table
