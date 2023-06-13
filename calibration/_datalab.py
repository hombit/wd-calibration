import re
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
from astropy.coordinates import SkyCoord
from dl import queryClient  # astro-datalab package


class DataLab:
    def __init__(self, table_name: str):
        self.table_name = table_name

    def cone_search_raw(self, coord: SkyCoord, *, radius_arcsec) -> str:
        return queryClient.query(sql=f"""
            SELECT *
            FROM {self.table_name}
            WHERE
                q3c_radial_query(ra, dec, {coord.ra.deg}, {coord.dec.deg}, {radius_arcsec / 3600.0})
        """)

    def cone_search(self, coord: SkyCoord, *, radius_arcsec) -> pd.DataFrame:
        data = self.cone_search_raw(coord, radius_arcsec=radius_arcsec)
        byte_stream = BytesIO(data.encode())
        byte_stream.seek(0)
        return pd.read_csv(byte_stream)

    @staticmethod
    def query_and_cache(query, *, directory=".", prefix="query", compress=False):
        query = re.sub(r"\s+", " ", query.strip())
        query_hash = sha256(query.encode()).hexdigest()

        data_suffix = f"_{query_hash}.csv"
        if compress:
            data_suffix += ".bz2"
        data_path = Path(directory) / f"{prefix}{data_suffix}"
        if data_path.exists():
            return pd.read_csv(data_path)

        query_suffix = f"_{query_hash}.sql"
        query_path = Path(directory) / f"{prefix}{query_suffix}"
        with open(query_path, "w") as fh:
            fh.write(query)

        data = queryClient.query(sql=query, timeout=3600)
        with open(data_path, "rw") as fh:
            fh.write(data)
        byte_stream = BytesIO(data.encode())
        byte_stream.seek(0)
        return pd.read_csv(byte_stream)
