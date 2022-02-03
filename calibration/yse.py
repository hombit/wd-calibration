import os
from dataclasses import dataclass

import requests
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
from regions import RectangleSkyRegion

from calibration.util import wd_coords


YSE_PZ_USERNAME = os.environ.get('YSE_PZ_USERNAME', 'kmalanchev')
try:
    YSE_PZ_PASSWORD = os.environ['YSE_PZ_PASSWORD']
except KeyError as e:
    raise ValueError('Please specify environment variables YSE_PZ_USERNAME and YSE_PZ_PASSWORD') from e


@dataclass
class Field:
    region: RectangleSkyRegion

    def __hash__(self):
        return hash(self.region.meta["id"])


def main():
    data = requests.get(
        "https://ziggy.ucolick.org/yse/api/surveyfieldmsbs",
        auth=requests.auth.HTTPBasicAuth(YSE_PZ_USERNAME, YSE_PZ_PASSWORD),
    ).json()

    fields = set()

    for result in data["results"]:
        for field in result["survey_fields"]:
            wcs = WCS(
                dict(
                    CTYPE1="RA---TAN",
                    CTYPE2="DEC--TAN",
                    CRPIX1=0,
                    CRPIX2=0,
                    CRVAL1=field["ra_cen"],
                    CRVAL2=field["dec_cen"],
                    CUNIT1="deg",
                    CUNIT2="deg",
                    CD1_1=1,
                    CD1_2=0,
                    CD2_1=0,
                    CD2_2=1,
                )
            )
            region = RectangleSkyRegion(
                center=SkyCoord(ra=field["ra_cen"], dec=field["dec_cen"], unit="deg"),
                width=Angle(field["width_deg"], "deg"),
                height=Angle(field["height_deg"], "deg"),
                meta=dict(id=field["field_id"], wcs=wcs),
            )
            fields.add(Field(region))

    wds = wd_coords()
    for name, wd in zip(wds.info.name, wds):
        for field in fields:
            region = field.region
            if region.contains(wd, region.meta["wcs"]).item():
                print(name, region.meta["id"], region.center)
