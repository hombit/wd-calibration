import os
from dataclasses import dataclass
from typing import Collection, Optional

import requests
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
from regions import RectangleSkyRegion

from calibration.util import wd_coords


YSE_PZ_USERNAME = os.environ.get('YSE_PZ_USERNAME', 'kmalanchev')
YSE_PZ_PASSWORD = os.environ.get('YSE_PZ_PASSWORD', None)


@dataclass
class YseField:
    region: RectangleSkyRegion

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f'YseField({self.id} center={self.center})'

    @property
    def center(self) -> SkyCoord:
        return self.region.center

    @property
    def id(self) -> str:
        return self.region.meta['id']

    @property
    def wcs(self) -> WCS:
        return self.region.meta['wcs']

    def contains(self, coord: SkyCoord) -> bool:
        if coord.size != 1:
            raise ValueError('Scalar SkyCoord supported only')
        return self.region.contains(coord, self.wcs).item()


@dataclass
class YseFields:
    fields: Collection[YseField]

    def contains(self, coord: SkyCoord) -> list[YseField]:
        return [field for field in self.fields if field.contains(coord)]

    @classmethod
    def from_yse_pz(cls, username: str = YSE_PZ_USERNAME, password: Optional[str] = YSE_PZ_PASSWORD):
        if password is None:
            ValueError('Please specify environment variables YSE_PZ_USERNAME and YSE_PZ_PASSWORD')
        response = requests.get(
            "https://ziggy.ucolick.org/yse/api/surveyfieldmsbs",
            auth=requests.auth.HTTPBasicAuth(username, password),
        )
        response.raise_for_status()
        data = response.json()

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
                fields.add(YseField(region))

        return cls(fields)


def main():
    fields = YseFields.from_yse_pz()

    wds = wd_coords()
    for name, wd in zip(wds.info.name, wds):
        for field in fields.contains(wd):
            print(name, field.id, field.center.ra.deg, field.center.dec.deg)


if __name__ == '__main__':
    main()
