#!/usr/bin/env python3

from functools import reduce
from pathlib import Path

import polars as pl


def cast_by_prefix(df: pl.DataFrame, to: pl.datatypes.DataType = pl.Float32, prefix: str = 'des_mag') -> pl.DataFrame:
    return df.with_columns(
        [pl.col(column).cast(to).alias(column) for column in df.columns if column.startswith(prefix)],
    )


def main():
    folder = Path('data')

    ps1_x_des = pl.read_parquet(folder / 'ps1_des-grizy.parquet')
    ps1_x_des = cast_by_prefix(ps1_x_des)

    des = pl.read_parquet(folder / 'des_stars.parquet')
    des = des.join(ps1_x_des, on='des_id', how='anti')
    des = cast_by_prefix(des)
    des = des.with_columns(
        [pl.lit(None, dtype).alias(column) for column, dtype in ps1_x_des.schema.items() if column not in des.columns],
    )

    ps1 = pl.read_parquet(folder / 'ps1_stars.parquet')
    ps1 = ps1.join(ps1_x_des, on='ps1_id', how='anti')
    ps1_transformed_to_des = [pl.read_parquet(folder / f'ps1_stars--DES_{band}.parquet') for band in 'grizy']
    ps1 = reduce(
        lambda df1, df2: df1.join(df2, on='ps1_id', how='inner'),
        ps1_transformed_to_des,
        ps1,
    )
    del ps1_transformed_to_des
    ps1 = ps1.with_columns(
        [pl.lit(None, dtype).alias(column) for column, dtype in ps1_x_des.schema.items() if column not in ps1.columns],
    )
    # No need to cast des_mag and des_magerr to Float32, because they are already Float32.

    stars = pl.concat([ps1_x_des, des.select(ps1_x_des.columns), ps1.select(ps1_x_des.columns)], how='vertical')
    del ps1_x_des, des, ps1

    stars.write_parquet(folder / 'stars.parquet')


if __name__ == '__main__':
    main()
