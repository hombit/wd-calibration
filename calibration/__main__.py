from functools import lru_cache

import pandas as pd


@lru_cache(1)
def get_object_names():
    url = 'https://github.com/gnarayan/WDdata/raw/master/out/tables/big_table_ILAPHv5_f99_nodisp_abmag_rvfix.txt'



def main():
    pass


if __name__ == '__main__':
    main()