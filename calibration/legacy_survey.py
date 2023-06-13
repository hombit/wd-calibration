from ._datalab import DataLab


class LegacySurvey(DataLab):
    __default_dr = 9

    def __init__(self, dr: int = __default_dr):
        self.dr = dr
        super().__init__(f'ls_dr{self.dr}.tractor')
