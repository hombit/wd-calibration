import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

from ._datalab import DataLab


class DesCutOut:
    base_url = 'https://des.ncsa.illinois.edu'
    api_url = f'{base_url}/desaccess/api'
    files_url = f'{base_url}/files-desaccess'

    database = 'desdr'
    release = 'dr2'

    def __init__(self, *, user='kostya', password):
        self.user = user
        self.session = requests.Session()
        self.token = self._get_new_auth_token(user, password)

    def _get_new_auth_token(self, user, password):
        response = self.session.post(
            f'{self.api_url}/login',
            data={
                'username': user,
                'password': password,
                'database': self.database,
            }
        )
        return response.json()['token']

    def submit_cutout_job(self, data):
        response = self.session.put(
            f'{self.api_url}/job/cutout',
            data=data,
            headers={'Authorization': f'Bearer {self.token}'}
        )
        json = self.process_response(response)
        job_id = json['jobid']
        return job_id

    def get_job_status(self, job_id):
        response = self.session.post(
            f'{self.api_url}/job/status',
            data={
                'job-id': job_id
            },
            headers={'Authorization': f'Bearer {self.token}'},
        )
        json = self.process_response(response)
        return json['jobs'][0]['job_status']

    def wait_job(self, job_id):
        while True:
            job_status = self.get_job_status(job_id)
            if job_status == 'success':
                return
            if job_status == 'failure':
                raise RuntimeError(f'Job {job_id} failed.')
            time.sleep(1)

    def process_response(self, response):
        response.raise_for_status()
        json = response.json()
        if json['status'] != 'ok':
            raise RuntimeError(f'Error submitting cutout job: {json["message"]}')
        self.token = json['new_token']
        return json

    def cutout_job_sync(self, data, path):
        job_id = self.submit_cutout_job(data)
        self.wait_job(job_id)
        self.download_dir(self.job_url(job_id), path)

    @staticmethod
    def coords_to_csv_bytes(coords):
        df = pd.DataFrame({'RA': coords.ra.deg, 'DEC': coords.dec.deg})
        byte_stream = BytesIO()
        df.to_csv(byte_stream, header=True, index=False)
        byte_stream.seek(0)
        return byte_stream.read()

    def get_cutouts(self, coords):
        raise NotImplementedError
        positions = self.coords_to_csv_bytes(coords)
        data = {
            'db': self.database,
            'release': self.release,
            'positions': positions,
        }
        path = Path('tmp_des')
        path.mkdir(exist_ok=True, parents=True)
        return self.cutout_job_sync(data, path)

    def job_type(self, job_id):
        response = self.session.post(
            f'{self.api_url}/job/status',
            data={
                'job-id': job_id
            },
            headers={'Authorization': f'Bearer {self.token}'},
        )
        json = self.process_response(response)
        return json['jobs'][0]['job_type']

    def job_url(self, job_id):
        return f'{self.files_url}/{self.user}/{self.job_type(job_id)}/{job_id}'

    def download_dir(self, url, path):
        response = self.session.get(f'{url}/json')
        json = self.process_response(response)
        for item in json:
            suburl = f'{url}/{item["name"]}'
            subpath = path / item['name']
            if item['type'] == 'directory':
                self.download_dir(suburl, subpath)
            elif item['type'] == 'file':
                self.download_file(suburl, subpath)
            else:
                raise RuntimeError(f'Unknown item type: {item["type"]}')

    def download_file(self, url, path):
        with self.session.get(url, stream=True) as response:
            response.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    fh.write(chunk)


class Des(DataLab):
    __default_dr = 2

    def __init__(self, dr: int= __default_dr):
        self.dr = dr
        super().__init__(f'des_dr{self.dr}.main')
