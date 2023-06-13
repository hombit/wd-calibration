import re
from collections import namedtuple


def compose_model_filename(*, input_survey: str, input_bands: list[str], output_survey: str, output_band: str):
    return f'{output_survey}_{output_band}-{input_survey}_{"--".join(input_bands)}.onnx'


ParsedModelFilename = namedtuple('ParsedModelFilename', ['output_survey', 'output_band', 'input_survey', 'input_bands',
                                                         'filename'])


def parse_model_filename(filename) -> ParsedModelFilename:
    pattern = r'''
        ^
        (?P<output_survey>[A-Za-z0-9]+)_
        (?P<output_band>[A-Za-z0-9]+)-
        (?P<input_survey>[A-Za-z0-9]+)_
        (?P<input_bands>[A-Za-z0-9-]+)
        \.onnx$
    '''
    match = re.match(pattern, filename, re.VERBOSE)

    if match is None:
        raise ValueError('Invalid model filename')

    input_survey = match.group('input_survey')
    input_bands = match.group('input_bands').split('--')
    output_survey = match.group('output_survey')
    output_band = match.group('output_band')
    return ParsedModelFilename(output_survey, output_band, input_survey, input_bands, filename)