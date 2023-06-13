from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import onnxruntime as rt
import pandas as pd
from tqdm import tqdm

from calibration.color_transformation.model_filename import parse_model_filename, ParsedModelFilename


def parse_args(args=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('model', type=Path, help='ONNX model from `train-photo-transform`')
    parser.add_argument('photometry', type=Path, help='Data file to apply the model to')
    parser.add_argument('--output-band', type=str, default=None,
                        help='Output band name, default is parsed from the model filename')
    parser.add_argument('--input-bands', type=str, nargs='+', default=None,
                        help='Input band names, default is parsed from the model filename')
    parser.add_argument('--input-survey', type=str, default=None,
                        help='Input survey name, default is parsed from the model filename')
    parser.add_argument('--output-survey', type=str, default=None,
                        help='Output survey name, default is parsed from the model filename')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output file name, default is built from inputs')
    parser.add_argument('--batch-size', type=int, default=1 << 20,)

    parsed_args = parser.parse_args(args)

    parsed_model_filename = parse_model_filename(parsed_args.model.name)
    if parsed_args.output_band is None:
        parsed_args.output_band = parsed_model_filename.output_band
    if parsed_args.input_bands is None:
        parsed_args.input_bands = parsed_model_filename.input_bands
    if parsed_args.input_survey is None:
        parsed_args.input_survey = parsed_model_filename.input_survey
    if parsed_args.output_survey is None:
        parsed_args.output_survey = parsed_model_filename.output_survey

    if parsed_args.output is None:
        parsed_args.output = (
                parsed_args.photometry.parent
                / f'{parsed_args.photometry.stem}--{parsed_args.output_survey}_{parsed_args.output_band}{parsed_args.photometry.suffix}'
        )

    return parsed_args


def main(args=None) -> None:
    args = parse_args(args)

    session = rt.InferenceSession(args.model, providers=rt.get_available_providers())

    if args.input_bands != session.get_inputs()[0].name.split('+'):
        raise ValueError(f'Input bands {args.input_bands} do not match model input {session.get_inputs()[0].name}')

    input_df = pd.read_parquet(args.photometry, engine='pyarrow')
    column_names = [f'{args.input_survey.lower()}_mag_{band}' for band in args.input_bands]
    X = input_df[column_names].to_numpy(dtype=np.float32)

    y = []
    for i_batch in tqdm(range(0, len(X), args.batch_size)):
        batch = X[i_batch:i_batch + args.batch_size]
        y.append(session.run([args.output_band], {session.get_inputs()[0].name: batch})[0].squeeze())
    y = np.concatenate(y)

    output_df = pd.DataFrame({f'{args.output_survey.lower()}_mag_{args.output_band}': y})
    output_df.to_parquet(args.output, engine='pyarrow', index=False)
