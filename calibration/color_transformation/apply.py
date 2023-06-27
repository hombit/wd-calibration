from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import onnxruntime as rt
import pandas as pd
from tqdm import tqdm

from calibration.color_transformation.model_filename import parse_model_filename


def parse_args(args=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('model', type=Path, help='ONNX model from `train-photo-transform`')
    parser.add_argument('photometry', type=Path, help='Data file to apply the model to')
    parser.add_argument('--estimate-errors', action='store_true',
                        help='Estimate errors from the model, if not specified, --var-model must be specified. If specified, it also requires --sigma to be specified')
    parser.add_argument('--var-model', type=Path, default=None, help='ONNX model to estimate errors')
    parser.add_argument('--sigma', type=float, default=0.0, help='Additional uncertainty to add')
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

    if parsed_args.var_model is None and not parsed_args.estimate_errors:
        raise ValueError('Either --estimate-errors or --var-model must be specified')
    if parsed_args.estimate_errors and parsed_args.sigma is None:
        raise ValueError('--sigma must be specified if --estimate-errors is specified')

    return parsed_args


def apply_model(session, X):
    return session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: X})[0].squeeze()


def estimate_errors(session, mag, magerr):
    X = np.stack(
        [mag + sign * np.hstack([np.zeros((mag.shape[0], i)),
                                 magerr[:, i:i + 1],
                                 np.zeros((mag.shape[0], mag.shape[1] - i - 1))])
         for i in range(mag.shape[1])
         for sign in [-1, +1]],
        axis=0,
    ).astype(np.float32).reshape(-1, mag.shape[1])

    y_minus_err, y_plus_err = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: X}
    )[0].reshape(2, mag.shape[1], mag.shape[0])
    y_err = 0.5 * (y_plus_err - y_minus_err)
    y_err = np.sqrt(np.sum(y_err * y_err, axis=0))

    return y_err


def main(args=None) -> None:
    args = parse_args(args)

    mag_session = rt.InferenceSession(args.model, providers=rt.get_available_providers())
    if args.var_model is not None:
        var_session = rt.InferenceSession(args.var_model, providers=rt.get_available_providers())

    if args.input_bands != mag_session.get_inputs()[0].name.split('+'):
        raise ValueError(f'Input bands {args.input_bands} do not match model input {mag_session.get_inputs()[0].name}')

    input_df = pd.read_parquet(args.photometry, engine='pyarrow')

    mag_column_names = [f'{args.input_survey.lower()}_mag_{band}' for band in args.input_bands]
    mag = input_df[mag_column_names].to_numpy(dtype=np.float32)
    magerr_column_names = [f'{args.input_survey.lower()}_magerr_{band}' for band in args.input_bands]
    magerr = input_df[magerr_column_names].to_numpy(dtype=np.float32)

    y = []
    y_err = []
    for i_batch in tqdm(range(0, len(mag), args.batch_size)):
        batch_mag = mag[i_batch:i_batch + args.batch_size]
        batch_magerr = magerr[i_batch:i_batch + args.batch_size]
        _y = apply_model(mag_session, batch_mag)
        if args.estimate_errors:
            _y_err = estimate_errors(mag_session, batch_mag, batch_magerr)
        elif args.var_model is not None:
            _y_err = np.sqrt(apply_model(var_session, batch_mag))
        else:
            raise ValueError('Either --estimate-errors or --var-model must be specified')
        y.append(_y)
        y_err.append(_y_err)
        del _y, _y_err
    y = np.concatenate(y)
    y_err = np.concatenate(y_err) + args.sigma

    output_df = pd.DataFrame(
        {
            'ps1_id': input_df['ps1_id'],
            f'{args.output_survey.lower()}_mag_{args.output_band}': y,
            f'{args.output_survey.lower()}_magerr_{args.output_band}': y_err,
        },
    )
    output_df.to_parquet(args.output, engine='pyarrow', index=False)
