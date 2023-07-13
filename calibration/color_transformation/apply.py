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
    parser.add_argument('--samples-for-errors', type=int, default=10_000,
                        help='Number of samples to estimate errors with --estimate-errors')
    parser.add_argument('--seed-for-errors', type=int, default=0,
                        help='Random seed for sampling with --estimate-errors')
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


def generate_random_samples(session, mag, magerr, *, rng=0, n_samples=10_000):
    n_objects, n_bands = mag.shape

    rng = np.random.default_rng(rng)
    deltas = rng.normal(loc=0, scale=1, size=(n_samples, n_bands)).astype(np.float32)

    # first axis is for object, second is for random sample, third is for passband
    X = mag[:, None, :] + magerr[:, None, :] * deltas[None, :, :]

    return session.run(
        [session.get_outputs()[0].name], {session.get_inputs()[0].name: X.reshape(-1, n_bands)}
    )[0].reshape(n_objects, n_samples)


def estimate_stochastic_errors(session, mag, magerr, *, rng=0, n_samples=10_000):
    samples = generate_random_samples(session, mag, magerr, rng=rng, n_samples=n_samples)
    return np.std(samples, axis=1)


def estimate_total_covariance(mag_session, var_session, mag, magerr,
                              *, rng=0, n_samples=10_000, systematics_sigma=1e-3):
    # Stochastic variance is given by the variance of the model output when varying the input magnitudes within
    # their errors
    stochastic_variance = np.square(estimate_stochastic_errors(var_session, mag, magerr, rng=rng, n_samples=n_samples))

    # Infer the total variance with pre-trained model
    total_variance = apply_model(var_session, mag)

    # In the case of a negative variance, we set it to zero
    # For DES r band it happens for <1% of objects
    systematics_variance = total_variance - stochastic_variance
    systematics_variance = np.where(systematics_variance > 0.0, systematics_variance, 0.0)

    # Variate all magnitudes by `systematics_sigma` to estimate the correlation matrix of the systematics
    # We don't know the true amplitude of the systematics, so we just use a small Gaussian std to estimate it
    systematics_samples = generate_random_samples(mag_session, mag, np.full_like(mag, systematics_sigma), rng=rng,
                                                  n_samples=n_samples)
    systematics_cor = np.corrcoef(systematics_samples, rowvar=True)

    # Convert the correlation matrix to covariance matrix
    systematics_cov = np.sqrt(systematics_variance[:, None] * systematics_variance[None, :]) * systematics_cor

    # Finally sum a diagonal matrix of independent stochastic variance and the systematics covariance
    total_cov = np.diag(stochastic_variance) + systematics_cov

    return total_cov


def main(args=None) -> None:
    args = parse_args(args)

    mag_session = rt.InferenceSession(args.model, providers=rt.get_available_providers())
    if args.var_model is not None:
        var_session = rt.InferenceSession(args.var_model, providers=rt.get_available_providers())

    if args.input_bands != mag_session.get_inputs()[0].name.split('+'):
        raise ValueError(f'Input bands {args.input_bands} do not match model input {mag_session.get_inputs()[0].name}')

    batch_size = args.batch_size
    if args.estimate_errors:
        batch_size //= args.samples_for_errors

    input_df = pd.read_parquet(args.photometry, engine='pyarrow')

    mag_column_names = [f'{args.input_survey.lower()}_mag_{band}' for band in args.input_bands]
    mag = input_df[mag_column_names].to_numpy(dtype=np.float32)
    magerr_column_names = [f'{args.input_survey.lower()}_magerr_{band}' for band in args.input_bands]
    magerr = input_df[magerr_column_names].to_numpy(dtype=np.float32)

    y = []
    y_err = []
    for i_batch in tqdm(range(0, len(mag), batch_size)):
        batch_mag = mag[i_batch:i_batch + batch_size]
        batch_magerr = magerr[i_batch:i_batch + batch_size]
        _y = apply_model(mag_session, batch_mag)
        if args.estimate_errors:
            _y_err = estimate_stochastic_errors(mag_session, batch_mag, batch_magerr, rng=args.seed_for_errors,
                                                n_samples=args.samples_for_errors)
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
