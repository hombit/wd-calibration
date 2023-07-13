import dataclasses
import json
from argparse import ArgumentParser, Namespace
from itertools import chain
from pathlib import Path
from typing import Collection, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.colors import Colormap, LinearSegmentedColormap
from pytorch_lightning.callbacks import EarlyStopping
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from calibration.color_transformation.model_filename import compose_model_filename
from calibration.scaler_to_torch import scaler_to_input_layer, scaler_to_output_layer


def get_data(path: Path, subsample: Optional[int] = None) -> pd.DataFrame:
    match path.suffix:
        case '.parquet':
            df = pd.read_parquet(path, engine='pyarrow')
        case _:
            df = pd.read_csv(path, engine='pyarrow')
    if subsample is None:
        subsample = df.shape[0]
    return df.sample(n=subsample, replace=False, random_state=0)


def get_Xy(df: pd.DataFrame, *,
           input_bands: Collection[str], output_band: str,
           input_survey: str, output_survey: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.stack([df[f"{input_survey.lower()}_mag_{band}"] for band in input_bands], axis=1)
    y = df[f"{output_survey.lower()}_mag_{output_band}"].to_numpy(copy=True)
    y_err = df[f"{output_survey.lower()}_magerr_{output_band}"].to_numpy(copy=True)
    return X, y, y_err


@dataclasses.dataclass
class ResudualStats:
    count: int
    mean: float
    std: float
    rmse: float
    median: float
    within_0_02: float
    outliers_10: int

    @classmethod
    def from_array(cls, a) -> 'ResudualStats':
        assert a.ndim == 1
        return ResudualStats(
            count=a.size,
            mean=np.mean(a).item(),
            std=np.std(a).item(),
            rmse=np.sqrt(np.mean(a ** 2)).item(),
            median=np.median(a).item(),
            within_0_02=np.mean(np.abs(a) < 0.02).item(),
            outliers_10=np.count_nonzero(np.abs(a) > 10.0),
        )

    def to_json(self, path: Path):
        with open(path, 'w') as f:
            json.dump(dataclasses.asdict(self), f, indent=4)


@dataclasses.dataclass
class Plots:
    output_band: str
    input_bands: list[str]
    support_band: str
    output_survey: str
    input_survey: str
    img_format: str = 'png'
    figdir: Optional[Path] = None

    cmap: Colormap = dataclasses.field(default_factory=lambda: LinearSegmentedColormap.from_list(
        'custom',
        [(0, 'blue'), (0.5, 'yellow'), (1, 'red')],
        N=256,
    ))

    def __post_init__(self):
        self.support_idx = self.input_bands.index(self.support_band)

        self.figdir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_args(cls, args: Namespace) -> 'Plots':
        return Plots(
            output_band=args.output_band,
            input_bands=args.input_bands,
            support_band=args.support_band,
            output_survey=args.output_survey,
            input_survey=args.input_survey,
            img_format=args.img_format,
            figdir=args.figdir,
        )

    def _show_or_save(self, filename: str) -> None:
        if self.figdir is None:
            plt.show()
            return
        plt.savefig(self.figdir / filename)

    def residual_hist(self, residuals, *, range_max: float = 0.04):
        plt.figure()
        residuals_range = np.linspace(-range_max, range_max, 100)
        residuals_mu, residuals_sigma = np.mean(residuals), np.std(residuals, ddof=3)
        stats = ResudualStats.from_array(residuals)
        plt.hist(residuals, bins=residuals_range.shape[0], range=[residuals_range[0], residuals_range[-1]],
                 label=f'Count={stats.count:,d}\nμ={residuals_mu:.6f}\nσ={stats.std:.6f}\nWithin ± 0.02: {stats.within_0_02 * 100:.2f}%\nOut of ± 10: {stats.outliers_10:,d}')
        plt.legend()
        plt.grid()
        plt.xlabel(f'{self.output_survey} {self.output_band} (data - model)')
        self._show_or_save(f'{self.output_band}_residual_hist.{self.img_format}')
        plt.close()

    def true_vs_model(self, true, model, model_err=None):
        plt.figure()
        plt.scatter(true, model, s=1, marker='x', color='b', alpha=0.1)
        line_min = np.quantile(true, 0.0001)
        line_max = np.quantile(true, 0.9999)
        plt.plot([line_min, line_max], [line_min, line_max], lw=1, color='k', ls='--')
        if model_err is not None:
            plt.errorbar(true, model, yerr=model_err, fmt='none', ecolor='r', elinewidth=0.5, capsize=0.5, alpha=0.1)
        plt.title(f'{self.output_survey} {self.output_band} - {self.input_survey} {self.support_band}')
        plt.xlabel('true')
        plt.ylabel('model')
        plt.grid()
        self._show_or_save(f'{self.output_band}_true_vs_model.{self.img_format}')
        plt.close()

    def color_color(self, x, residuals, c1, c2):
        blue_band1, red_band1 = c1
        blue_band2, red_band2 = c2
        blue_idx1, red_idx1 = (self.input_bands.index(band) for band in c1)
        blue_idx2, red_idx2 = (self.input_bands.index(band) for band in c2)
        plt.figure()
        plt.scatter(
            x=x[:, blue_idx1] - x[:, red_idx1],
            y=x[:, blue_idx2] - x[:, red_idx2],
            s=1,
            c=residuals,
            alpha=0.5,
            vmin=-0.02,
            vmax=0.02,
            cmap=self.cmap,
        )
        plt.xlabel(f'{self.input_survey} {blue_band1}-{red_band1}')
        plt.ylabel(f'{self.input_survey} {blue_band2}-{red_band2}')
        plt.colorbar().set_label('residual')
        plt.grid()
        self._show_or_save(
            f'{self.output_band}_{self.input_survey}_{blue_band2}-{red_band2}_{blue_band1}-{red_band1}.{self.img_format}')
        plt.close()

    def correction_color(self, x, y, residuals, color, err=None):
        blue_band, red_band = color
        blue_idx, red_idx = (self.input_bands.index(band) for band in color)
        plt.scatter(
            x=x[:, blue_idx] - x[:, red_idx],
            y=y - x[:, self.support_idx],
            s=1,
            c=residuals,
            alpha=0.5,
            vmin=-0.02,
            vmax=0.02,
            cmap=self.cmap,
        )
        if err is not None:
            plt.scatter(x[:, blue_idx] - x[:, red_idx], y - x[:, self.support_idx] + err, s=1, c='black', alpha=0.5)
            plt.scatter(x[:, blue_idx] - x[:, red_idx], y - x[:, self.support_idx] - err, s=1, c='black', alpha=0.5)
        plt.xlabel(f'{self.input_survey} {blue_band} - {red_band}')
        plt.ylabel(f'{self.output_survey} {self.output_band} (true) - {self.input_survey} {self.support_band}')
        plt.colorbar().set_label('residual')
        plt.grid()
        self._show_or_save(f'{self.output_band}_correction_{blue_band}-{red_band}.{self.img_format}')
        plt.close()

    def correction_magnitude(self, x, y, residuals, band):
        band_idx = self.input_bands.index(band)
        plt.scatter(
            x=x[:, band_idx],
            y=y - x[:, self.support_idx],
            s=1,
            c=residuals,
            alpha=0.5,
            vmin=-0.02,
            vmax=0.02,
            cmap=self.cmap,
        )
        plt.xlabel(f'{self.input_survey} {band}')
        plt.ylabel(f'{self.output_survey} {self.output_band} (true) - {self.input_survey} {self.support_band}')
        plt.colorbar().set_label('residual')
        plt.grid()
        self._show_or_save(f'{self.output_band}_correction_{band}.{self.img_format}')
        plt.close()

    def covariation_color(self, x, residuals, color, *, subsample: int = 10_000, resolution : int = 32):
        assert residuals.shape[0] >= subsample * 2

        blue_band, red_band = color
        blue_idx, red_idx = (self.input_bands.index(band) for band in color)

        coord1 = x[:subsample, blue_idx] - x[:subsample, red_idx]
        coord2 = x[-subsample:, blue_idx] - x[-subsample:, red_idx]
        coord_min, coord_max = np.min(coord1), np.max(coord1)
        coord_range = np.linspace(coord_min, coord_max, resolution + 1)

        idx1 = np.digitize(coord1, coord_range)
        idx2 = np.digitize(coord2, coord_range)
        labels = idx1[:, None] * resolution + idx2

        r1 = residuals[:subsample]
        r2 = residuals[-subsample:]
        r_prod = r1[:, None] @ r2[None, :]

        image = ndimage.mean(r_prod, labels=labels, index=np.arange(resolution*resolution))
        image = np.array(image).reshape(resolution, resolution)

        plt.imshow(
            image,
            extent=[coord_min, coord_max, coord_min, coord_max],
            origin='lower',
            cmap=self.cmap,
            vmin=-0.02**2,
            vmax=0.02**2,
            interpolation='nearest',
        )
        plt.colorbar().set_label('residual')
        plt.xlabel(f'{self.input_survey} {blue_band}-{red_band}')
        plt.ylabel(f'{self.input_survey} {blue_band}-{red_band}')
        self._show_or_save(f'{self.output_band}_covariation_{blue_band}-{red_band}.{self.img_format}')
        plt.close()


class TransformationModel(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        self.hidden_layers = nn.Sequential(
            nn.Linear(self.in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(32, 1)

    def features(self, x):
        return self.hidden_layers(x)

    def forward(self, x):
        return self.output_layer(self.hidden_layers(x))


class VariationModel(nn.Module):
    def __init__(self, in_features: int, minimum: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.minimum = minimum
        activation = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.layers = nn.Sequential(
            nn.Linear(self.in_features, 64),
            activation,
            nn.Linear(64, 32),
            activation,
            nn.Linear(32, 32),
            activation,
            nn.Linear(32, 1),
        )
        # Output must be positive
        self.output_regularizer = torch.square

    def forward(self, x):
        return self.output_regularizer(self.layers(x)) + self.minimum


class ScaledTransformationModel(nn.Module):
    def __init__(self, model: TransformationModel, x_scaler: StandardScaler, y_scaler: StandardScaler, idx_support: int):
        super().__init__()

        self.x_scaler = scaler_to_input_layer(x_scaler)
        self.y_scaler = scaler_to_output_layer(y_scaler)
        self.idx_support = idx_support

        self.layers = nn.Sequential(
            self.x_scaler,
            model,
            self.y_scaler,
        )

    def forward(self, x):
        return self.layers(x) + x[:, self.idx_support].unsqueeze(-1)


class ScaledVariationModel(nn.Module):
    def __init__(self, model: VariationModel, x_scaler: StandardScaler, y_scaler: StandardScaler):
        super().__init__()

        self.x_scaler = scaler_to_input_layer(x_scaler)
        self.y_scaler = scaler_to_output_layer(y_scaler)

        self.layers = nn.Sequential(
            self.x_scaler,
            model,
            self.y_scaler,
        )

    def forward(self, x):
        return self.layers(x)


class TransformationRegressionTask(pl.LightningModule):
    def __init__(self, model, learning_rate=3e-4):
        super(TransformationRegressionTask, self).__init__()
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def _loss(self, batch):
        x, y = batch
        predictions = self(x)
        return torch.mean((predictions.squeeze() - y.squeeze()) ** 2)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


class VarianceRegressionTask(TransformationRegressionTask):
    def _loss(self, batch):
        x, y, err2 = batch
        predictions = self(x)
        return torch.mean(
            # learn the variance
            torch.square((predictions.squeeze() - y.squeeze()))
            # and regularize it to be larger than the error
            # + torch.square(torch.relu(err2 - predictions.squeeze()))
        )

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("val_loss", loss)


@dataclasses.dataclass
class TrainedTransformationModel:
    idx_support: int
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    task: TransformationRegressionTask
    model: TransformationModel
    input_bands: list[str]
    output_band: str

    def __post_init__(self):
        assert len(self.input_bands) == self.model.in_features

    def __call__(self, x):
        x_scaled = torch.tensor(self.x_scaler.transform(x), dtype=torch.float32)
        predictions = self.model(x_scaled)
        predictions_inv_scaled = self.y_scaler.inverse_transform(predictions.detach().numpy()).squeeze()
        return predictions_inv_scaled + x[..., self.idx_support]

    def export(self, path):
        input_names = '+'.join(self.input_bands)
        output_names = self.output_band

        torch.onnx.export(
            model=ScaledTransformationModel(self.model, self.x_scaler, self.y_scaler, self.idx_support),
            args=torch.zeros((1, self.x_scaler.n_features_in_), dtype=torch.float32),
            f=path,
            input_names=[input_names],
            output_names=[output_names],
            dynamic_axes={input_names: {0: "batch_size"}, output_names: {0: "batch_size"}},
        )


@dataclasses.dataclass
class TrainedVariationModel:
    model: VariationModel
    task: TransformationRegressionTask
    input_bands: list[str]
    output_band: str
    x_scaler: StandardScaler
    y_scaler: StandardScaler

    def __post_init__(self):
        assert len(self.input_bands) == self.model.in_features

    def __call__(self, x):
        x_scaled = torch.tensor(self.x_scaler.transform(x), dtype=torch.float32)
        predictions = self.model(x_scaled)
        return self.y_scaler.inverse_transform(predictions.detach().numpy()).squeeze()

    def export(self, path):
        input_names = '+'.join(self.input_bands)
        output_names = f'{self.output_band}_var'

        torch.onnx.export(
            model=ScaledVariationModel(self.model, self.x_scaler, self.y_scaler),
            args=torch.zeros((1, self.model.in_features), dtype=torch.float32),
            f=path,
            input_names=[input_names],
            output_names=[output_names],
            dynamic_axes={input_names: {0: "batch_size"}, output_names: {0: "batch_size"}},
        )


def train_transformation(X, y, *, batch_size: int = 1024, n_epoch: int = 300, idx_support: int, input_bands: list[str],
                         output_band: str):
    y = y - X[:, idx_support]

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0, shuffle=True)

    x_scaler: StandardScaler = StandardScaler().fit(x_train)
    x_train, x_val = x_scaler.transform(x_train), x_scaler.transform(x_val)

    y_scaler: StandardScaler = StandardScaler().fit(y_train[:, None])
    y_train, y_val = y_scaler.transform(y_train[:, None]).squeeze(), y_scaler.transform(y_val[:, None]).squeeze()

    x_train, y_train, x_val, y_val = (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    model = TransformationModel(in_features=x_train.shape[1])
    task = TransformationRegressionTask(model)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1 << 14)

    trainer = pl.Trainer(
        max_epochs=n_epoch,
        accelerator='auto',
        enable_progress_bar=False,
        logger=True,
        callbacks=[
            # LearningRateFinder(),
            EarlyStopping('val_loss', patience=10, mode='min'),
        ]
    )
    trainer.fit(task, train_dataloader, val_dataloader)

    return TrainedTransformationModel(
        idx_support=idx_support,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        input_bands=input_bands,
        output_band=output_band,
        task=task,
        model=model,
    )


def train_variation(X, y, err2, *, batch_size: int = 1024, n_epoch: int = 300, input_bands: list[str],
                    output_band: str, minimum_variance: float = 0):
    x_train, x_val, y_train, y_val, err2_train, err2_val = train_test_split(X, y, err2, test_size=0.25, random_state=0,
                                                                            shuffle=True)

    x_scaler: StandardScaler = StandardScaler().fit(x_train)
    x_train, x_val = x_scaler.transform(x_train), x_scaler.transform(x_val)

    y_scaler: StandardScaler = StandardScaler(with_mean=False).fit(y_train[:, None])
    y_train, y_val = y_scaler.transform(y_train[:, None]).squeeze(), y_scaler.transform(y_val[:, None]).squeeze()
    err2_train, err2_val, minimum_variance = (
        y_scaler.transform(err2_train[:, None]).squeeze(),
        y_scaler.transform(err2_val[:, None]).squeeze(),
        y_scaler.transform(np.array([[minimum_variance]])).item(),
    )

    x_train, y_train, x_val, y_val, err2_train, err2_val = (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(err2_train, dtype=torch.float32),
        torch.tensor(err2_val, dtype=torch.float32),
    )

    train_dataset = TensorDataset(x_train, y_train, err2_train)
    val_dataset = TensorDataset(x_val, y_val, err2_val)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1 << 14)

    model = VariationModel(in_features=x_train.shape[1], minimum=minimum_variance)
    task = VarianceRegressionTask(model)

    trainer = pl.Trainer(
        max_epochs=n_epoch,
        accelerator='auto',
        enable_progress_bar=False,
        logger=True,
        callbacks=[
            # LearningRateFinder(),
            EarlyStopping('val_loss', patience=10, mode='min'),
        ]
    )
    trainer.fit(task, train_dataloader, val_dataloader)

    return TrainedVariationModel(
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        input_bands=input_bands,
        output_band=output_band,
        model=model,
        task=task,
    )


def parse_args(args=None) -> Namespace:
    def color_parser(s: str) -> tuple[str, str]:
        color_blue, color_red = s.split(',')
        return color_blue, color_red

    parser = ArgumentParser()
    parser.add_argument('photometry', type=Path)
    parser.add_argument('--output-band', type=str, required=True)
    parser.add_argument('--input-bands', type=str, nargs='+', required=True)
    parser.add_argument('--support-band', type=str, default=None,
                        help='Input band to use as support for the regression, default is the same as the output band')
    parser.add_argument('--input-survey', type=str, required=True,
                        help='Input survey name')
    parser.add_argument('--output-survey', type=str, required=True,
                        help='Output survey name')
    parser.add_argument('--random-subsample', type=int, default=None,
                        help='Randomly subsample the data to this number of data points')
    parser.add_argument('--figdir', type=Path, default=None,
                        help='Directory to save the figures, if not given, show the figures')
    parser.add_argument('--img-format', type=str, default='png', )
    parser.add_argument('--fig-phot-colors', type=color_parser, nargs='+', default=None,
                        help='Photometric colors to use as plots: a list of comma separated passbands specified by --input-bands. Default is `zip(input_colors[-1:], input_colors[1:])`')
    parser.add_argument('--fig-support-color', type=color_parser, default=None,
                        help='Base color to use for color-color plots, default is the first two bands in --input-bands')
    parser.add_argument('--modeldir', type=Path, default=None,
                        help='Path to save the model, if not given, do not save the model')
    parser.add_argument('--train-test-split', type=float, default=0.2,
                        help='Fraction of data to use for training')
    parser.add_argument('--force-retrain', default=None, choices=['both', 'var', 'no', ''],
                        help='NOT IMPLEMENTED Force retrain the model, default is to retrain only if the model file is not found')

    parsed = parser.parse_args(args)

    if parsed.support_band is None:
        parsed.support_band = parsed.output_band
    if parsed.support_band not in parsed.input_bands:
        raise ValueError(f'Support band "{parsed.support_band}" not in input bands: {parsed.input_bands}')

    if parsed.fig_phot_colors is None:
        parsed.fig_phot_colors = list(zip(parsed.input_bands[:-1], parsed.input_bands[1:]))
    if set(chain.from_iterable(parsed.fig_phot_colors)) != set(parsed.input_bands):
        raise ValueError(
            f'Bands of the photometric colors must use all of the input bands: --fig_phot_colors={parsed.fig_phot_colors}, --input_bands={parsed.input_bands}')

    if parsed.fig_support_color is None:
        parsed.fig_support_color = parsed.input_bands[0], parsed.input_bands[1]
    if not set(parsed.fig_support_color).issubset(parsed.input_bands):
        raise ValueError(
            f'Support color must use two of the input bands: --fig_support_color={parsed.fig_support_color}, --input_bands={parsed.input_bands}')

    if parsed.train_test_split >= 0.5:
       raise ValueError('We are training two models, so the train-test split must be less than 0.5')

    return parsed


def main(args=None) -> None:
    args = parse_args(args)

    df = get_data(args.photometry, args.random_subsample)
    X, y, y_err = get_Xy(df, input_bands=args.input_bands, output_band=args.output_band,
                         input_survey=args.input_survey, output_survey=args.output_survey)
    del df

    n_train = int(args.train_test_split * X.shape[0])
    train_transform_idx = slice(0, n_train, 1)
    test_transform_idx = slice(n_train, None, 1)
    # We train variation model on the test set of the main transformation model
    train_variation_idx = slice(n_train, n_train + n_train, 1)
    test_variation_idx = slice(n_train + n_train, None, 1)

    idx_support = args.input_bands.index(args.support_band)

    torch.manual_seed(0)
    transform_model_fn = train_transformation(
        torch.tensor(X[train_transform_idx], dtype=torch.float32),
        torch.tensor(y[train_transform_idx], dtype=torch.float32),
        batch_size=1024,
        n_epoch=10000,
        idx_support=idx_support,
        input_bands=args.input_bands,
        output_band=args.output_band,
    )

    # we should never use train set, but it is convinient to have the whole dataset for slicing
    all_pred = transform_model_fn(X)
    test_transform_pred = all_pred[test_transform_idx]
    all_residuals = y - all_pred
    test_transform_residuals = all_residuals[test_transform_idx]

    if args.modeldir is not None:
        args.modeldir.mkdir(exist_ok=True, parents=True)
        model_filename = compose_model_filename(output_survey=args.output_survey, output_band=args.output_band,
                                                input_survey=args.input_survey, input_bands=args.input_bands)
        model_path = args.modeldir / model_filename
        transform_model_fn.export(model_path)

        stats = ResudualStats.from_array(test_transform_residuals)
        stats_filename = f"{model_path.stem}.json"
        stats.to_json(args.modeldir / stats_filename)

    romanian_idx = np.abs(test_transform_residuals) < 0.02


    torch.manual_seed(0)
    variation_model_fn = train_variation(
        torch.tensor(X[train_variation_idx], dtype=torch.float32),
        torch.tensor(np.square(all_residuals[train_variation_idx]), dtype=torch.float32),
        torch.tensor(np.square(y_err[train_variation_idx]), dtype=torch.float32),
        batch_size=1024,
        n_epoch=10000,
        input_bands=args.input_bands,
        output_band=args.output_band,
        # I think it is ok to use the whole dataset to pick the minimum error,
        # it will help us to avoid crazy outliers in the predictions
        minimum_variance=np.square(0.5 * np.min(y_err)),
    )

    all_var_pred = variation_model_fn(X)
    test_var_pred = all_var_pred[test_variation_idx]
    all_var_residuals = all_residuals - all_var_pred
    test_var_residuals = all_var_residuals[test_variation_idx]

    if args.modeldir is not None:
        args.modeldir.mkdir(exist_ok=True, parents=True)
        model_filename = 'var_' + compose_model_filename(output_survey=args.output_survey, output_band=args.output_band,
                                                         input_survey=args.input_survey, input_bands=args.input_bands)
        model_path = args.modeldir / model_filename
        variation_model_fn.export(model_path)

        stats = ResudualStats.from_array(test_var_residuals)
        stats_filename = f"{model_path.stem}.json"
        stats.to_json(args.modeldir / stats_filename)


    plots_transform = Plots.from_args(args)
    plots_transform.residual_hist(test_transform_residuals)
    plots_transform.true_vs_model(
        y[test_transform_idx] - X[test_transform_idx, idx_support],
        test_transform_pred - X[test_transform_idx, idx_support],
        model_err=np.sqrt(all_var_pred[test_transform_idx]),
    )
    for phot_color in args.fig_phot_colors:
        if phot_color == args.fig_support_color:
            continue
        plots_transform.color_color(X[test_transform_idx][romanian_idx], test_transform_residuals[romanian_idx], args.fig_support_color, phot_color)
    for phot_color in args.fig_phot_colors:
        plots_transform.correction_color(
            X[test_transform_idx][romanian_idx],
            y[test_transform_idx][romanian_idx],
            test_transform_residuals[romanian_idx],
            phot_color,
            # err=np.sqrt(all_var_pred[test_transform_idx][romanian_idx]),
        )
    for band in args.input_bands:
        plots_transform.correction_magnitude(
            X[test_transform_idx][romanian_idx],
            y[test_transform_idx][romanian_idx],
            test_transform_residuals[romanian_idx],
            band,
        )
    for phot_color in args.fig_phot_colors:
        plots_transform.covariation_color(X[test_transform_idx][romanian_idx], test_transform_residuals[romanian_idx], phot_color)


    plots_variation = Plots.from_args(args)
    plots_variation.output_band = f'var_{args.output_band}'
    plots_variation.residual_hist(all_residuals[test_variation_idx] / np.sqrt(test_var_pred), range_max=3.0)
