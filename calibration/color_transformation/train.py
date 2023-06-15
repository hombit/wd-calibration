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
from torch.utils.data import DataLoader, TensorDataset, random_split

from calibration.color_transformation.model_filename import compose_model_filename


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
           input_survey: str, output_survey: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([df[f"{input_survey.lower()}_mag_{band}"] for band in input_bands], axis=1)
    y = df[f"{output_survey.lower()}_mag_{output_band}"].to_numpy(copy=True)
    return X, y


@dataclasses.dataclass
class ResudualStats:
    count: int
    mean: float
    std: float
    rmse: float
    median: float
    within_0_02: float

    @classmethod
    def from_array(cls, a) -> 'ResudualStats':
        assert a.ndim == 1
        return ResudualStats(
            count=a.size,
            mean=np.mean(a),
            std=np.std(a),
            rmse=np.sqrt(np.mean(a ** 2)),
            median=np.median(a),
            within_0_02=np.mean(np.abs(a) < 0.02),
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

    def residual_hist(self, residuals):
        plt.figure()
        residuals_range = np.linspace(-0.04, 0.04, 100)
        residuals_mu, residuals_sigma = np.mean(residuals), np.std(residuals, ddof=3)
        stats = ResudualStats.from_array(residuals)
        plt.hist(residuals, bins=residuals_range.shape[0], range=[residuals_range[0], residuals_range[-1]],
                 label=f'Count={stats.count:,d}\nμ={residuals_mu:.6f}\nσ={stats.std:.6f}\nWithin ± 0.02: {stats.within_0_02 * 100:.2f}%')
        plt.legend()
        plt.grid()
        plt.xlabel(f'{self.output_survey} {self.output_band} (data - model)')
        self._show_or_save(f'{self.output_band}_residual_hist.{self.img_format}')
        plt.close()

    def true_vs_model(self, true, model):
        plt.figure()
        plt.scatter(true, model, s=1, marker='x', color='b')
        line_min = np.quantile(true, 0.0001)
        line_max = np.quantile(true, 0.9999)
        plt.plot([line_min, line_max], [line_min, line_max], lw=1, color='k', ls='--')
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

    def correction_color(self, x, y, residuals, color):
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

    def covariance_color(self, x, residuals, color, *, subsample: int = 10_000, resolution : int = 32):
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
        self._show_or_save(f'{self.output_band}_covariance_{blue_band}-{red_band}.{self.img_format}')
        plt.close()


class Model(nn.Module):
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


class ScaledModel(nn.Module):
    def __init__(self, model: Model, x_scaler: StandardScaler, y_scaler: StandardScaler, idx_support: int):
        super().__init__()

        self.x_scaler = nn.Linear(model.in_features, model.in_features, bias=True)
        self.x_scaler.requires_grad = False
        self.x_scaler.weight = nn.Parameter(torch.diag(torch.tensor(1.0 / x_scaler.scale_, dtype=torch.float32)))
        self.x_scaler.bias = nn.Parameter(torch.tensor(-x_scaler.mean_ / x_scaler.scale_, dtype=torch.float32))

        self.y_scaler = nn.Linear(1, 1, bias=True)
        self.y_scaler.requires_grad = False
        self.y_scaler.weight = nn.Parameter(torch.diag(torch.tensor(y_scaler.scale_, dtype=torch.float32)))
        self.y_scaler.bias = nn.Parameter(torch.tensor(y_scaler.mean_, dtype=torch.float32))

        self.idx_support = idx_support

        self.layers = nn.Sequential(
            self.x_scaler,
            model,
            self.y_scaler,
        )

    def forward(self, x):
        return self.layers(x) + x[:, self.idx_support].unsqueeze(-1)


class RegressionTask(pl.LightningModule):
    def __init__(self, model, learning_rate=3e-4):
        super(RegressionTask, self).__init__()
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = torch.mean((predictions.squeeze() - y.squeeze()) ** 2)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = torch.mean((predictions.squeeze() - y.squeeze()) ** 2)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


@dataclasses.dataclass
class TrainedModel:
    idx_support: int
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    task: RegressionTask
    model: Model
    input_bands: list[str]
    output_band: str

    def __call__(self, x):
        x_scaled = torch.tensor(self.x_scaler.transform(x), dtype=torch.float32)
        predictions = self.model(x_scaled)
        predictions_inv_scaled = self.y_scaler.inverse_transform(predictions.detach().numpy()).squeeze()
        return predictions_inv_scaled + x[..., self.idx_support]

    def export(self, path):
        input_names = '+'.join(self.input_bands)
        output_names = self.output_band

        torch.onnx.export(
            model=ScaledModel(self.model, self.x_scaler, self.y_scaler, self.idx_support),
            args=torch.zeros((1, self.x_scaler.n_features_in_), dtype=torch.float32),
            f=path,
            input_names=[input_names],
            output_names=[output_names],
            dynamic_axes={input_names: {0: "batch_size"}, output_names: {0: "batch_size"}},
        )


def train(X, y, *, batch_size: int = 1024, n_epoch: int = 300, idx_support: int, input_bands: list[str],
          output_band: str):
    y = y - X[:, idx_support]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, shuffle=True)

    x_scaler: StandardScaler = StandardScaler().fit(x_train)
    x_train, x_test = x_scaler.transform(x_train), x_scaler.transform(x_test)

    y_scaler: StandardScaler = StandardScaler().fit(y_train[:, None])
    y_train, y_test = y_scaler.transform(y_train[:, None]).squeeze(), y_scaler.transform(y_test[:, None]).squeeze()

    x_train, y_train, x_test, y_test = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train,
                                                                                                dtype=torch.float32), torch.tensor(
        x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    model = Model(in_features=x_train.shape[1])
    task = RegressionTask(model)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_test, y_test)

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

    return TrainedModel(
        idx_support=idx_support,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        input_bands=input_bands,
        output_band=output_band,
        task=task,
        model=model,
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

    return parsed


def main(args=None) -> None:
    args = parse_args(args)

    df = get_data(args.photometry, args.random_subsample)
    X, y = get_Xy(df, input_bands=args.input_bands, output_band=args.output_band,
                  input_survey=args.input_survey, output_survey=args.output_survey)
    del df

    n_train = int(args.train_test_split * X.shape[0])
    train_idx = slice(0, n_train, 1)
    test_idx = slice(n_train, None, 1)

    idx_support = args.input_bands.index(args.support_band)

    torch.manual_seed(0)
    model_fn = train(
        torch.tensor(X[train_idx], dtype=torch.float32),
        torch.tensor(y[train_idx], dtype=torch.float32),
        batch_size=1024,
        n_epoch=10000,
        idx_support=idx_support,
        input_bands=args.input_bands,
        output_band=args.output_band,
    )

    pred = model_fn(X[test_idx])
    residuals = y[test_idx] - pred

    if args.modeldir is not None:
        args.modeldir.mkdir(exist_ok=True, parents=True)
        model_filename = compose_model_filename(output_survey=args.output_survey, output_band=args.output_band,
                                                input_survey=args.input_survey, input_bands=args.input_bands)
        model_path = args.modeldir / model_filename
        model_fn.export(model_path)

        stats = ResudualStats.from_array(residuals)
        stats_filename = f"{model_path.stem}.json"
        stats.to_json(args.modeldir / stats_filename)


    idx = np.abs(residuals) < 0.02

    plots = Plots.from_args(args)
    plots.residual_hist(residuals)
    plots.true_vs_model(y[test_idx] - X[test_idx, idx_support], pred - X[test_idx, idx_support])
    for phot_color in args.fig_phot_colors:
        if phot_color == args.fig_support_color:
            continue
        plots.color_color(X[test_idx][idx], residuals[idx], args.fig_support_color, phot_color)
    for phot_color in args.fig_phot_colors:
        plots.correction_color(X[test_idx][idx], y[test_idx][idx], residuals[idx], phot_color)
    for band in args.input_bands:
        plots.correction_magnitude(X[test_idx][idx], y[test_idx][idx], residuals[idx], band)
    for phot_color in args.fig_phot_colors:
        plots.covariance_color(X[test_idx][idx], residuals[idx], phot_color)
