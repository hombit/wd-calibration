import dataclasses
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from astropy.coordinates import Angle, SkyCoord
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from calibration.scaler_to_torch import scaler_to_output_layer


# def preprocess_coord(ra, dec, *, np=np):
#     """Converts input coordinates in degrees to 0-1 ranges"""
#     x1 = ra / 360.0
#     x2 = 0.5 * (np.sin(np.deg2rad(dec)) + 1.0)
#     return np.stack([x1, x2], axis=1)
def preprocess_coord(ra, dec, *, np=np):
    """Convert input coordinates in degrees to xyz"""
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    x = np.cos(ra_rad) * np.cos(dec_rad)
    y = np.sin(ra_rad) * np.cos(dec_rad)
    z = np.sin(dec_rad)
    return np.stack([x, y, z], axis=1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


class PreprocessCoord(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ra, dec):
        return preprocess_coord(ra, dec, np=torch)


class ScaledModel(nn.Module):
    def __init__(self, model: Model, y_scaler: StandardScaler):
        super().__init__()

        self.x_scaler = PreprocessCoord()
        self.y_scaler = scaler_to_output_layer(y_scaler)

        self.layers = nn.Sequential(
            self.x_scaler,
            model,
            self.y_scaler,
        )

    def forward(self, ra, dec):
        return self.layers(ra, dec)


class RegressionTask(pl.LightningModule):
    def __init__(self, model, learning_rate=3e-4):
        super().__init__()
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


def random_coord_offset(ra_deg, dec_deg, amplitude_deg, *, rng=0) -> tuple[np.ndarray, np.ndarray]:
    n = len(ra_deg)
    coord = SkyCoord(ra_deg, dec_deg, unit='deg')
    rng = np.random.default_rng(rng)
    dx, dy = rng.normal(loc=0, scale=amplitude_deg, size=(2, n))
    position_angle = np.arctan2(dy, dx)
    distance = np.hypot(dx, dy)
    new_coord = coord.directional_offset_by(Angle(position_angle, 'deg'), Angle(distance, 'deg'))
    return new_coord.ra.deg, new_coord.dec.deg


@dataclasses.dataclass
class TrainedModel:
    y_scaler: StandardScaler
    task: RegressionTask
    model: Model

    def __call__(self, ra, dec):
        x = preprocess_coord(ra, dec)
        predictions = self.model(torch.tensor(x, dtype=torch.float32))
        return self.y_scaler.inverse_transform(predictions.detach().numpy()).squeeze()

    def export(self, path):
        output_names = 'offset'

        args = {
            'ra-deg': torch.tensor([0.0], dtype=torch.float32),
            'dec-deg': torch.tensor([0.0], dtype=torch.float32),
        }

        torch.onnx.export(
            model=ScaledModel(self.model, self.y_scaler),
            args=(args,),
            f=path,
            input_names=list(args.keys()),
            output_names=[output_names],
            dynamic_axes={arg: {0: "batch_size"} for arg in args} | {output_names: {0: "batch_size"}},
        )


def train(ra, dec, offset, *, variance_deg: Optional[float], n_epoch: int = 10_000):
    ra_train, ra_val, dec_train, dec_val, y_train, y_val = train_test_split(ra, dec, offset, test_size=0.5,
                                                                            random_state=0, shuffle=True)
    if variance_deg is not None:
        ra_train, dec_train = random_coord_offset(ra_train, dec_train, variance_deg, rng=0)

    X_train, X_val = preprocess_coord(ra_train, dec_train), preprocess_coord(ra_val, dec_val)

    y_scaler: StandardScaler = StandardScaler().fit(y_train[:, None])
    y_train, y_val = y_scaler.transform(y_train[:, None]).squeeze(), y_scaler.transform(y_val[:, None]).squeeze()

    X_train, X_val, y_train, y_val = (torch.tensor(a, dtype=torch.float32) for a in [X_train, X_val, y_train, y_val])

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1 << 14)

    model = Model()
    task = RegressionTask(model)

    trainer = pl.Trainer(
        max_epochs=n_epoch,
        accelerator='auto',
        enable_progress_bar=False,
        logger=True,
        callbacks=[
            EarlyStopping('val_loss', patience=10, mode='min'),
        ]
    )
    trainer.fit(task, train_dataloader, val_dataloader)

    return TrainedModel(y_scaler=y_scaler, task=task, model=model)
