import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import pytorch_lightning as pl
import numpy as np

from model.ternausnet import DecoderBlockV2, Interpolate, ConvRelu


class CAE11(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 50,
        num_filters: int = 32,
        pretrained: bool = False,
        is_deconv: bool = False
    ) -> None:
        """Convolutional AutoEncoder (CAE) with VGG11 encoder.
        Args:
            latent_dim: dimension of latent code
            pretrained:
                False - no pre-trained network is used
                True  - encoder is pre-trained with VGG11
            is_deconv:
                True - use deconv
                False - use Interpolate
        """
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.fc1 = nn.Linear(512 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, self.latent_dim)

        self.fc3 = nn.Linear(self.latent_dim, 128)
        self.fc4 = nn.Linear(128, 512 * 8 * 8)

        self.decoder =  nn.Sequential(
            DecoderBlockV2(
                num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8 * 2, is_deconv
            ),
            DecoderBlockV2(
                num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8, is_deconv
            ),
            DecoderBlockV2(
                num_filters * 8, num_filters * 8, num_filters * 4, is_deconv
            ),
            DecoderBlockV2(
                num_filters * 4, num_filters * 4, num_filters * 2, is_deconv
            ),
            DecoderBlockV2(
                num_filters * 2, num_filters * 2, in_channels, is_deconv
            )
        )
        # self.final = nn.Conv2d(num_filters * 2, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(-1, 512*8*8)
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.fc3(h)
        h = self.fc4(h)
        h = h.view(-1, 512, 8, 8)
        h = self.decoder(h)
        return h

class CAE11FCN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_filters: int = 32,
        pretrained: bool = False,
        is_deconv: bool = False
    ) -> None:
        """Convolutional AutoEncoder (CAE) with VGG11 encoder.
        Args:
            latent_dim: dimension of latent code
            pretrained:
                False - no pre-trained network is used
                True  - encoder is pre-trained with VGG11
            is_deconv:
                True - use deconv
                False - use Interpolate
        """
        super().__init__()

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.decoder =  nn.Sequential(
            DecoderBlockV2(
                num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8 * 2, is_deconv
            ),
            DecoderBlockV2(
                num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8, is_deconv
            ),
            DecoderBlockV2(
                num_filters * 8, num_filters * 8, num_filters * 4, is_deconv
            ),
            DecoderBlockV2(
                num_filters * 4, num_filters * 4, num_filters * 2, is_deconv
            ),
            DecoderBlockV2(
                num_filters * 2, num_filters * 2, in_channels, is_deconv
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.decoder(h)
        return h


class Autoencoder(pl.LightningModule):
    def __init__(self, model="cae11", pretrained=False, in_channels=3, latent_dim=50, is_deconv=False):
        super().__init__()
        if model == "cae11":
            self.model = CAE11(in_channels=in_channels, pretrained=pretrained, latent_dim=latent_dim, is_deconv=is_deconv)
        elif model == "cae11_fcn":
            self.model = CAE11FCN(in_channels=in_channels, pretrained=pretrained, is_deconv=is_deconv)
        else:
            return NotImplementedError()

        self.criterion = nn.MSELoss()
        # self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        x_hat = self(x)
        loss = self.criterion(x, x_hat)
        return {
            # "x": x.cpu().detach(),
            # "x_hat": x_hat.cpu().detach(),
            "loss": loss,
        }

    def training_step_end(self, outputs):
        self.log("train_loss", outputs["loss"].cpu().detach().mean())

    # def training_epoch_end(self, outputs):
        # x = np.concatenate([o["x"] for o in outputs])
        # x_hat = np.concatenate([o["x_hat"] for o in outputs])

        # print()
        # self.log("train_f1", f1_score(y_true, y_pred, average="binary", pos_label=1))
        # self.log("train_precision", precision_score(y_true, y_pred, average="binary", pos_label=1))
        # self.log("train_recall", recall_score(y_true, y_pred, average="binary", pos_label=1))
        # self.log("train_jaccard", jaccard_score(y_true, y_pred, average="binary", pos_label=1))

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        x_hat = self(x)
        loss = self.criterion(x, x_hat)
        return {
            "x": x.cpu().detach(),
            "x_hat": x_hat.cpu().detach(),
            "loss": loss.cpu().numpy(),
        }

    def validation_epoch_end(self, outputs):
        x = np.concatenate([o["x"] for o in outputs])
        x_hat = np.concatenate([o["x_hat"] for o in outputs])
        loss = np.stack([o["loss"] for o in outputs])

        # log first image
        self.log_tb_images(x[4], x_hat[4])

        print()
        self.log("val_loss", loss.mean())
        # self.log("val_f1", f1_score(y_true, y_pred, average="binary", pos_label=1))
        # self.log("val_precision", precision_score(y_true, y_pred, average="binary", pos_label=1))
        # self.log("val_recall", recall_score(y_true, y_pred, average="binary", pos_label=1))
        # self.log("val_jaccard", jaccard_score(y_true, y_pred, average="binary", pos_label=1))

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        x_hat = self(x)
        loss = self.criterion(x, x_hat)
        return {
            "x": x.cpu().detach(),
            "x_hat": x_hat.cpu().detach(),
            "loss": loss.cpu().numpy(),
        }

    def test_epoch_end(self, outputs):
        x = np.concatenate([o["x"] for o in outputs])
        x_hat = np.concatenate([o["x_hat"] for o in outputs])
        loss = np.stack([o["loss"] for o in outputs])

        print()
        self.log("test_loss", loss.mean())
        # self.log("test_f1", f1_score(y_true, y_pred, average="binary", pos_label=1))
        # self.log("test_precision", precision_score(y_true, y_pred, average="binary", pos_label=1))
        # self.log("test_recall", recall_score(y_true, y_pred, average="binary", pos_label=1))
        # self.log("test_jaccard", jaccard_score(y_true, y_pred, average="binary", pos_label=1))

    def log_tb_images(self, image_true, image_pred) -> None:

        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')

        # Log the images
        res = (image_true - image_pred)**2
        tb_logger.add_image("Image/Original", image_true, self.global_step)
        tb_logger.add_image("Image/Reconstruction", image_pred, self.global_step)
        tb_logger.add_image("Image/Residual", res, self.global_step)
        tb_logger.add_image("Image/ResidualNorm", (res-res.min())/(res.max()-res.min()), self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-5)
