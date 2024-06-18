import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from model.ternausnet import UNet11, UNet16

from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score


def dice_loss(pred, target, smooth = 1e-5):

    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))

    dice= 2.0 * (intersection + smooth) / (union + smooth)
    loss = 1.0 - dice

    return loss.mean()


class Segmenter(pl.LightningModule):
    def __init__(self, model="unet11", pretrained=False, num_classes=2):
        super().__init__()
        if model == "unet11":
            self.model = UNet11(num_classes=num_classes, pretrained=pretrained)
        elif model == "unet16":
            self.model = UNet16(num_classes=num_classes, pretrained=pretrained)
        else:
            return NotImplementedError()

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = dice_loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        #y_scores = y_hat.softmax(dim=1)
        y_scores = y_hat.sigmoid()
        loss = self.criterion(y_scores, y.unsqueeze(1))
        return {
            "y_scores": y_scores.cpu().detach().numpy(),
            "y_true": y.cpu().detach().numpy(),
            "loss": loss,
        }

    def training_step_end(self, outputs):
        y_true = outputs["y_true"]
        y_scores = outputs["y_scores"]

        y_true = (y_true > 0.5).reshape(-1).astype(int)
        #y_pred = y_scores.argmax(axis=1).reshape(-1)
        y_pred = (y_scores > 0.5).astype(int).reshape(-1)

        print()
        self.log("train_loss", outputs["loss"].cpu().detach().mean())
        self.log("train_f1", f1_score(y_true, y_pred, average="binary", pos_label=1))
        # self.log("train_precision", precision_score(y_true, y_pred, average="binary", pos_label=1))
        # self.log("train_recall", recall_score(y_true, y_pred, average="binary", pos_label=1))
        # self.log("train_jaccard", jaccard_score(y_true, y_pred, average="binary", pos_label=1))


    # def training_epoch_end(self, outputs):
    #     y_true = np.concatenate([o["y_true"] for o in outputs])
    #     y_scores = np.concatenate([o["y_scores"] for o in outputs])

    #     y_true = y_true.reshape(-1).astype(int)
    #     #y_pred = y_scores.argmax(axis=1).reshape(-1)
    #     y_pred = (y_scores > 0.5).astype(int).reshape(-1)

    #     print()
    #     self.log("train_f1", f1_score(y_true, y_pred, average="binary", pos_label=1))
    #     self.log("train_precision", precision_score(y_true, y_pred, average="binary", pos_label=1))
    #     self.log("train_recall", recall_score(y_true, y_pred, average="binary", pos_label=1))
    #     self.log("train_jaccard", jaccard_score(y_true, y_pred, average="binary", pos_label=1))

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        #y_scores = y_hat.softmax(dim=1)
        y_scores = y_hat.sigmoid()
        loss = self.criterion(y_scores, y.unsqueeze(1))
        return {
            "y_scores": y_scores.cpu().detach().numpy(),
            "y_true": y.cpu().detach().numpy(),
            "loss": loss.cpu().detach().numpy(),
        }

    def validation_step_end(self, outputs):
        y_true = outputs["y_true"]
        y_scores = outputs["y_scores"]

        # log first image
        self.log_tb_images(y_true[0], y_scores.squeeze(1)[0])

        y_true = (y_true > 0.5).reshape(-1).astype(int)
        #y_pred = y_scores.argmax(axis=1).reshape(-1)
        y_pred = (y_scores > 0.5).astype(int).reshape(-1)

        print()
        self.log("val_loss", outputs["loss"].mean())
        self.log("val_f1", f1_score(y_true, y_pred, average="binary", pos_label=1))
        # self.log("val_precision", precision_score(y_true, y_pred, average="binary", pos_label=1))
        # self.log("val_recall", recall_score(y_true, y_pred, average="binary", pos_label=1))
        # self.log("val_jaccard", jaccard_score(y_true, y_pred, average="binary", pos_label=1))

    # def validation_epoch_end(self, outputs):
    #     y_true = np.concatenate([o["y_true"] for o in outputs])
    #     y_scores = np.concatenate([o["y_scores"] for o in outputs])
    #     loss = np.stack([o["loss"] for o in outputs])

    #     # log first image
    #     self.log_tb_images(y_true[0], y_scores.squeeze(1)[0])

    #     y_true = y_true.reshape(-1).astype(int)
    #     #y_pred = y_scores.argmax(axis=1).reshape(-1)
    #     y_pred = (y_scores > 0.5).astype(int).reshape(-1)

    #     print()
    #     self.log("val_loss", loss.mean())
    #     self.log("val_f1", f1_score(y_true, y_pred, average="binary", pos_label=1))
    #     self.log("val_precision", precision_score(y_true, y_pred, average="binary", pos_label=1))
    #     self.log("val_recall", recall_score(y_true, y_pred, average="binary", pos_label=1))
    #     self.log("val_jaccard", jaccard_score(y_true, y_pred, average="binary", pos_label=1))

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        #y_scores = y_hat.softmax(dim=1)
        y_scores = y_hat.sigmoid()
        loss = self.criterion(y_scores, y.unsqueeze(1))
        return {
            "y_scores": y_scores.cpu().detach(),
            "y_true": y.cpu().detach(),
            "loss": loss.cpu().numpy(),
        }

    def test_epoch_end(self, outputs):
        y_true = np.concatenate([o["y_true"] for o in outputs])
        y_scores = np.concatenate([o["y_scores"] for o in outputs])
        loss = np.stack([o["loss"] for o in outputs])

        y_true = (y_true > 0.5).reshape(-1).astype(int)
        #y_pred = y_scores.argmax(axis=1).reshape(-1)
        y_pred = (y_scores > 0.5).astype(int).reshape(-1)

        print()
        self.log("test_loss", loss.mean())
        self.log("test_f1", f1_score(y_true, y_pred, average="binary", pos_label=1))
        self.log("test_precision", precision_score(y_true, y_pred, average="binary", pos_label=1))
        self.log("test_recall", recall_score(y_true, y_pred, average="binary", pos_label=1))
        self.log("test_jaccard", jaccard_score(y_true, y_pred, average="binary", pos_label=1))

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
        tb_logger.add_image("Segmentation/True segmentation", image_true.reshape(1, *image_true.shape), self.global_step)
        tb_logger.add_image("Segmentation/Predicted segmentation", image_pred.reshape(1, *image_pred.shape), self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4) # 1e-4, 1e-8
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 60])
            },
        }
