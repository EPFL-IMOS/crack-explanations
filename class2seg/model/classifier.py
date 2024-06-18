import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torchvision import models
from torchvision.models import VisionTransformer

from model.classification.explLRP.ViT_LRP import vit_base_patch16_224 as vit_LRP
from model.classification.explLRP.layers_ours import Linear as LinearLRP
from model.classification.resnet import BasicBlock as BasicBlock_Mod
from model.classification.resnet import resnet18_deeplift, resnet34_deeplift
from model.classification import vgg_bcos
from model.classification.bcosconv2d import BcosConv2d, FinalLayer

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


class Conv2dAlphaBeta(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Classifier(pl.LightningModule):
    def __init__(self, model="torchvit", pretrained=False, num_classes=2):
        super().__init__()
        if model == "torchvit":
            self.model = VisionTransformer(
                image_size=32,
                patch_size=1,
                num_layers=1,
                num_heads=4,
                hidden_dim=64,
                mlp_dim=64,
            )
            self.model.conv_proj = nn.Conv2d(12, 64, kernel_size=(1, 1), stride=(1, 1))
            self.model.heads.head = nn.Linear(in_features=64, out_features=num_classes, bias=True)
        elif model == "lrpvit":
            self.model = vit_LRP(pretrained=True)
            self.model.head = LinearLRP(in_features=768, out_features=num_classes, bias=True)
        elif model == "vit":
            self.model = models.vit_b_16(
                weights=models.ViT_B_16_Weights,
                # patch_size=16,
                # num_layers=1,
                # num_heads=4,
                # hidden_dim=64,
                # mlp_dim=64
            )
            # for param in self.model.encoder.layers[:-2].parameters():
            #     param.requires_grad = False
            self.model.heads.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        elif model == "resnet18":
            if pretrained:
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet18()
            self.model.fc = nn.Linear(512, num_classes)
        elif model == "resnet18_deeplift":
            if pretrained:
                self.model = resnet18_deeplift(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.model = resnet18_deeplift()
            self.model.fc = nn.Linear(512, num_classes)
        elif model == "resnet34_deeplift":
            if pretrained:
                self.model = resnet34_deeplift(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.model = resnet34_deeplift()
            self.model.fc = nn.Linear(512, num_classes)
        elif model == "vgg11":
            self.model = models.vgg11(pretrained=pretrained)

            # params = list(self.model.features[3].parameters())
            self.model.features[3] = Conv2dAlphaBeta(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
            # for param1, param2 in zip(self.model.features[3].parameters(), params):
            #     param1.data = param2.data

            # params = list(self.model.features[6].parameters())
            self.model.features[6] = Conv2dAlphaBeta(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
            # for param1, param2 in zip(self.model.features[6].parameters(), params):
            #     param1.data = param2.data

            # self.model.avgpool = nn.Flatten()
            self.model.avgpool = nn.AdaptiveAvgPool2d((8, 8))  # or 7,7
            self.model.classifier[6] = nn.Linear(4096, num_classes)
        elif model == "vgg11_bn":
            self.model = models.vgg11_bn(pretrained=pretrained)

            # 3 -> 4, 6 -> 8 because of BN
            # params = list(self.model.features[4].parameters())
            self.model.features[4] = Conv2dAlphaBeta(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
            # for param1, param2 in zip(self.model.features[4].parameters(), params):
            #     param1.data = param2.data

            # params = list(self.model.features[8].parameters())
            self.model.features[8] = Conv2dAlphaBeta(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
            # for param1, param2 in zip(self.model.features[8].parameters(), params):
            #     param1.data = param2.data

            # self.model.avgpool = nn.Flatten()
            self.model.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # or 7,7
            self.model.classifier[6] = nn.Linear(4096, num_classes)
        elif model == "vgg11_128":
            self.model = models.vgg11(pretrained=pretrained)

            # 3 -> 4, 6 -> 8 because of BN
            # params = list(self.model.features[3].parameters())
            self.model.features[3] = Conv2dAlphaBeta(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
            # for param1, param2 in zip(self.model.features[3].parameters(), params):
            #     param1.data = param2.data

            # params = list(self.model.features[6].parameters())
            self.model.features[6] = Conv2dAlphaBeta(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
            # for param1, param2 in zip(self.model.features[6].parameters(), params):
            #     param1.data = param2.data

            # self.model.avgpool = nn.Flatten()
            self.model.avgpool = nn.AdaptiveAvgPool2d((8, 8))  # or 7,7
            (
                self.model.classifier[0],
                self.model.classifier[3],
                self.model.classifier[6],
            ) = (nn.Linear(512 * 8 * 8, 128), nn.Linear(128, 128), nn.Linear(128, num_classes))
        elif model == "vgg11_bn_128":
            self.model = models.vgg11_bn(pretrained=pretrained)

            # 3 -> 4, 6 -> 8 because of BN
            # params = list(self.model.features[4].parameters())
            self.model.features[4] = Conv2dAlphaBeta(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
            # for param1, param2 in zip(self.model.features[4].parameters(), params):
            #     param1.data = param2.data

            # params = list(self.model.features[8].parameters())
            self.model.features[8] = Conv2dAlphaBeta(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
            # for param1, param2 in zip(self.model.features[8].parameters(), params):
            #     param1.data = param2.data

            # self.model.avgpool = nn.Flatten()
            self.model.avgpool = nn.AdaptiveAvgPool2d((8, 8))  # or 7,7
            (
                self.model.classifier[0],
                self.model.classifier[3],
                self.model.classifier[6],
            ) = (nn.Linear(512 * 8 * 8, 128), nn.Linear(128, 128), nn.Linear(128, num_classes))
        elif model == "vgg11_bcos":
            self.model = vgg_bcos.vgg11(pretrained=pretrained)

            self.model.classifier = nn.Sequential(
                BcosConv2d(512, 4096, kernel_size=8, padding=3, scale_fact=1000),
                # nn.ReLU(True),
                # nn.Dropout(),
                BcosConv2d(4096, 4096, scale_fact=1000),
                # nn.ReLU(True),
                # nn.Dropout(),
                BcosConv2d(4096, num_classes, scale_fact=1000),
                nn.AdaptiveAvgPool2d((1, 1)),
                FinalLayer(bias=0, norm=1)
            )
        elif model == "vgg11_128_bcos":
            self.model = vgg_bcos.vgg11(pretrained=pretrained)

            self.model.classifier = nn.Sequential(
                BcosConv2d(512, 128, kernel_size=8, padding=3, scale_fact=1000),
                BcosConv2d(128, 128, scale_fact=1000),
                BcosConv2d(128, num_classes, scale_fact=1000),
                nn.AdaptiveAvgPool2d((1, 1)),
                FinalLayer(bias=0, norm=1)
            )
        elif model == "vgg11_128_bcos_dropout":
            self.model = vgg_bcos.vgg11(pretrained=pretrained)

            self.model.classifier = nn.Sequential(
                BcosConv2d(512, 128, kernel_size=8, padding=3, scale_fact=1000),
                nn.Dropout(),
                BcosConv2d(128, 128, scale_fact=1000),
                nn.Dropout(),
                BcosConv2d(128, num_classes, scale_fact=1000),
                nn.AdaptiveAvgPool2d((1, 1)),
                FinalLayer(bias=0, norm=1)
            )
        else:
            return NotImplementedError()

        #self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        y_scores = y_hat.softmax(dim=1)
        # y_hat = self(x).squeeze(1)
        # loss = self.criterion(y_hat, y.float())
        # y_scores = y_hat.sigmoid()
        return {
            "y_scores": y_scores.cpu().detach(),
            "y_true": y.cpu().detach(),
            "loss": loss,
        }

    def training_step_end(self, outputs):
        self.log("train_loss", outputs["loss"].cpu().detach().mean())

    def training_epoch_end(self, outputs):
        y_true = np.concatenate([o["y_true"] for o in outputs])
        y_scores = np.concatenate([o["y_scores"] for o in outputs])

        y_true = y_true.reshape(-1).astype(int)
        y_pred = y_scores.argmax(axis=1)
        # y_pred = y_scores > 0.5

        print()
        self.log("train_accuracy", accuracy_score(y_true, y_pred))
        self.log("train_balanced_accuracy", balanced_accuracy_score(y_true, y_pred))

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        y_scores = y_hat.softmax(dim=1)
        # y_hat = self(x).squeeze(1)
        # loss = self.criterion(y_hat, y.float())
        # y_scores = y_hat.sigmoid()
        return {
            "y_scores": y_scores.cpu().detach(),
            "y_true": y.cpu().detach(),
            "loss": loss.cpu().numpy(),
        }

    def validation_epoch_end(self, outputs):
        y_true = np.concatenate([o["y_true"] for o in outputs])
        y_scores = np.concatenate([o["y_scores"] for o in outputs])
        loss = np.stack([o["loss"] for o in outputs])

        y_true = y_true.reshape(-1).astype(int)
        y_pred = y_scores.argmax(axis=1)
        # y_pred = y_scores > 0.5

        print()
        self.log("val_loss", loss.mean())
        self.log("val_accuracy", accuracy_score(y_true, y_pred))
        self.log("val_balanced_accuracy", balanced_accuracy_score(y_true, y_pred))

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        y_scores = y_hat.softmax(dim=1)
        # y_hat = self(x).squeeze(1)
        # loss = self.criterion(y_hat, y.float())
        # y_scores = y_hat.sigmoid()
        return {
            "y_scores": y_scores.cpu().detach(),
            "y_true": y.cpu().detach(),
            "loss": loss.cpu().numpy(),
        }

    def test_epoch_end(self, outputs):
        y_true = np.concatenate([o["y_true"] for o in outputs])
        y_scores = np.concatenate([o["y_scores"] for o in outputs])
        loss = np.stack([o["loss"] for o in outputs])

        y_true = y_true.reshape(-1).astype(int)
        y_pred = y_scores.argmax(axis=1)
        # y_pred = y_scores > 0.5

        print()
        print(y_true)
        print(y_pred)
        print(y_scores)
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        self.log("test_loss", loss.mean())
        self.log("test_accuracy", accuracy_score(y_true, y_pred))
        self.log("test_balanced_accuracy", balanced_accuracy_score(y_true, y_pred))
        self.log("test_f1_score", f1_score(y_true, y_pred))
        self.log("test_precision", precision_score(y_true, y_pred))
        self.log("test_recall", recall_score(y_true, y_pred))
        self.log("test_tpr", cm[1,1] / (cm[1,1] + cm[1,0]))
        self.log("test_tnr", cm[0,0] / (cm[0,0] + cm[0,1]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-8) # for VGG-11-128-alphabeta classifier
        # return torch.optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=1e-8) # for VGG-11-128-bcos classifier
