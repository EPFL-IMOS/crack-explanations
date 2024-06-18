import pytorch_lightning as pl

from model.ternausnet import UNet11, UNet16


class UNet11Explainer(pl.LightningModule):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.explainer = UNet11(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        x = self.explainer(x)
        return x


class UNet16Explainer(pl.LightningModule):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.explainer = UNet16(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        x = self.explainer(x)
        return x
