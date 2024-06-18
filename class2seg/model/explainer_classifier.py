import torch
import pytorch_lightning as pl

from torch import nn
from torch.optim import Adam
from pathlib import Path

from model.explainer import UNet11Explainer, UNet16Explainer
from model.classifier import Classifier
from utils import helper
from utils.image_utils import save_mask, save_masked_image, save_all_class_masks
from utils.loss import TotalVariationConv, ClassMaskAreaLoss, entropy_loss
from utils.metrics import SingleLabelMetrics, BinarySegmentationMetrics
from transforms import UnNormalize, MEAN, STD


class ExplainerClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes=2,
        image_size=224,
        classifier_type="vgg11",
        classifier_checkpoint=None,
        fix_classifier=True,
        explainer_type="unet11",
        learning_rate=1e-5,
        class_mask_min_area=0.05,
        class_mask_max_area=0.3,
        use_inverse_classification_loss=True,
        entropy_regularizer=1.0,
        use_mask_variation_loss=True,
        mask_variation_regularizer=1.0,
        use_mask_area_loss=True,
        mask_area_constraint_regularizer=1.0,
        mask_total_area_regularizer=0.1,
        ncmask_total_area_regularizer=0.3,
        baseline=None,
        metrics_threshold=-1.0,
        save_masked_images=False,
        save_masks=False,
        save_all_class_masks=False,
        save_path="./results/"
    ):

        super().__init__()

        self.num_classes = num_classes

        self.setup_explainer(explainer_type=explainer_type, num_classes=num_classes, pretrained=True)  # VERY IMPORTANT
        self.setup_classifier(classifier_type=classifier_type, classifier_checkpoint=classifier_checkpoint, fix_classifier=fix_classifier, num_classes=num_classes)

        self.setup_losses(image_size, class_mask_min_area, class_mask_max_area)
        self.setup_metrics(num_classes=num_classes, metrics_threshold=metrics_threshold)

        self.classifier_type = classifier_type

        # Hyperparameters
        self.learning_rate = learning_rate
        self.entropy_regularizer = entropy_regularizer
        self.use_mask_variation_loss = use_mask_variation_loss
        self.mask_variation_regularizer = mask_variation_regularizer
        self.use_mask_area_loss = use_mask_area_loss
        self.mask_area_constraint_regularizer = mask_area_constraint_regularizer
        self.mask_total_area_regularizer = mask_total_area_regularizer
        self.ncmask_total_area_regularizer = ncmask_total_area_regularizer
        self.use_inverse_classification_loss = use_inverse_classification_loss
        self.baseline = baseline
        if self.in_distribution:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            with open(self.baseline, "rb") as f:
                self.baseline_dist = torch.load(f).to(device)

        # Image display/save settings
        self.save_masked_images = save_masked_images
        self.save_masks = save_masks
        self.save_all_class_masks = save_all_class_masks
        self.save_path = save_path

        self.unnorm = UnNormalize(MEAN, STD)

    def setup_explainer(self, explainer_type, num_classes, pretrained):
        if explainer_type == "unet11":
            self.explainer = UNet11Explainer(num_classes=num_classes, pretrained=pretrained)
        elif explainer_type == "unet16":
            self.explainer = UNet16Explainer(num_classes=num_classes, pretrained=pretrained)
        else:
            raise ValueError("explainer_type must be unet11 or unet16")

    def setup_classifier(self, classifier_type, classifier_checkpoint, fix_classifier, num_classes):
        self.classifier = Classifier(model=classifier_type, num_classes=num_classes, pretrained=False)

        if classifier_checkpoint is not None:
            self.classifier = self.classifier.load_from_checkpoint(classifier_checkpoint, model=classifier_type, num_classes=num_classes)
            if fix_classifier:
                self.classifier.freeze()

    def setup_losses(self, image_size, class_mask_min_area, class_mask_max_area):
        self.total_variation_conv = TotalVariationConv()

        self.classification_loss_fn = nn.CrossEntropyLoss()
        # self.classification_loss_fn = nn.BCEWithLogitsLoss()

        self.class_mask_area_loss_fn = ClassMaskAreaLoss(
            image_size=image_size,
            min_area=class_mask_min_area,
            max_area=class_mask_max_area
        )

    def setup_metrics(self, num_classes, metrics_threshold):
        self.train_metrics = SingleLabelMetrics(num_classes=num_classes)
        self.valid_metrics = SingleLabelMetrics(num_classes=num_classes)
        self.test_metrics = SingleLabelMetrics(num_classes=num_classes)
        self.train_segmentation_metrics = BinarySegmentationMetrics()
        self.valid_segmentation_metrics = BinarySegmentationMetrics()
        self.test_segmentation_metrics = BinarySegmentationMetrics()

    def forward(self, image, targets):
        segmentations = self.explainer(image)
        target_mask, non_target_mask = helper.extract_masks(segmentations, targets)
        inversed_target_mask = torch.ones_like(target_mask) - target_mask

        masked_image = target_mask.unsqueeze(1) * image
        inversed_masked_image = inversed_target_mask.unsqueeze(1) * image

        logits_mask = self.classifier(masked_image)
        logits_inversed_mask = self.classifier(inversed_masked_image)

        return logits_mask, logits_inversed_mask, target_mask, non_target_mask, segmentations


    def training_step(self, batch, batch_idx):
        image, mask, id = batch
        targets = torch.zeros((image.shape[0], self.num_classes)).to(image.device)
        targets[:, mask.amax(dim=(1,2))] = 1.0

        logits_mask, logits_inversed_mask, target_mask, non_target_mask, segmentations = self(image, targets)

        loss = 0
        classification_loss_mask = self.classification_loss_fn(logits_mask, targets)
        self.log('train_loss/classification_loss_mask', classification_loss_mask)
        loss += classification_loss_mask

        # classification_loss_inversed_mask = self.entropy_regularizer * entropy_loss(logits_inversed_mask)
        if self.use_inverse_classification_loss:
            healthy_targets = 1.0 - targets
            classification_loss_inversed_mask = self.entropy_regularizer * self.classification_loss_fn(logits_inversed_mask, healthy_targets)
            self.log('train_loss/classification_loss_inversed_mask', classification_loss_inversed_mask)
            loss += classification_loss_inversed_mask

        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(target_mask) + self.total_variation_conv(non_target_mask))
            self.log('train_loss/mask_variation_loss', mask_variation_loss)
            loss += mask_variation_loss

        if self.use_mask_area_loss:
            mask_area_loss = self.class_mask_area_loss_fn(segmentations, targets)
            self.log('train_loss/mask_area_constraint', mask_area_loss)
            mask_area_loss *= self.mask_area_constraint_regularizer
            mask_area_loss += self.mask_total_area_regularizer * target_mask.mean()
            mask_area_loss += self.mask_total_area_regularizer * (target_mask * (1 - target_mask)).mean()  # entropy
            mask_area_loss += self.ncmask_total_area_regularizer * non_target_mask.mean()
            mask_area_loss += self.ncmask_total_area_regularizer * (non_target_mask * (1 - non_target_mask)).mean()  # entropy
            self.log('train_loss/mask_area_loss', mask_area_loss)
            self.log('train_loss/mask_total_area', target_mask.mean())
            self.log('train_loss/ncmask_total_area', non_target_mask.mean())
            loss += mask_area_loss

        self.log('train_loss/total_loss', loss)
        self.train_metrics(logits_mask, targets)
        self.train_segmentation_metrics(target_mask, mask)

        if batch_idx == 0:
            self.visualize_results("train", "image", image[0], target_mask[0], non_target_mask[0])

        return loss

    def training_epoch_end(self, outs):
        metrics = self.train_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(f'train_metrics/{metric_name}', metric_value, prog_bar=True)
        self.train_metrics.reset()
        metrics = self.train_segmentation_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(f'train_segmentation_metrics/{metric_name}', metric_value, prog_bar=True)
        self.train_segmentation_metrics.reset()

    def validation_step(self, batch, batch_idx):
        image, mask, id = batch
        targets = torch.zeros((image.shape[0], self.num_classes)).to(image.device)
        targets[:, mask.amax(dim=(1,2))] = 1.0

        logits_mask, logits_inversed_mask, target_mask, non_target_mask, segmentations = self(image, targets)

        loss = 0
        classification_loss_mask = self.classification_loss_fn(logits_mask, targets)
        self.log('val_loss/classification_loss_mask', classification_loss_mask)
        loss += classification_loss_mask

        # classification_loss_inversed_mask = self.entropy_regularizer * entropy_loss(logits_inversed_mask)
        if self.use_inverse_classification_loss:
            healthy_targets = 1.0 - targets
            classification_loss_inversed_mask = self.entropy_regularizer * self.classification_loss_fn(logits_inversed_mask, healthy_targets)
            self.log('val_loss/classification_loss_inversed_mask', classification_loss_inversed_mask)
            loss += classification_loss_inversed_mask

        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(target_mask) + self.total_variation_conv(non_target_mask))
            self.log('val_loss/mask_variation_loss', mask_variation_loss)
            loss += mask_variation_loss

        if self.use_mask_area_loss:
            mask_area_loss = self.mask_area_constraint_regularizer * self.class_mask_area_loss_fn(segmentations, targets)
            self.log('val_loss/mask_area_constraint', mask_area_loss)
            mask_area_loss += self.mask_total_area_regularizer * target_mask.mean()
            mask_area_loss += self.ncmask_total_area_regularizer * non_target_mask.mean()
            self.log('val_loss/mask_area_loss', mask_area_loss)
            self.log('val_loss/mask_total_area', self.mask_total_area_regularizer * target_mask.mean())
            self.log('val_loss/ncmask_total_area', self.ncmask_total_area_regularizer * non_target_mask.mean())
            loss += mask_area_loss

        self.log('val_loss/total_loss', loss)
        self.valid_metrics(logits_mask, targets)
        self.valid_segmentation_metrics(target_mask, mask)

        if batch_idx == 0:
            self.visualize_results("val", "image", image[0], target_mask[0], non_target_mask[0])

        return loss

    def validation_epoch_end(self, outs):
        metrics = self.valid_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(f'val_metrics/{metric_name}', metric_value, prog_bar=True)
        self.valid_metrics.reset()
        metrics = self.valid_segmentation_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(f'val_segmentation_metrics/{metric_name}', metric_value, prog_bar=True)
        self.valid_segmentation_metrics.reset()

    def visualize_results(self, fold, id, image, target_mask, non_target_mask):
        self.logger.experiment.add_image(f'{fold}_visualization/{id}', self.unnorm(image.clone().detach()), self.global_step)
        self.logger.experiment.add_image(f'{fold}_visualization/target_mask', target_mask.unsqueeze(0), self.global_step)
        # self.logger.experiment.add_image(f'{fold}_visualization/non_target_mask', non_target_mask.unsqueeze(0), self.global_step)

    def test_step(self, batch, batch_idx):
        image, mask, id = batch
        targets = torch.zeros((image.shape[0], self.num_classes)).to(image.device)
        targets[:, mask.amax(dim=(1,2))] = 1.0
        # targets = get_targets_from_annotations(annotations, dataset=self.dataset)
        logits_mask, logits_inversed_mask, target_mask, non_target_mask, segmentations = self(image, targets)

        if self.save_masked_images and image.size()[0] == 1:
            save_masked_image(image, target_mask, Path(self.save_path) / "masked_images" / id[0])

        if self.save_masks and image.size()[0] == 1:
            save_mask(target_mask, Path(self.save_path) / id[0])

        loss = 0
        classification_loss_mask = self.classification_loss_fn(logits_mask, targets)
        loss += classification_loss_mask

        if self.use_inverse_classification_loss:
            healthy_targets = 1.0 - targets
            classification_loss_inversed_mask = self.entropy_regularizer * self.classification_loss_fn(logits_inversed_mask, healthy_targets)
            loss += classification_loss_inversed_mask

        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(target_mask) + self.total_variation_conv(non_target_mask))
            loss += mask_variation_loss

        if self.use_mask_area_loss:
            mask_area_loss = self.mask_area_constraint_regularizer * self.class_mask_area_loss_fn(segmentations, targets)
            mask_area_loss += self.mask_total_area_regularizer * target_mask.mean()
            mask_area_loss += self.ncmask_total_area_regularizer * non_target_mask.mean()
            loss += mask_area_loss

        self.log('test_loss', loss)
        self.test_metrics(logits_mask, targets)
        self.test_segmentation_metrics(target_mask, mask)

    def test_epoch_end(self, outs):
        self.log('test_metrics', self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.save(model="explainer", classifier_type=self.classifier_type)
        self.log('test_segmentation_metrics', self.test_segmentation_metrics.compute(), prog_bar=True)
        self.test_segmentation_metrics.save(model="explainer", classifier_type=self.classifier_type)

        self.test_metrics.reset()
        self.test_segmentation_metrics.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
