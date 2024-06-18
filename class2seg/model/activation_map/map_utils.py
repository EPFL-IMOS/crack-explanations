import os
from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from captum.attr import (
    DeepLift,
    DeepLiftShap,
    GradientShap,
    IntegratedGradients,
    InputXGradient
)
from PIL import Image
from pytorch_grad_cam import (
    AblationCAM,
    EigenGradCAM,
    FullGrad,
    GradCAM,
    GradCAMPlusPlus,
)
from pytorch_grad_cam.base_cam import BaseCAM
from zennit.composites import Composite, EpsilonGammaBox
from zennit.rules import AlphaBeta
from model.classification.explLRP.ViT_explanation_generator import LRP as LRPViT
import cv2
from functools import partial

from model.classifier import Conv2dAlphaBeta
from transforms import MEAN, STD


def lrpVGG11(rules: str, **kwargs) -> Composite:

    if rules == "segmentation":
        composite = EpsilonGammaBox(
            low=0.0,
            high=1.0,
            gamma=0.25,
            layer_map=[(Conv2dAlphaBeta, AlphaBeta(alpha=2.0, beta=1.0))],
        )
    else:
        raise NotImplementedError()

    return composite


def lrpResNet(rules: str, **kwargs) -> Composite:

    composite = EpsilonGammaBox(
        low=0.0,
        high=1.0,
        gamma=0.25,
        # layer_map=[(Conv2dAlphaBeta, AlphaBeta(alpha=2.0, beta=1.0))],
    )

    return composite


def lrpViT(rules: str, **kwargs) -> LRPViT:

    return LRPViT


def map_factory(
    method: str, **kwargs
) -> Tuple[Union[Composite, BaseCAM, DeepLift, IntegratedGradients], Optional[Callable]]:

    if method == "gradcam":
        return GradCAM(**kwargs), None
    elif method == "gradcam++":
        return GradCAMPlusPlus(**kwargs), None
    elif method == "ablationcam":
        return AblationCAM(**kwargs), None
    elif method == "fullgrad":
        return FullGrad(**kwargs), None
    elif method == "eigengrad":
        return EigenGradCAM(**kwargs), None
    elif method == "deeplift":
        assert "model" in kwargs
        return DeepLift(model=kwargs["model"]), partial(run_DeepLift, **kwargs)
    elif method == "lrpVGG":
        return lrpVGG11(**kwargs), run_LRP
    elif method == "lrpViT":
        return lrpViT(**kwargs), run_LRPViT
    elif method == "lrpResNet":
        return lrpResNet(**kwargs), run_LRP
    elif method == "inputxgrad":
        return InputXGradient(forward_func=kwargs["model"]), run_InputXGrad
    elif method == "intgrad":
        assert "model" in kwargs
        return IntegratedGradients(forward_func=kwargs["model"]), partial(run_IntGrad, **kwargs)
    elif method == "gradientshap":
        assert "model" in kwargs
        return GradientShap(forward_func=kwargs["model"]), partial(run_GradientShap, **kwargs)
    elif method == "deepliftshap":
        assert "model" in kwargs
        return DeepLiftShap(model=kwargs["model"]), partial(run_DeepLiftShap, **kwargs)
    elif method == "bcos_grad":
        return None, run_BcosGrad
    elif method == "baseline":
        return None, run_baseline
    else:
        raise NotImplementedError()


def layer_selection(model: nn.Module, layer: str):

    if layer == "resnet_common":
        return [model.model.layer4[-1]]
    elif layer == "resnet_low":
        return [model.model.layer1[-1]]
    elif layer == "resnet_mid":
        return [model.model.layer2[-1]]
    elif layer == "resnet_multiple":
        return [model.model.layer1[-1], model.model.layer2[-1]]
    elif layer == "resnet_all":
        return [
            model.model.layer1[-1],
            model.model.layer2[-1],
            model.model.layer3[-1],
            model.model.layer4[-1],
        ]
    elif layer == "vit_commmon":
        return [model.model.blocks[-1].norm1]
    elif layer == "vgg_common":
        return [model.model.features[18]]
    elif layer == "vgg_low":
        return [model.model.features[3]]
    elif layer == "vgg_mid":
        return [model.model.features[8]]
    elif layer == "vgg_mid2":
        return [model.model.features[13]]
    elif layer == "vgg_mid3":
        return [model.model.features[16]]
    elif layer == "vgg_all2":
        return [model.model.features[i] for i in (3, 7, 14, 21)]
    # conv layers: 0, 3, 6, 8, 11, 13, 16, 18
    # with BN:  3, 7, 14, 21, 28
    else:
        raise NotImplementedError()


def run_DeepLift(map: DeepLift, image: torch.Tensor, baseline: str = None, **kwargs) -> np.ndarray:
    if baseline is not None:
        with open(baseline, "rb") as f:
            baseline = torch.load(f)[..., :image.shape[-2], :image.shape[-1]]  # 224px for ViT
            # normalization
            # baseline = TF.normalize(baseline, MEAN, STD)
    attribution = map.attribute(image, target=1, baselines=baseline)

    attribution = attribution.detach().numpy()
    attribution = np.mean(attribution, axis=1)

    return attribution


def run_LRP(
    map: Composite, image: torch.Tensor, model: nn.Module, **kwargs
) -> np.ndarray:

    with map.context(model) as modified_model:
        # execute the hooked/modified model
        image.requires_grad = True
        output = modified_model(image)

        # compute the attribution via the gradient
        (attribution,) = torch.autograd.grad(
            output, image, grad_outputs=torch.ones_like(output)
        )
    image.requires_grad = False
    attribution = attribution.detach().numpy()
    attribution = np.mean(attribution, axis=1)
    return attribution


def run_LRPViT(map: LRPViT, model: nn.Module, image: torch.Tensor, **kwargs):
    attribution_generator = map(model.model.cuda())
    attribution = attribution_generator.generate_LRP(image.cuda(), method="transformer_attribution", index=1).detach()
    attribution = attribution.view((1, 14, 14)).cpu().numpy()
    # attribution = np.mean(attribution.cpu().numpy(), axis=1)
    return attribution


def run_InputXGrad(map: InputXGradient, image: torch.Tensor, **kwargs) -> np.ndarray:

    attribution = map.attribute(image, target=1)
    attribution = attribution.detach().numpy()
    attribution = np.mean(attribution, axis=1)

    return attribution


def run_IntGrad(map: IntegratedGradients, image: torch.Tensor, baseline: str = None, **kwargs) -> np.ndarray:
    if baseline is not None:
        with open(baseline, "rb") as f:
            baseline = torch.load(f)[..., :image.shape[-2], :image.shape[-1]]  # 224px for ViT
            # # normalization
            # baseline = TF.normalize(baseline, MEAN, STD)
    attribution = map.attribute(image, target=1, baselines=baseline)
    attribution = attribution.detach().numpy()
    attribution = np.mean(attribution, axis=1)

    return attribution


def run_GradientShap(map: GradientShap, image: torch.Tensor, baseline: str = None, **kwargs) -> np.ndarray:
    # We define a distribution of baselines and draw `n_samples` from that
    # distribution in order to estimate the expectations of gradients across all baselines
    if baseline is not None:
        with open(baseline, "rb") as f:
            baseline_dist = torch.load(f)[..., :image.shape[-2], :image.shape[-1]]  # 224px for ViT
            # # normalization
            # for i in range(baseline_dist.size(0)):
            #     baseline_dist[i] = TF.normalize(baseline_dist[i], MEAN, STD)
    else:
        baseline_dist = torch.randn((100, *image.shape[1:])) * 0.001
    attribution = map.attribute(image, stdevs=0.09, n_samples=4, baselines=baseline_dist, target=1)
    attribution = attribution.detach().numpy()
    attribution = np.mean(attribution, axis=1)

    return attribution


def run_DeepLiftShap(map: DeepLiftShap, image: torch.Tensor, baseline: str = None, **kwargs) -> np.ndarray:
    if baseline is not None:
        with open(baseline, "rb") as f:
            baseline_dist = torch.load(f)[:5, :, :image.shape[-2], :image.shape[-1]]  # 224px for ViT
            # # normalization
            # for i in range(baseline_dist.size(0)):
            #     baseline_dist[i] = TF.normalize(baseline_dist[i], MEAN, STD)
    else:
        baseline_dist = torch.randn((10, *image.shape[1:])) * 0.001
    with torch.no_grad():
        attribution = map.attribute(image, baselines=baseline_dist, target=1)
        attribution = attribution.detach().numpy()
        attribution = np.mean(attribution, axis=1)

    return attribution


def run_BcosGrad(model: nn.Module, image: torch.Tensor, smooth: float = None, alpha_percentile: float = 99.5, **kwargs):
    """
    Computing color image from dynamic linear mapping of B-cos models.
    Args:
        img: Original input image (encoded with 6 color channels)
        linear_mapping: linear mapping W_{1\rightarrow l} of the B-cos model
        smooth: kernel size for smoothing the alpha values
        alpha_percentile: cut-off percentile for the alpha value
    Returns:
        image explanation of the B-cos model
    """
    img = torch.autograd.Variable(image, requires_grad=True)
    out = model(img).max()
    model.zero_grad()
    out.backward()
    model.zero_grad()

    linear_mapping = img.grad.data[0]

    # shape of image and linmap is [C, H, W], summing over first dimension gives the contribution map per location
    contribs = (img[0] * linear_mapping).sum(0, keepdim=True)
    contribs = contribs[0]

    # Set alpha value to the strength (L2 norm) of each location's gradient
    alpha = (linear_mapping.norm(p=2, dim=0, keepdim=True))
    # Only show positive contributions
    alpha = torch.where(contribs[None] < 0, torch.zeros_like(alpha) + 1e-12, alpha)
    if smooth:
        alpha = F.avg_pool2d(alpha, smooth, stride=1, padding=(smooth-1)//2)
    alpha = alpha.numpy()
    alpha = (alpha / np.percentile(alpha, alpha_percentile)).clip(0, 1)

    return alpha


def run_baseline(image: torch.Tensor, **kwargs):
    """
    Dummy baseline using raw inverted grayscale image.
    """
    img_grayscale = cv2.cvtColor(image[0].numpy().transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)

    return [1 - img_grayscale]


def process_save_cams(
    image: np.ndarray, grayscale_cam: np.ndarray, id_region: str, save_path: os.PathLike
):

    for img, g_cam, id_r in zip(image, grayscale_cam, id_region):

        img_ar = img.transpose(1, 2, 0)[:, :, :3]  # limit to 3 channels
        image_save = np.uint8(img_ar * 255)
        image_save = Image.fromarray(image_save)

        (save_path / id_r).mkdir(exist_ok=True, parents=True)
        image_save.save(save_path / id_r / f"rgb_img.jpg")

        if g_cam.shape != (256, 256):
            g_cam = cv2.resize(g_cam, (image.shape[2], image.shape[3]), interpolation=cv2.INTER_NEAREST)

        np.save(save_path / id_r / "cam_grayscale", g_cam)
        plt.imsave(save_path / id_r / "cam.jpg", g_cam)
