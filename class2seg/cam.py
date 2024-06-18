import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from fire import Fire
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from tqdm import tqdm

import data
from model.activation_map.map_utils import (
    layer_selection,
    map_factory,
    process_save_cams,
)
from model.classifier import Classifier
from transforms import get_val_transform, augmentation_smoothing
from functools import partial


def batch_cam(
    batch_image: torch.Tensor,
    batch_target: List[int],
    cam: BaseCAM,
    **kwargs,
) -> np.ndarray:

    targets = [ClassifierOutputTarget(target) for target in batch_target]
    grayscale_cam = cam(input_tensor=batch_image, targets=targets, **kwargs)

    return grayscale_cam


def run_cam(
    cam_method: str = "gradcam",
    dataset: str = "DICCracksDataset",
    num_classes: int = 2,
    data_path: os.PathLike = "/home/florent/imos/data/DIC_crack_dataset",
    fold: str = "val",
    batch_size: int = 32,
    workers: int = 8,
    model_type: str = "resnet18",
    model_path: os.PathLike = "checkpoints/output/dic-resnet18-pretrained-test/epoch=5-val_accuracy=0.95.ckpt",
    layer_type: str = "resnet_multiple",
    output_dir: os.PathLike = "output/dic-resnet18-pretrained-test",
    aug_smooth: bool = False,
    **kwargs,
):

    Dataset = data.__dict__[dataset]

    dataset = Dataset(
        root=data_path,
        fold=fold,
        transform=get_val_transform(),
        segmentation=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, drop_last=False
    )

    model = Classifier.load_from_checkpoint(model_path, model=model_type, num_classes=num_classes).eval()
    layers = layer_selection(model=model, layer=layer_type)

    cam, _ = map_factory(
        method=cam_method, model=model, target_layers=layers, use_cuda=torch.cuda.is_available()#, **kwargs
    )
    output_path = Path(output_dir)

    if aug_smooth:
        cam_method += "_augsmooth"
    cam_path = (
        output_path / "activation_map" / fold / f"{cam_method}_{model_type}_{layer_type}"
    )
    cam_path.mkdir(exist_ok=True, parents=True)

    for image, target, id in tqdm(dataloader):

        if aug_smooth:
            grayscale_cam = augmentation_smoothing(partial(batch_cam, batch_target=target, cam=cam), image)
        else:
            grayscale_cam = batch_cam(batch_image=image, batch_target=target, cam=cam)

        image = image.numpy()
        id = np.array(id)

        process_save_cams(
            image=image, grayscale_cam=grayscale_cam, id_region=id, save_path=cam_path
        )


if __name__ == "__main__":
    Fire(run_cam)
