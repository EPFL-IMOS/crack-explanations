import os
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from tqdm import tqdm
import time

import data
from model.activation_map.map_utils import map_factory, process_save_cams
from model.classifier import Classifier
from transforms import get_val_transform, augmentation_smoothing, augmentation_intersection
from functools import partial

from transforms import UnNormalize, MEAN, STD


def run_map(
    cam_method: str = "deeplift",
    dataset: str = "DICCracksDataset",
    num_classes: int = 2,
    data_path: os.PathLike = "/home/florent/imos/data/DIC_crack_dataset",
    fold: str = "val",
    batch_size: int = 32,
    workers: int = 8,
    imagesize: int = 256,
    model_type: str = "resnet18_deeplift",
    model_path: os.PathLike = "checkpoints/output/dic-resnet18-pretrained-test/epoch=5-val_accuracy=0.95.ckpt",
    output_dir: os.PathLike = "output/dic-resnet18-deeplift-pretrained-test",
    aug_smooth: str = None,
    **kwargs,
):

    Dataset = data.__dict__[dataset]

    dataset = Dataset(
        root=data_path,
        fold=fold,
        transform=get_val_transform(add_inverse=("bcos" in model_type), size=imagesize),
        segmentation=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, drop_last=False
    )

    if cam_method != "baseline":
        model = Classifier.load_from_checkpoint(model_path, model=model_type, num_classes=num_classes).eval()
    else:
        model = None

    # Fix for DeepLift
    if cam_method == "deeplift":
        for module in model.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False

    map, run = map_factory(method=cam_method, model=model, **kwargs)
    output_path = Path(output_dir)

    if aug_smooth == "average":
        cam_method += "_augsmooth"
    elif aug_smooth == "intersection":
        cam_method += "_augprod"
    cam_path = (
        output_path / "activation_map" / fold / f"{cam_method}_{model_type}"
    )  # TODO: Track folder
    cam_path.mkdir(exist_ok=True, parents=True)

    # unnorm = UnNormalize(MEAN, STD)

    n = 0
    t, t0 = 0, 0

    for image, _, id in tqdm(dataloader):

        t0 = time.time()
        if aug_smooth == "average":
            attribution = augmentation_smoothing(partial(run, map=map, model=model, **kwargs), image)
        elif aug_smooth == "intersection":
            attribution = augmentation_intersection(partial(run, map=map, model=model, **kwargs), image)
        else:
            attribution = run(map=map, image=image, model=model, **kwargs)
        t += time.time() - t0
        n += 1
        # if n > 100:
        #     break

        # un-normalize image
        # image = unnorm(image)
        image = image.numpy()
        id = np.array(id)

        process_save_cams(
            image=image, grayscale_cam=attribution, id_region=id, save_path=cam_path
        )
    print(f"TIME PER IMAGE: {t/n}")

if __name__ == "__main__":
    Fire(run_map)
