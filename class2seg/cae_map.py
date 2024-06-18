import os
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from tqdm import tqdm
import time

import data
from model.activation_map.map_utils import process_save_cams
from model.autoencoder import Autoencoder
from transforms import get_val_transform, augmentation_smoothing


def run_map(
    cam_method: str = "cae",
    dataset: str = "DICCracksDataset",
    data_path: os.PathLike = "/home/florent/imos/data/DIC_crack_dataset",
    fold: str = "test",
    batch_size: int = 16,
    workers: int = 8,
    model_type: str = "cae11",
    latent_dim: int = 50,
    deconv: bool = False,
    model_path: os.PathLike = "checkpoints/output/dic-cae11/last.ckpt",
    output_dir: os.PathLike = "output/dic-cae11",
    aug_smooth: bool = False,
    **kwargs,
):

    Dataset = data.__dict__[dataset]

    dataset = Dataset(
        root=data_path,
        fold=fold,
        transform=get_val_transform(add_inverse=("bcos" in model_type)),
        segmentation=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, drop_last=False
    )

    model = Autoencoder.load_from_checkpoint(model_path, model=model_type, in_channels=3, latent_dim=latent_dim, is_deconv=deconv).eval()

    output_path = Path(output_dir)

    if aug_smooth:
        cam_method += "_augsmooth"
    cam_path = (
        output_path / "activation_map" / fold / f"{cam_method}_{model_type}"
    )
    cam_path.mkdir(exist_ok=True, parents=True)

    n = 0
    t, t0 = 0, 0

    for image, _, id in tqdm(dataloader):

        t0 = time.time()
        res = torch.square(model(image) - image).sum(dim=1)
        if aug_smooth:
            attribution = augmentation_smoothing(lambda x: res.squeeze(1).detach().numpy(), image)
        else:
            attribution = res.squeeze(1).detach().numpy()
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
