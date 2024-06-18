import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import data
from model.segmenter import Segmenter
from transforms import get_train_transform, get_val_transform

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="DICCracksDataset")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument(
        "--data-path", type=str, default="/home/florent/imos/data/DIC_crack_dataset"
    )
    parser.add_argument("--model", type=str, default="unet11")
    parser.add_argument("--imagesize", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-fold", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main(args):

    pl.seed_everything(args.seed)

    imagesize = args.imagesize  # in pixel
    crop_size = args.crop_size
    workers = args.workers
    model_type = args.model
    checkpoint = args.resume_from
    Dataset = data.__dict__[args.dataset]

    ds = Dataset(
        root=args.data_path,
        fold="train",
        transform=get_train_transform(add_inverse=("bcos" in model_type)),
        segmentation=True,
        positive_only=True
    )

    val_ds = Dataset(
        root=args.data_path,
        fold="val",
        transform=get_val_transform(add_inverse=("bcos" in model_type)),
        segmentation=True,
        positive_only=True
    )

    test_ds = Dataset(
        root=args.data_path,
        fold="test",
        transform=get_val_transform(add_inverse=("bcos" in model_type)),
        segmentation=True,
        positive_only=True
    )

    run_name = args.output_dir
    logger = TensorBoardLogger(save_dir=run_name)

    checkpointer = ModelCheckpoint(
        dirpath=f"checkpoints/{run_name}",
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    model = Segmenter(model=model_type, pretrained=args.pretrained, num_classes=args.num_classes)

    train_loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=workers, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=True, num_workers=workers, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, num_workers=workers, drop_last=False
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.device,
        callbacks=[checkpointer],
        logger=logger,
        fast_dev_run=False,
        resume_from_checkpoint=checkpoint,
        log_every_n_steps=10,
        val_check_interval=100,
        limit_val_batches=1
    )

    if args.eval:
        model = Segmenter.load_from_checkpoint(checkpoint, model=model_type, num_classes=args.num_classes).eval()
        if args.eval_fold == "train":
            train_ds = Dataset(
                root=args.data_path,
                fold="train",
                transform=get_val_transform(add_inverse=("bcos" in args.classifier_type)),
                segmentation=True,
                positive_only=True
            )
            test_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=args.batch_size, num_workers=workers, drop_last=False
            )
        elif args.eval_fold == "val":
            test_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=args.batch_size, num_workers=workers, drop_last=False
            )
        return trainer.test(model=model, dataloaders=test_loader)

    return trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main(parse_args())
