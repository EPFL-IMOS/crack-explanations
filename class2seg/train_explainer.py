import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import data
from model.explainer_classifier import ExplainerClassifier
from transforms import get_train_transform, get_val_transform
from utils.argparser import get_parser, write_config_file


def main(args):

    pl.seed_everything(args.seed)

    workers = 8
    Dataset = data.__dict__[args.dataset]

    ds = Dataset(
        root=args.data_path,
        fold="train",
        transform=get_train_transform(add_inverse=("bcos" in args.classifier_type)),
        segmentation=True,  # only for evaluation
        positive_only=True
    )

    val_ds = Dataset(
        root=args.data_path,
        fold="val",
        transform=get_val_transform(add_inverse=("bcos" in args.classifier_type)),
        segmentation=True,
        positive_only=True
    )

    test_ds = Dataset(
        root=args.data_path,
        fold="test",
        transform=get_val_transform(add_inverse=("bcos" in args.classifier_type)),
        segmentation=True,
        positive_only=True
    )

    image_size = ds[0][0].shape[1]

    run_name = args.output_dir
    logger = TensorBoardLogger(save_dir=run_name)

    checkpointer = ModelCheckpoint(
        dirpath=f"checkpoints/{run_name}",
        filename="{epoch}-{val_loss/total_loss:.2f}",
        monitor="val_loss/total_loss",
        mode="min",
        save_last=True
    )

    model = ExplainerClassifier(
        num_classes=args.num_classes,
        image_size=image_size,
        classifier_type=args.classifier_type,
        classifier_checkpoint=args.classifier_checkpoint,
        fix_classifier=args.fix_classifier,
        explainer_type=args.explainer_type,
        learning_rate=args.learning_rate, 
        class_mask_min_area=args.class_mask_min_area,
        class_mask_max_area=args.class_mask_max_area,
        use_inverse_classification_loss=args.use_inverse_classification_loss,
        entropy_regularizer=args.entropy_regularizer,
        use_mask_variation_loss=args.use_mask_variation_loss, 
        mask_variation_regularizer=args.mask_variation_regularizer,
        use_mask_area_loss=args.use_mask_area_loss,
        mask_area_constraint_regularizer=args.mask_area_constraint_regularizer, 
        mask_total_area_regularizer=args.mask_total_area_regularizer,
        ncmask_total_area_regularizer=args.ncmask_total_area_regularizer,
        dilate_mask=args.dilate_mask,
        topo_explainer=args.topo_explainer,
        in_distribution=args.in_distribution,
        baseline=args.baseline,
        metrics_threshold=args.metrics_threshold, 
        save_masked_images=args.save_masked_images,
        save_masks=args.save_masks,
        save_all_class_masks=args.save_all_class_masks,
        save_path=args.save_path
    )

    if args.explainer_classifier_checkpoint is not None:
        model = model.load_from_checkpoint(
            args.explainer_classifier_checkpoint,
            num_classes=args.num_classes,
            image_size=image_size,
            classifier_type=args.classifier_type,
            classifier_checkpoint=args.classifier_checkpoint,
            fix_classifier=args.fix_classifier,
            explainer_type=args.explainer_type,
            learning_rate=args.learning_rate, 
            class_mask_min_area=args.class_mask_min_area,
            class_mask_max_area=args.class_mask_max_area,
            entropy_regularizer=args.entropy_regularizer,
            use_mask_variation_loss=args.use_mask_variation_loss, 
            mask_variation_regularizer=args.mask_variation_regularizer,
            use_mask_area_loss=args.use_mask_area_loss,
            mask_area_constraint_regularizer=args.mask_area_constraint_regularizer, 
            mask_total_area_regularizer=args.mask_total_area_regularizer,
            ncmask_total_area_regularizer=args.ncmask_total_area_regularizer,
            dilate_mask=args.dilate_mask,
            topo_explainer=args.topo_explainer,
            in_distribution=args.in_distribution,
            baseline=args.baseline,
            metrics_threshold=args.metrics_threshold, 
            save_masked_images=args.save_masked_images,
            save_masks=args.save_masks,
            save_all_class_masks=args.save_all_class_masks,
            save_path=args.save_path
        )

    train_loader = torch.utils.data.DataLoader(
        ds, batch_size=args.train_batch_size, shuffle=True, num_workers=workers, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.val_batch_size, num_workers=workers, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.test_batch_size, num_workers=workers, drop_last=False
    )

    # Define Early Stopping condition
    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss",
    #     min_delta=args.early_stop_min_delta,
    #     patience=args.early_stop_patience,
    #     verbose=False,
    #     mode="min"
    # )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger = logger,
        callbacks = [checkpointer],
        gpus = [0] if torch.cuda.is_available() else 0,
        # checkpoint_callback = args.checkpoint_callback,
        log_every_n_steps=1
    )

    if args.train_model:
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(model=model, dataloaders=test_loader)
    else:
        if args.eval_fold == "train":
            train_ds = Dataset(
                root=args.data_path,
                fold="train",
                transform=get_val_transform(add_inverse=("bcos" in args.classifier_type)),
                segmentation=True,  # only for evaluation
                positive_only=True
            )
            test_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=args.test_batch_size, num_workers=workers, drop_last=False
            )
        elif args.eval_fold == "val":
            test_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=args.test_batch_size, num_workers=workers, drop_last=False
            )
        trainer.test(model=model, dataloaders=test_loader)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.arg_log:
        write_config_file(args)
    main(args)
