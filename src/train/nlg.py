import argparse

import train.litmodules as lms
import train.datamodules as dms

import wandb

import pytorch_lightning as pl

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelPruning,
    QuantizationAwareTraining,
)
from pytorch_lightning.loggers import WandbLogger


def main(args):
    dict_args = vars(args)

    wandb.login()

    if args.ckpt_path is None:
        model = lms.LitT5(**dict_args)
    else:
        model = lms.LitT5.load_from_checkpoint(args.ckpt_path)

    coqa = dms.EmpChat(**dict_args)

    wandb_logger = WandbLogger(
        name=args.name,
        project=args.project_name,
        save_dir=args.ckpt_path,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
        dirpath=args.ckpt_dir,
        filename="{epoch}-{val_loss:.6f}",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        strict=False,
        verbose=False,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    def compute_amount(epoch):
        # the sum of all returned values need to be smaller than 1
        if epoch == 10:
            return 0.5

        elif epoch == 50:
            return 0.25

        elif 75 < epoch < 99:
            return 0.01

    pruning = ModelPruning("l1_unstructured", amount=compute_amount)
    quantization = QuantizationAwareTraining()

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            pruning,
            # quantization,
        ],
    )
    if args.auto_lr_find:
        trainer.tune(model, coqa)
    else:
        trainer.fit(model, coqa)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size, learning_rate doesn't change with batch size",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=16,
        help="metric batch size",
    )
    parser.add_argument(
        "--use_adafactor",
        action="store_true",
        default=False,
        help="flag for Adafactor",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="flag for gradient checkpointing",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="weight decay coeeficient",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers arg in DataLoader",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        help="wandb project name",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="experiment",
        help="wandb logger name",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=15,
        help="number of epochs",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="number of warmup steps",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=512,
        help="max input length",
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=64,
        help="max input length",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="KETI-AIR/ke-t5-base",
        help="huggingface model name or path",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="checkpointfile directory",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="checkpointfile path",
    )
    parser.add_argument(
        "--freeze_layers",
        nargs="*",
        default=[],
        help="list of layers to freeze",
    )
    parser.add_argument(
        "--constant_schedule",
        action="store_true",
        default=False,
        help="constant scheduler flag",
    )

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
