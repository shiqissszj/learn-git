# run_info_vae.py
# -----------------------------------------------------------
#  训练 InfoVAE（含多模态重构）以学习 CirCor PCG 潜向量
#  ⚙️  默认使用 1 kHz × 2 500 点滑窗 + 50 % overlap
# -----------------------------------------------------------

import os
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from torch.utils.data import DataLoader, random_split

from PCG_dataset import PCGDataset
from mymodels.info_pcg import InfoVAE


# ---------------------------------------------------------------- #
#                       配 置 项                                     #
# ---------------------------------------------------------------- #
DATA_DIR = "/data/myj/circor_heart_sound/training_data/"           # ← ← ←  改成你实际路径
BATCH_SIZE = 64
TARGET_LENGTH = 2500        # 2 500 sample @ 1 kHz  ≈ 2.5 s
HOP_LENGTH = 1250           # 50 % overlap
SAMPLE_RATE = 1000
MAX_EPOCHS = 150
LATENT_DIM = 64
LEARNING_RATE = 1e-3
SEED = 42


def get_dataloaders():
    """构建训练 / 验证 DataLoader"""
    dataset = PCGDataset(
        folder_path=DATA_DIR,
        target_length=TARGET_LENGTH,
        hop_length=HOP_LENGTH,
        sample_rate=SAMPLE_RATE,
        band=(10, 200),
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, train_size, val_size


def build_trainer(logger: TensorBoardLogger) -> Trainer:
    """Lightning Trainer（含 Checkpoint / Early-Stopping / LR 监控）"""
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        filename="epoch{epoch:03d}-val_loss{val_loss:.4f}",
    )
    es_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        verbose=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0, 3],
        max_epochs=MAX_EPOCHS,
        logger=logger,
        callbacks=[ckpt_cb, es_cb, lr_cb],
        precision="16-mixed" if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
    )
    return trainer


def main():
    # reproducibility ---------------------------------------------------- #
    seed_everything(SEED, workers=True)

    # data ----------------------------------------------------------------#
    train_loader, val_loader, train_sz, val_sz = get_dataloaders()

    # model ---------------------------------------------------------------#
    model = InfoVAE(
        input_dim=TARGET_LENGTH,
        latent_dim=LATENT_DIM,
        learning_rate=LEARNING_RATE,
        sample_rate=SAMPLE_RATE,
        train_dataset_size=train_sz,
        val_dataset_size=val_sz,
    )

    # logger --------------------------------------------------------------#
    logger = TensorBoardLogger("logs", name="info_vae_pcg")

    # trainer -------------------------------------------------------------#
    trainer = build_trainer(logger)

    # training ------------------------------------------------------------#
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
