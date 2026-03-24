import os
import sys
import os.path as osp
from datetime import datetime

root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import torch
from torch.utils.data import DataLoader
from src.model.detector import CenterPoint
from src.dataset import ViewOfDelft, collate_vod_batch


def _metric_to_float(metric_value):
    if metric_value is None:
        return None
    if isinstance(metric_value, torch.Tensor):
        return float(metric_value.item())
    return float(metric_value)


def _format_metric(metric_value):
    metric_value = _metric_to_float(metric_value)
    if metric_value is None:
        return "-"
    return f"{metric_value:.6f}"


def _format_report_path(path):
    if not path:
        return "-"
    if osp.isabs(path):
        path = osp.relpath(path, root)
    return f"`{path}`"


def append_training_report(
    cfg,
    report_path,
    run_timestamp,
    run_name,
    run_output_dir,
    roi_checkpoint_callback,
    entire_checkpoint_callback,
):
    report_dir = osp.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    if not osp.exists(report_path):
        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write("# Training Report\n\n")
            report_file.write(
                "| Timestamp | Exp ID | Run Name | Model | Epochs | Batch | LR | WD | Val Every | Best ROI mAP | Best ROI Model | Best Entire mAP | Best Entire Model | Run Dir |\n"
            )
            report_file.write(
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            )

    optimizer_cfg = cfg.model.get("optimizer", {})
    row = (
        f"| {run_timestamp}"
        f" | {cfg.exp_id}"
        f" | {run_name}"
        f" | {cfg.model.name}"
        f" | {cfg.epochs}"
        f" | {cfg.batch_size}"
        f" | {optimizer_cfg.get('lr', '-')}"
        f" | {optimizer_cfg.get('weight_decay', '-')}"
        f" | {cfg.val_every}"
        f" | {_format_metric(roi_checkpoint_callback.best_model_score)}"
        f" | {_format_report_path(roi_checkpoint_callback.best_model_path)}"
        f" | {_format_metric(entire_checkpoint_callback.best_model_score)}"
        f" | {_format_report_path(entire_checkpoint_callback.best_model_path)}"
        f" | {_format_report_path(run_output_dir)} |\n"
    )

    with open(report_path, "a", encoding="utf-8") as report_file:
        report_file.write(row)


@hydra.main(version_base=None, config_path="../config", config_name="train")
def train(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.exp_id}_{run_timestamp}"
    run_output_dir = osp.join(cfg.output_dir, run_timestamp)
    checkpoint_dir = osp.join(run_output_dir, "checkpoints")
    report_path = cfg.report_path
    if not osp.isabs(report_path):
        report_path = osp.join(root, report_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Run output directory: {run_output_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Report file: {report_path}")

    dataset_cfg = OmegaConf.to_container(cfg.model.get("dataset", {}), resolve=True) or {}
    train_dataset = ViewOfDelft(data_root=cfg.data_root, split="train", **dataset_cfg)
    val_dataset = ViewOfDelft(data_root=cfg.data_root, split="val", **dataset_cfg)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        collate_fn=collate_vod_batch,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=cfg.num_workers,
        shuffle=False,
        collate_fn=collate_vod_batch,
    )
    model = CenterPoint(cfg.model)
    roi_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="roi-ep{epoch}-" + cfg.exp_id,
        save_last=True,
        monitor="validation/ROI/mAP",
        mode="max",
        auto_insert_metric_name=False,
        save_top_k=cfg.save_top_model,
    )
    entire_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="entire-ep{epoch}-" + cfg.exp_id,
        save_last=False,
        monitor="validation/entire_area/mAP",
        mode="max",
        auto_insert_metric_name=False,
        save_top_k=cfg.save_top_model,
    )
    callbacks = [
        roi_checkpoint_callback,
        entire_checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = WandbLogger(
        save_dir=osp.join(run_output_dir, "wandb_logs"),
        project="amp",
        name=run_name,
        log_model=False,
    )
    logger.watch(model, log_graph=False)

    trainer = L.Trainer(
        logger=logger,
        log_every_n_steps=cfg.log_every,
        accelerator="gpu",
        devices=cfg.gpus,
        check_val_every_n_epoch=cfg.val_every,
        strategy="auto",
        callbacks=callbacks,
        max_epochs=cfg.epochs,
        sync_batchnorm=cfg.sync_bn,
        enable_model_summary=True,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.checkpoint_path,
    )
    append_training_report(
        cfg=cfg,
        report_path=report_path,
        run_timestamp=run_timestamp,
        run_name=run_name,
        run_output_dir=run_output_dir,
        roi_checkpoint_callback=roi_checkpoint_callback,
        entire_checkpoint_callback=entire_checkpoint_callback,
    )
    print(f"Appended run summary to {report_path}")
    # wandb.finish()


if __name__ == "__main__":
    train()
