import os
import sys
root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)
    
import hydra
from  omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import torch
from torch.utils.data import DataLoader

from src.model.detector import CenterPoint
from src.dataset import ViewOfDelft, collate_vod_batch


def _resolve_dataset_cfg(checkpoint_params, radar_sweeps_override=None):
    model_cfg = checkpoint_params.get("config", {})
    dataset_cfg_node = model_cfg.get("dataset") if model_cfg is not None else {}
    if dataset_cfg_node is None:
        dataset_cfg = {}
    elif OmegaConf.is_config(dataset_cfg_node):
        dataset_cfg = OmegaConf.to_container(dataset_cfg_node, resolve=True) or {}
    else:
        dataset_cfg = dict(dataset_cfg_node)

    if radar_sweeps_override is not None:
        dataset_cfg["radar_sweeps"] = int(radar_sweeps_override)
    return dataset_cfg


@hydra.main(version_base=None, config_path='../config', config_name="eval")
def eval(cfg: DictConfig) -> None:
    print('Evaluating model...')
    L.seed_everything(cfg.seed, workers=True)

    checkpoint = torch.load(cfg.checkpoint_path, weights_only=False)
    checkpoint_params = DictConfig(checkpoint["hyper_parameters"])
    print('checkpoint params:', checkpoint_params.keys())
    print('checkpoint params cfg:', checkpoint_params.config.keys())

    dataset_cfg = _resolve_dataset_cfg(
        checkpoint_params,
        radar_sweeps_override=cfg.radar_sweeps,
    )
    print('eval dataset cfg:', dataset_cfg)
    val_dataset = ViewOfDelft(data_root=cfg.data_root, split='val', **dataset_cfg)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                num_workers=cfg.num_workers,
                                shuffle=False,
                                collate_fn=collate_vod_batch)
    
    model = CenterPoint.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path)
    model.eval()
    
    trainer = L.Trainer(
        accelerator="gpu",
        devices=cfg.gpus,
    )
    
    trainer.validate(model = model,
                     dataloaders=val_dataloader)
                     
    
if __name__ == '__main__':
    eval()
