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


def build_dataset(data_root, split, model_cfg):
    return ViewOfDelft(
        data_root=data_root,
        split=split,
        num_sweeps=model_cfg.get('num_sweeps', 1),
        use_camera=model_cfg.get('use_camera', False),
        image_size=model_cfg.get('image_size', None),
    )


@hydra.main(version_base=None, config_path='../config', config_name="test")
def eval(cfg: DictConfig) -> None:
    print('Evaluating model...')
    L.seed_everything(cfg.seed, workers=True)

    checkpoint = torch.load(cfg.checkpoint_path,weights_only=False)
    checkpoint_params = DictConfig(checkpoint["hyper_parameters"])
    print('checkpoint params:', checkpoint_params.keys())
    print('checkpoint params cfg:', checkpoint_params.config.keys())
    model_cfg = checkpoint_params.config

    test_dataset = build_dataset(cfg.data_root, 'test', model_cfg)
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=1, 
                                num_workers=cfg.num_workers, 
                                shuffle=False,
                                collate_fn=collate_vod_batch)
    
    model = CenterPoint.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path)
    model.inference_mode = 'test'
    model.save_results = True
    model.eval()
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
    )
    
    trainer.validate(model = model, dataloaders=test_dataloader)
                     
    
if __name__ == '__main__':
    eval()
