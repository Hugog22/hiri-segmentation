import argparse
import logging
import os
import random
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW

from src.training.dataset import get_dataloaders
from src.training.losses import get_loss_function
from src.training.models import create_model, load_pretrained_swin_unetr
from src.training.schedulers import get_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MonaiTrainer:
    def __init__(self, config: dict, args):
        self.config = config
        self.args = args
        self.is_ddp = dist.is_initialized()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if self.is_ddp else 0
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        
        self.epochs = args.epochs or config["training"]["epochs"].get("monai", 500)
        self.lr = args.lr or config["training"]["lr"].get("monai", 0.001)
        self.use_amp = config["training"].get("amp", True)
        
        self.output_dir = Path(args.output_dir)
        if self.local_rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        self._setup_components()
        
    def _setup_components(self):
        # 1. Dataloaders
        splits_path = "data/splits.json"
        data_dir = "data/processed"
        self.train_loader, self.val_loader = get_dataloaders(
            self.config, self.args.fold, splits_path, data_dir, self.is_ddp
        )
        
        # 2. Model
        self.model = create_model(self.args.model, self.config).to(self.device)
        if self.args.pretrained and "swin_unetr" in self.args.model:
            # Example URL for MONAI SSL weights
            url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
            load_pretrained_swin_unetr(self.model, url)
            
        if self.is_ddp:
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
            
        # 3. Optimizer & Scheduler
        optimizer_name = self.config["training"].get("optimizer", "AdamW")
        if optimizer_name == "AdamW":
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            
        self.scheduler = get_scheduler(self.optimizer, self.config)
        
        # 4. Loss & Metrics
        self.criterion = get_loss_function(self.config).to(self.device)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        
        # 5. AMP
        self.scaler = GradScaler(enabled=self.use_amp)
        
        self.start_epoch = 0
        self.best_metric = -1.0
        
        # 6. Resume
        if self.args.resume:
            self._load_checkpoint(self.args.resume)
            
        # 7. Wandb
        if self.local_rank == 0:
            wandb_cfg = self.config.get("wandb", {})
            wandb.init(
                project=wandb_cfg.get("project", "hiri-segmentation"),
                entity=wandb_cfg.get("entity", None),
                config=self.config,
                name=f"fold_{self.args.fold}_{self.args.model}"
            )

    def _save_checkpoint(self, filename: str, metric: float = -1.0):
        if self.local_rank != 0:
            return
            
        model_state = self.model.module.state_dict() if self.is_ddp else self.model.state_dict()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
        }
        torch.save(checkpoint, self.output_dir / filename)
        
    def _load_checkpoint(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found at {path}")
            return
            
        checkpoint = torch.load(path, map_location=self.device)
        
        model_state = checkpoint['model_state_dict']
        if self.is_ddp:
            self.model.module.load_state_dict(model_state)
        else:
            self.model.load_state_dict(model_state)
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('best_metric', -1.0)
        logger.info(f"Resumed from epoch {self.start_epoch}")

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        
        if self.is_ddp:
            self.train_loader.sampler.set_epoch(self.current_epoch)
            
        for batch_data in self.train_loader:
            inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(self.device)
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            epoch_loss += loss.item()
            
        epoch_loss /= len(self.train_loader)
        
        # Synchronize loss across GPUs if DDP
        if self.is_ddp:
            loss_tensor = torch.tensor([epoch_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = (loss_tensor.item() / dist.get_world_size())
            
        return epoch_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.dice_metric.reset()
        
        patch_size = self.config["training"]["patch_size"]
        
        for batch_data in self.val_loader:
            inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(self.device)
            
            with autocast(enabled=self.use_amp):
                val_outputs = sliding_window_inference(
                    inputs, patch_size, 4, self.model, overlap=0.5, mode="gaussian"
                )
                
            # Discard multi-scale outputs during validation if any
            if isinstance(val_outputs, (list, tuple)):
                val_outputs = val_outputs[0]
                
            val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
            
            # Convert to one-hot for metric computation
            val_outputs_onehot = torch.nn.functional.one_hot(val_outputs.squeeze(1), num_classes=4).permute(0, 4, 1, 2, 3)
            labels_onehot = torch.nn.functional.one_hot(labels.squeeze(1), num_classes=4).permute(0, 4, 1, 2, 3)
            
            self.dice_metric(y_pred=val_outputs_onehot, y=labels_onehot)
            
        metric = self.dice_metric.aggregate()
        if self.is_ddp:
            dist.all_reduce(metric, op=dist.ReduceOp.SUM)
            metric /= dist.get_world_size()
        
        # metric is shape [3] -> [Liver, Right Kidney, Left Kidney]
        metric_liver = metric[0].item()
        metric_rk = metric[1].item()
        metric_lk = metric[2].item()
        mean_dice = metric.mean().item()
        
        return mean_dice, metric_liver, metric_rk, metric_lk

    def fit(self):
        val_interval = 5
        patience = 100
        patience_counter = 0
        
        for epoch in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch
            
            train_loss = self.train_epoch()
            self.scheduler.step()
            
            if self.local_rank == 0:
                logger.info(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f}")
                wandb.log({"train/loss": train_loss, "train/lr": self.scheduler.get_last_lr()[0]}, step=epoch)
                self._save_checkpoint("model_last.pth")
                
            if (epoch + 1) % val_interval == 0:
                mean_dice, liver_dice, rk_dice, lk_dice = self.validate()
                
                if self.local_rank == 0:
                    logger.info(f"Val Dice - Mean: {mean_dice:.4f} | Liver: {liver_dice:.4f} | R.Kidney: {rk_dice:.4f} | L.Kidney: {lk_dice:.4f}")
                    wandb.log({
                        "val/mean_dice": mean_dice,
                        "val/liver_dice": liver_dice,
                        "val/right_kidney_dice": rk_dice,
                        "val/left_kidney_dice": lk_dice
                    }, step=epoch)
                    
                    if mean_dice > self.best_metric:
                        self.best_metric = mean_dice
                        self._save_checkpoint("model_best.pth")
                        patience_counter = 0
                        logger.info("New best model saved!")
                    else:
                        patience_counter += val_interval
                        
            if self.local_rank == 0 and (epoch + 1) % 100 == 0:
                self._save_checkpoint(f"epoch_{epoch+1}.pth")
                
            if patience_counter >= patience:
                if self.local_rank == 0:
                    logger.info("Early stopping triggered.")
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--fold", type=int, required=True, help="Fold index")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output dir")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    # DDP Setup
    is_ddp = "WORLD_SIZE" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        
    set_seed(42)
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    trainer = MonaiTrainer(config, args)
    trainer.fit()
    
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
