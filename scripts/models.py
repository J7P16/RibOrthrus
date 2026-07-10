import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef
from loss import multinomial_loss

class DilatedBlock(nn.Module):
    def __init__(self, ch, dilation, kernel=3, p=0.1, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.conv  = nn.Conv1d(ch, ch, kernel, padding=dilation * (kernel - 1) // 2, dilation=dilation)
        self.pointwise = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(p)
        
    def forward(self, x):
        h = self.norm(x)
        h = F.gelu(h)
        h = self.conv(h)
        h = self.norm(h)
        h = F.gelu(h)
        h = self.pointwise(h)
        h = self.drop(h)
        return x + h             

class DenseRateHead(nn.Module):
    def __init__(self, d_in=512, n_blocks=5, n_cell_lines=24, p=0.1):
        super().__init__()
        self.n_cell_lines = n_cell_lines
        self.proj_in = nn.Conv1d(d_in, 32, 1)
        self.tower = nn.ModuleList(DilatedBlock(32, dilation=2**i, p=p) for i in range(n_blocks))
        self.ribo_head = nn.Conv1d(32, n_cell_lines, 1)
        self.rna_head = nn.Conv1d(32, n_cell_lines, 1)

    def forward(self, x):       
        x = x.transpose(1, 2)          
        x = self.proj_in(x)          
        for blk in self.tower:
            x = blk(x)                 
        ribo_out = self.ribo_head(x)
        rna_out = self.rna_head(x)
        
        x = torch.stack([ribo_out, rna_out], dim=2)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], 2 * self.n_cell_lines)
        x = F.softplus(x)
        return x

class PredictionHead(pl.LightningModule):
    def __init__(
        self,
        number_of_cell_lines: int,
        lr: float = 1e-3,
        multinomial_resolution: int = 1,
        positional_weight: float = 5.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.head = DenseRateHead(d_in=512, n_blocks=5, n_cell_lines=number_of_cell_lines, p=0.1)

        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []

    def forward(self, x, lengths):
        return self.head(x)

    def shared_step(self, batch, stage: str):
        x = batch["x"]
        y_true = batch["y_true"]
        mask = batch["mask"]
        lengths = batch["lengths"]

        y_pred = self.forward(x, lengths)

        loss_dict = multinomial_loss(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
            multinomial_resolution=self.hparams.multinomial_resolution,
            positional_weight=self.hparams.positional_weight,
        )

        expanded_mask = mask.unsqueeze(-1).expand_as(y_true)

        flat_pred = y_pred[expanded_mask].detach().cpu()
        flat_true = y_true[expanded_mask].detach().cpu()
        if stage == "train":
            self.train_preds.append(flat_pred)
            self.train_targets.append(flat_true)
        else:
            self.val_preds.append(flat_pred)
            self.val_targets.append(flat_true)

        self.log(
            f"{stage}_loss",
            loss_dict["loss"],
            prog_bar=True,
            batch_size=x.size(0)
        )

        return loss_dict["loss"]

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")
    
    def on_train_epoch_end(self):
        if len(self.train_preds) == 0:
            return

        preds = torch.cat(self.train_preds)
        targets = torch.cat(self.train_targets)

        ribo_pcc = pearson_corrcoef(preds[..., 0::2].flatten(), targets[..., 0::2].flatten())
        rna_pcc = pearson_corrcoef(preds[..., 1::2].flatten(), targets[..., 1::2].flatten())
        pcc = pearson_corrcoef(preds.flatten(), targets.flatten())

        self.log("train_pcc", pcc, prog_bar=True)
        self.log("train_rna_pcc", rna_pcc, prog_bar=True)
        self.log("train_ribo_pcc", ribo_pcc, prog_bar=True)

        self.train_preds.clear()
        self.train_targets.clear()

    def on_validation_epoch_end(self):
        if len(self.val_preds) == 0:
            return

        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)

        ribo_pcc = pearson_corrcoef(preds[..., 0::2].flatten(), targets[..., 0::2].flatten())
        rna_pcc = pearson_corrcoef(preds[..., 1::2].flatten(), targets[..., 1::2].flatten())
        pcc = pearson_corrcoef(preds.flatten(), targets.flatten())
        self.log("val_pcc", pcc, prog_bar=True)
        self.log("val_rna_pcc", rna_pcc, prog_bar=True)
        self.log("val_ribo_pcc", ribo_pcc, prog_bar=True)

        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=1e-2
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr * 0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
