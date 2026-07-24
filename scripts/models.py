import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef
from loss import multinomial_loss
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from typing import Sequence

class DilatedConvBlock(nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        kernel_size: int,
        dilation: int, 
        dropout: float, 
    ):
        super().__init__()
        self.dilated_conv = nn.Sequential(
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=num_channels,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=kernel_size,
                padding=dilation * (kernel_size - 1) // 2,
                dilation=dilation,
            )
        )
        self.pointwise_conv = nn.Sequential(
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=num_channels,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=1,
            ),
            nn.Dropout(p=dropout),
        )
        
    def forward(self, x):
        h = self.dilated_conv(x)
        h = self.pointwise_conv(h)
        return x + h             

class PredictionHead(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        tower_channels: int, 
        num_blocks: int, 
        num_cell_lines: int, 
        dropout: float,
        model_type: str,
    ):
        super().__init__()
        self.num_cell_lines = num_cell_lines
        self.input_projection = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=tower_channels, 
            kernel_size=1,
        )
        #self.input_projection = nn.Identity()
        self.dilation_tower = nn.ModuleList(
            [
                DilatedConvBlock(
                    num_groups=1,
                    num_channels=tower_channels, 
                    kernel_size=3,
                    dilation=2**i, 
                    dropout=dropout,
                ) 
                for i in range(num_blocks)
            ]
        )
        self.model_type = model_type
        self.ribo_head = nn.Sequential(
            nn.Conv1d(
                in_channels=tower_channels,
                out_channels=num_cell_lines,
                kernel_size=1,
            ),
            nn.Softplus(),
        )
        self.rna_head = nn.Sequential(
            nn.Conv1d(
                in_channels=tower_channels,
                out_channels=num_cell_lines,
                kernel_size=1,
            ),
            nn.Softplus(),
        )

    def _concatenate_outputs(self, ribo_output, rna_output):
        x = torch.stack([ribo_output, rna_output], dim=2)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], 2 * self.num_cell_lines)
        return x

    def forward(self, x):
        x = x.transpose(1, 2)         
        x = self.input_projection(x) 
        for block in self.dilation_tower:
            x = block(x)
        if self.model_type == "ribo":
            return self.ribo_head(x).transpose(1, 2)
        if self.model_type == "rna":
            return self.rna_head(x).transpose(1, 2)
        return self._concatenate_outputs(self.ribo_head(x), self.rna_head(x))

class LinearModel(nn.Module):
    def __init__(self, d_in=512, n_cell_lines=24):
        super().__init__()
        self.n_cell_lines = n_cell_lines
        self.input = nn.Linear(d_in, 1024)
        self.lin = nn.Linear(1024, 1024)
        self.out = nn.Linear(1024, 2 * n_cell_lines)
        
    def forward(self, x):
        x = self.input(x)
        x = self.lin(x)
        x = self.lin(x)
        x = self.out(x)
        return F.softplus(x)

class RibOrthrus(pl.LightningModule):
    def __init__(
        self,
        num_cell_lines: int,
        cell_lines: Sequence[str] | None = None,
        inspect_cell_line: str | None = None,
        lr: float = 1e-3,
        orthrus_lr: float = 1e-5,
        multinomial_resolution: int = 1,
        positional_weight: float = 5.0,
        fine_tune: bool = False,
        model_type: str = "ribo",
    ):
        super().__init__()
        self.save_hyperparameters()
        """
        # inspecting orthrus architecture
        for name, module in base_orthrus.named_modules():
            if isinstance(module, nn.Linear):
                print(name, module)
        """
        self.fine_tune = fine_tune
        self.cell_lines = list(cell_lines or [])
        self.inspect_cell_line = inspect_cell_line
        self.inspect_cell_index = self._resolve_cell_line_index(inspect_cell_line)
        self.orthrus = None
        if self.fine_tune:
            self.orthrus = get_peft_model(
                AutoModel.from_pretrained(
                    "antichronology/orthrus-mlm-6-track",
                    trust_remote_code=True,
                    torch_dtype="auto",
                ), 
                LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules=["x_proj"],
                    bias="none"
                ),
            ) 
        self.head = PredictionHead(
            in_channels=512,
            tower_channels=128,
            num_blocks=7, 
            num_cell_lines=num_cell_lines, 
            dropout=0.1,
            model_type=model_type,
        )
         
    def _resolve_cell_line_index(self, cell_line: str | None) -> int | None:
        if cell_line is None or not self.cell_lines:
            return None

        lower_cell_line = cell_line.lower()
        for index, candidate in enumerate(self.cell_lines):
            if candidate.lower() == lower_cell_line:
                return index

        for index, candidate in enumerate(self.cell_lines):
            if lower_cell_line in candidate.lower():
                return index

        print(f"Warning: {cell_line} not found in cell lines: {self.cell_lines}")
        return None

    def forward(self, x):
        if self.orthrus is not None:
            x = self.orthrus(x)
        x = self.head(x)
        return x

    def shared_step(self, batch, stage: str):
        if self.fine_tune:
            x = batch["six_track"]
        else:
            x = batch["x"]
        mask = batch["mask"]
        y_pred = self.forward(x)
        y_true = batch["y_true"] 
        batch_size = x.size(0)
        
        loss_dict = multinomial_loss(
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
            multinomial_resolution=self.hparams.multinomial_resolution,
            positional_weight=self.hparams.positional_weight,
        )
        
        expanded_mask = mask.unsqueeze(-1).expand_as(y_true)

        valid_pred = y_pred[expanded_mask]
        valid_true = y_true[expanded_mask]

        pcc = pearson_corrcoef(
            valid_pred.double(),
            valid_true.double(),
        )

        if self.hparams.model_type == "mixed":
            target_channels = {"ribo": slice(0, None, 2), "rna": slice(1, None, 2)}
        else:
            target_channels = {self.hparams.model_type: slice(None)}

        for target, channels in target_channels.items():
            target_pred = y_pred[..., channels]
            target_true = y_true[..., channels]
            target_mask = mask.unsqueeze(-1).expand_as(target_pred)
            target_pcc = pearson_corrcoef(
                target_pred[target_mask].double(),
                target_true[target_mask].double(),
            )
            self.log(
                f"{stage}_{target}_pcc",
                target_pcc.float(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )

        if self.inspect_cell_index is not None:
            inspected_mask = mask
            for target, channels in target_channels.items():
                channel = self.inspect_cell_index if self.hparams.model_type != "mixed" else (
                    2 * self.inspect_cell_index + (target == "rna")
                )
                inspected_pcc = pearson_corrcoef(
                    y_pred[..., channel][inspected_mask].double(),
                    y_true[..., channel][inspected_mask].double(),
                )
                self.log(
                    f"{stage}_{self.inspect_cell_line}_{target}_pcc",
                    inspected_pcc.float(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=batch_size,
                )

        self.log(
            f"{stage}_pcc",
            pcc.float(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size
        )

        self.log(
            f"{stage}_loss",
            loss_dict["loss"],
            prog_bar=True,
            batch_size=batch_size,
        )
        
        return loss_dict["loss"]

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")
    
    def configure_optimizers(self):
        parameter_groups = [
            {
                "params": self.head.parameters(), 
                "lr": self.hparams.lr
            }
        ]
        
        if self.orthrus is not None:
            parameter_groups.append(
                {
                    "params": [p for p in self.orthrus.parameters() if p.requires_grad],
                    "lr": self.hparams.orthrus_lr,
                }
            )
        
        optimizer = torch.optim.AdamW(
            parameter_groups,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=1e-3
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
