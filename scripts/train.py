import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from prepare_data import MultiCellRiboDataset
from torch.utils.data import random_split, DataLoader
import torch
from collators import PrecomputedEmbeddingCollator
from models import PredictionHead
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

def main():
    ribo_dir = "data"
    batch_size = 32

    full_dataset = MultiCellRiboDataset(
        ribo_dir=ribo_dir,
        read_length_min=22,
        read_length_max=33,
        normalize_per_track=False,
    )

    number_of_cell_lines = len(full_dataset.cell_lines)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    fixed_sequence_length = 16384 
    collate_fn = PrecomputedEmbeddingCollator(
        embedding_dir="embeddings", 
        fixed_sequence_length=fixed_sequence_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
    )
    system = PredictionHead(
        number_of_cell_lines=number_of_cell_lines,
        #lr=3e-4,
        lr=0.01,
        multinomial_resolution=fixed_sequence_length // 8,
        positional_weight=5.0,
    )
    trainer = pl.Trainer(
        callbacks=[EarlyStopping(monitor="val_pcc", mode="max", patience=8)],
        max_epochs=1000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
    )
    trainer.fit(system, train_loader, val_loader)

if __name__ == "__main__":
    main()
