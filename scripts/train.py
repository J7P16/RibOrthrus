import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from prepare_data import MultiCellRiboDataset
from torch.utils.data import random_split, DataLoader
import torch
from collators import PrecomputedEmbeddingCollator
from models import PredictionHead
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import argparse

def main(args):
    ribo_dir = "data"

    full_dataset = MultiCellRiboDataset(
        ribo_dir=ribo_dir,
        read_length_min=args.min_read,
        read_length_max=args.max_read,
    )

    number_of_cell_lines = len(full_dataset.cell_lines)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.manual_seed),
    )
    fixed_sequence_length = args.seq_len 
    collate_fn = PrecomputedEmbeddingCollator(
        embedding_dir="embeddings", 
        fixed_sequence_length=fixed_sequence_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    system = PredictionHead(
        number_of_cell_lines=number_of_cell_lines,
        lr=args.learning_rate,
        multinomial_resolution=fixed_sequence_length // 8,
        positional_weight=5.0,
    )
    trainer = pl.Trainer(
        callbacks=[EarlyStopping(monitor="val_ribo_pcc", mode="max", patience=args.patience)],
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
    )
    trainer.fit(system, train_loader, val_loader)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RibOrthrus')
    
    parser.add_argument("--seed", dest="manual_seed", default=42, 
                        type=int, help="Random seed for training")
    parser.add_argument("--assembly", dest="reference", default="gencode.v41",
                        help="Genome Reference for Obtaining Sequences")
    parser.add_argument("--model-type", dest="model_type", default="ribo",
                        help="Specific Task for RibOrthrus")
    
    # Training Parameters
    parser.add_argument("--max-epochs", dest="max_epochs", default=200,
                        type=int, help="Max Epochs for Training")
    parser.add_argument("--patience", dest="patience", default=8,
                        type=int, help="Patience for Training")
    parser.add_argument("--learning-rate", dest="learning_rate", default=0.001,
                        type=float, help="Learning Rate for Training")
    parser.add_argument("--seq-len", dest="seq_len", default=16384,
                        type=int, help="Fixed Sequence Length for Training (Must be Divisible by 8!)")
    parser.add_argument("--min-read", dest="min_read", default=26,
                        type=int, help="Minimum Read Length in Training Data")
    parser.add_argument("--max-read", dest="max_read", default=33,
                        type=int, help="Maximum Read Length in Training Data")

    # Dataloader Parameters
    parser.add_argument("--batch-size", dest="batch_size", default=32,
                        type=int, help="Batch Size for Dataloader")
    parser.add_argument("--num-workers", dest="num_workers", default=8,
                        type=int, help="Number of Workers for Dataloader")

    args = parser.parse_args()

    main(args)
