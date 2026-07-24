import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from prepare_data import MultiCellRiboDataset
from torch.utils.data import random_split, DataLoader
import torch
from collators import PrecomputedEmbeddingCollator, FineTuneCollator
from models import RibOrthrus
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import argparse
from genome_kit import Genome

def main(args):
    # Keep the data split, data-loader order, and model initialization stable
    # between autoresearch experiments.
    pl.seed_everything(args.manual_seed, workers=True)

    full_dataset = MultiCellRiboDataset(
        ribo_dir="data",
        read_length_min=args.min_read,
        read_length_max=args.max_read,
        model_type=args.model_type,
    )

    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.manual_seed),
    )
    
    if args.fine_tune:
        collate_fn = FineTuneCollator(
            genome=Genome("gencode.v41"), 
            fixed_sequence_length=args.seq_len,
        )
    else:
        collate_fn = PrecomputedEmbeddingCollator(
            embedding_dir="embeddings",
            fixed_sequence_length=args.seq_len,
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
    system = RibOrthrus(
        num_cell_lines=len(full_dataset.cell_lines),
        cell_lines=full_dataset.cell_lines,
        inspect_cell_line=args.inspect_cell_line,
        lr=args.learning_rate,
        multinomial_resolution=args.seq_len // 8,
        positional_weight=5.0,
        fine_tune=args.fine_tune,
        model_type=args.model_type,
    )
    
    trainer = pl.Trainer(
        #callbacks=[EarlyStopping(
        #    # Autoresearch optimizes the same quantity reported at the end of
        #    # each run, so early stopping must use validation loss too.
        #    monitor="val_loss",
        #    mode="min",
        #    patience=args.patience,
        #)],
        max_epochs=args.max_epochs,
        max_time=args.max_time,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
    )
    trainer.fit(system, train_loader, val_loader)

    val_loss = trainer.callback_metrics.get("val_loss")
    if val_loss is None:
        raise RuntimeError("Training finished without reporting val_loss")
    print(f"AUTORESEARCH_METRIC val_loss={val_loss.detach().cpu().item():.8f}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RibOrthrus')
    
    parser.add_argument("--seed", dest="manual_seed", default=42, 
                        type=int, help="Random seed for training")
    parser.add_argument("--assembly", dest="reference", default="gencode.v41",
                        help="Genome Reference for Obtaining Sequences")
    parser.add_argument("--model-type", dest="model_type", default="ribo",
                        choices=("ribo", "rna", "mixed"),
                        help="Prediction target: ribo, rna, or both (mixed)")
    
    # Training Parameters
    parser.add_argument("--train-split", dest="train_split", default=0.8,
                        type=float, help="Train/Test Split for Training Session")
    parser.add_argument("--max-epochs", dest="max_epochs", default=200,
                        type=int, help="Max Epochs for Training")
    parser.add_argument("--max-time", dest="max_time", default="00:00:10:00",
                        help="Fixed wall-clock budget per experiment (DD:HH:MM:SS)")
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
    parser.add_argument("--fine-tune", dest="fine_tune", action="store_true",
                        help="Enable LoRA fine-tuning of Orthrus")
    parser.add_argument("--inspect-cell-line", dest="inspect_cell_line", default="HEK293",
                        help="Cell line to log individual Ribo/RNA PCC metrics for")

    # Dataloader Parameters
    parser.add_argument("--batch-size", dest="batch_size", default=32,
                        type=int, help="Batch Size for Dataloader")
    parser.add_argument("--num-workers", dest="num_workers", default=8,
                        type=int, help="Number of Workers for Dataloader")

    args = parser.parse_args()

    main(args)
