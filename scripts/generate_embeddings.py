import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from tqdm import tqdm
import torch
from genome_kit import Genome, Interval
from transformers import AutoModel
from orthrus.gk_utils import find_transcript_by_gene_name, create_six_track_encoding
from prepare_data import MultiCellRiboDataset
import argparse

def main(args):
    ribo_dir = "data"
    out_dir = "embeddings"
    genome_name = args.reference
    fixed_sequence_length = args.seq_len

    os.makedirs(out_dir, exist_ok=True)

    dataset = MultiCellRiboDataset(
        ribo_dir=ribo_dir,
        read_length_min=args.min_read,
        read_length_max=args.max_read,
        normalize_per_track=False,
    )

    genome = Genome(genome_name)

    model = AutoModel.from_pretrained(
        "antichronology/orthrus-mlm-6-track",
        trust_remote_code=True,
        torch_dtype="auto",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    for item in tqdm(dataset):
        tx_id = item["transcript_id"]
        enst_id = str(tx_id).split("|")[0]

        out_path = os.path.join(out_dir, f"{enst_id}.pt")

        if os.path.exists(out_path):
            continue

        transcript = genome.transcripts[enst_id]

        sixt = create_six_track_encoding(transcript, genome)
        sixt = torch.tensor(sixt, dtype=torch.float32).T  # [L, 6]

        L = sixt.shape[0]
        true_len = min(L, fixed_sequence_length)

        if L > fixed_sequence_length:
            sixt = sixt[:fixed_sequence_length]
        else:
            pad_len = fixed_sequence_length - L
            sixt = torch.nn.functional.pad(sixt, (0, 0, 0, pad_len))

        x = sixt.T.unsqueeze(0).to(device)  # [1, 6, L]

        with torch.no_grad():
            hidden = model(x)  # [1, L, 512]

        hidden = hidden.squeeze(0).cpu().to(torch.float16)  # [L, 512]

        torch.save(
            {
                "transcript_id": tx_id,
                "enst_id": enst_id,
                "embedding": hidden,
                "length": true_len,
            },
            out_path,
        )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RibOrthrus')

    parser.add_argument("--assembly", dest="reference", default="gencode.v41",
                        help="Genome Reference for Obtaining Sequences")
    parser.add_argument("--seq-len", dest="seq_len", default=16384,
                        type=int, help="Fixed Sequence Length for Training (Must be Divisible by 8!)")
    parser.add_argument("--min-read", dest="min_read", default=26,
                        type=int, help="Minimum Read Length in Training Data")
    parser.add_argument("--max-read", dest="max_read", default=33,
                        type=int, help="Maximum Read Length in Training Data")

    args = parser.parse_args()

    main(args)
