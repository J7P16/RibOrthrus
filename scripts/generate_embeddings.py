import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from tqdm import tqdm
import torch
from genome_kit import Genome, Interval
from transformers import AutoModel
from orthrus.gk_utils import find_transcript_by_gene_name, create_six_track_encoding
from prepare_ribo_data import MultiCellRiboDataset

"""
# loading in a sample transcript
genome = Genome("gencode.v41")
interval = Interval("chr7", "+", 117120016, 117120201, genome)
print("\n")
print("Sample Transcript Subsequence:")
print(genome.dna(interval) + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading in model from huggingface
model = AutoModel.from_pretrained("antichronology/orthrus-6-track", trust_remote_code=True, torch_dtype="auto")
model.to(device)
model.eval()

# need to freeze model parameters for my pipeline
for param in model.parameters():
    param.requires_grad = False
print("Frozen Orthrus Model Architecture and Layers:")
print(model)
print("\n")

# generate embedding for entire transcrit
transcripts = find_transcript_by_gene_name(genome, "BCL2L1")
print(transcripts)
print("\n")
t = transcripts[0]
sixt = create_six_track_encoding(t, genome)
sixt = torch.tensor(sixt, dtype=torch.float32)
sixt = sixt.unsqueeze(0)
sixt = sixt.to(device)
print("Six-Track Encoding Input Shape:")
print(sixt.shape)
print("\n")
lengths = torch.tensor([sixt.shape[2]], device=device)
with torch.no_grad():
    embedding = model.representation(sixt, lengths)

#print("Singular Output Embedding Shape:")
#print(embedding.shape)
#print("\n")
#print("Embedding Contents:")
#print(embedding)
#print("\n")
with torch.no_grad():
    out = model(sixt)

#print(type(out)) # <class 'torch.Tensor'>
print("Orthrus Model Output Shape (Ideally Embeddings that Represent Each Nucleotide):")
print(out.shape)
print("\n")

print("Predictive Model Head Input Shape (Post-squeeze since there's only one transcript):")
hidden = out.squeeze(0) # [1, 512]
#hidden = embedding
#print(hidden.shape)
print(hidden.shape)
print("\n")

SEQUENCE_LENGTH = lengths[0]
NUMBER_OF_CELL_LINES = 24 # currently matches the data given from lab
head = torch.nn.Linear(512, 2 * NUMBER_OF_CELL_LINES).to(device)
raw_preds = head(hidden)
print("Predictive Model Head Architecture:")
print(head)
print("\n")

# Quick check to count the total parameters in MLP to make sure my logic is correct
total_params = sum(p.numel() for p in head.parameters())
print("Number of Predictive Model Head Parameters:")
print(total_params)
print("\n")

print("Predictive Model Head Output Shape:")
preds = torch.nn.functional.softplus(raw_preds)
print(preds.shape)
print("\n")

test = multinomial_loss(
        y_true=torch.poisson(torch.ones_like(preds) * 2.0), 
        #y_pred=torch.rand(32, 128, 1) * 3.0, 
        #mask=torch.ones(32, 128, 1, dtype=torch.bool),
        #y_true=preds,
        y_pred=preds,
        mask=torch.ones_like(preds, dtype=torch.bool),
        multinomial_resolution=1,
        positional_weight=5
)


print("Random General Test Case Output for Imported Multinomial Loss Function (Ideally should be 0 since y_true and y_pred are the same)):")
print(test)
"""

def main():
    ribo_dir = "data"
    out_dir = "embeddings"
    genome_name = "gencode.v41"
    fixed_sequence_length = 16384

    os.makedirs(out_dir, exist_ok=True)

    dataset = MultiCellRiboDataset(
        ribo_dir=ribo_dir,
        read_length_min=28,
        read_length_max=28,
        normalize_per_track=False,
    )

    genome = Genome(genome_name)

    model = AutoModel.from_pretrained(
        "antichronology/orthrus-6-track",
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
    main()
