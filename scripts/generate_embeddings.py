import torch
from genome_kit import Genome, Interval
from transformers import AutoModel
from orthrus.gk_utils import find_transcript_by_gene_name, create_six_track_encoding
from loss import multinomial_loss

# loading in a sample transcript
genome = Genome("gencode.v29")
interval = Interval("chr7", "+", 117120016, 117120201, genome)
print("\n")
print("Sample Transcript Subsequence:")
print(genome.dna(interval) + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading in model from huggingface
model = AutoModel.from_pretrained("antichronology/orthrus-6-track", trust_remote_code=True, dtype="auto")
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

print("Singular Output Embedding Shape:")
print(embedding.shape)
print("\n")
print("Embedding Contents:")
print(embedding)
print("\n")
#with torch.no_grad():
#    out = model(sixt)

#print(type(out)) # <class 'torch.Tensor'>
#print("Orthrus Model Output Shape:")
#print(out.shape)
#print("\n")

print("Predictive Model Head Input Shape (Post-squeeze since there's only one transcript):")
#hidden = out.squeeze(0) # [1, 512]
hidden = embedding
#print(hidden.shape)
print(hidden.shape)
print("\n")

SEQUENCE_LENGTH = lengths[0]
head = torch.nn.Linear(512, SEQUENCE_LENGTH).to(device)
preds = head(embedding)
print("Predictive Model Head Architecture:")
print(head)
print("\n")

total_params = sum(p.numel() for p in head.parameters())
print("Number of Predictive Model Head Parameters:")
print(total_params)
print("\n")

print("Predictive Model Head Output Shape:")
print(preds.shape)
print("\n")

test = multinomial_loss(
        y_true=torch.poisson(torch.ones(32, 128, 1) * 2.0), 
        y_pred=torch.rand(32, 128, 1) * 3.0, 
        mask=torch.ones(32, 128, 1, dtype=torch.bool),
        multinomial_resolution=4,
        positional_weight=0.281
)
print("Random General Test Case Output for Imported Multinomial Loss Function:")
print(test)

