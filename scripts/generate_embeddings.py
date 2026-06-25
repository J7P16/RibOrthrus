import torch
from genome_kit import Genome, Interval
from transformers import AutoModel
from orthrus.gk_utils import find_transcript_by_gene_name, create_six_track_encoding

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

# generate embeddings
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
#lengths = torch.tensor([sixt.shape[2]], device=device)
#with torch.no_grad():
#    embedding = model.representation(sixt, lengths)

#print(embedding.shape)
#print(embedding)
with torch.no_grad():
    out = model(sixt)

#print(type(out)) # <class 'torch.Tensor'>
print("Orthrus Model Output Shape:")
print(out.shape)
print("\n")

print("Predictive Model Head Input Shape (Post-squeeze since there's only one transcript):")
hidden = out.squeeze(0) # [2578, 512]
print(hidden.shape)
print("\n")

head = torch.nn.Linear(512, 25).to(device)
preds = head(hidden)
print("Predictive Model Head Architecture:")
print(head)
print("\n")

print("Predictive Model Head Output Shape:")
print(preds.shape)
print("\n")