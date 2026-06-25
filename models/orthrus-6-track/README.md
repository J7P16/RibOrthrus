---
license: mit
tags:
- rna
- biology
- mamba
- foundation-model
- orthrus
library_name: transformers
pipeline_tag: feature-extraction
---

# Orthrus 6-track

10M-param Mamba RNA encoder, contrastive pretrained on 45M+ mature mRNA transcripts. 6-track input (one-hot + CDS + splice).

## Model zoo

| Repo | Tracks | Embed dim | Objective | Used in |
|---|---|---|---|---|
| [`antichronology/orthrus-4-track`](https://huggingface.co/antichronology/orthrus-4-track) | 4 | 512 | contrastive | Nature Methods publication |
| [`antichronology/orthrus-6-track`](https://huggingface.co/antichronology/orthrus-6-track) | 6 | 512 | contrastive | Nature Methods publication |
| [`antichronology/orthrus-small-6-track`](https://huggingface.co/antichronology/orthrus-small-6-track) | 6 | 256 | contrastive | Nature Methods publication |
| [`antichronology/orthrus-mlm-6-track`](https://huggingface.co/antichronology/orthrus-mlm-6-track) | 6 | 512 | contrastive + MLM | Nature Methods publication |
| [`quietflamingo/orthrus-base-4-track`](https://huggingface.co/quietflamingo/orthrus-base-4-track) | 4 | 256 | contrastive | Pre-publication |
| [`quietflamingo/orthrus-large-4-track`](https://huggingface.co/quietflamingo/orthrus-large-4-track) | 4 | 512 | contrastive | Pre-publication |
| [`quietflamingo/orthrus-large-6-track`](https://huggingface.co/quietflamingo/orthrus-large-6-track) | 6 | 512 | contrastive | Pre-publication |

## Inference interface

Every Orthrus model exposes the same three inference methods plus a one-hot helper:

| Method | Output shape | Notes |
|---|---|---|
| `representation(x, lengths, channel_last=True)` | `(B, D)` | Mean-pooled, padding-aware |
| `representation_unpooled(x, channel_last=True)` | `(B, L, D)` | Per-position hidden states |
| `predict_tokens(x, lengths, channel_last=True)` | `(B, L, 4)` | MLM logits over `[A, C, G, T]`. Available on MLM-pretrained repos (`*-mlm-*`); raises `NotImplementedError` on contrastive-only checkpoints. |
| `seq_to_oh(seq)` | `(L, 4)` | One-hot helper, ordering `[A, C, G, T]` (U is treated as T) |

## Environment setup

The model uses a Mamba state-space backbone, which requires CUDA. The recommended environment matches the [Orthrus GitHub repo](https://github.com/bowang-lab/Orthrus):

```bash
# Conda env with Python 3.10
mamba create -n orthrus python=3.10
mamba activate orthrus

# PyTorch + transformers + huggingface_hub
pip install 'torch>=2.2' 'transformers<4.46' 'huggingface_hub>=0.24' safetensors

# Mamba kernels (require CUDA; pin versions for the published checkpoints)
pip install causal-conv1d==1.2.0.post2 --no-build-isolation --no-cache-dir
pip install mamba-ssm==1.2.0.post1 --no-build-isolation --no-cache-dir

# GenomeKit, only if you want to build 6-track inputs from real transcripts
mamba install "genomekit>=6.0.0"
wget -O starter_build.sh https://raw.githubusercontent.com/deepgenomics/GenomeKit/main/starter/build.sh
chmod +x starter_build.sh
./starter_build.sh
```

## Load the model

```python
import torch
from transformers import AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained(
    "antichronology/orthrus-6-track",
    trust_remote_code=True,
).to(device).eval()
```

## Pooled representation (6-track input)

For 6-track models, you need two extra channels: CDS (1 at the first base of each codon, i.e. positions 0, 3, 6, ... from the CDS start; 0 elsewhere) and splice (1 at the last base of each exon, 0 elsewhere). The cleanest way to build these from a real transcript is via GenomeKit:

```python
import numpy as np
import torch
from genome_kit import Genome

genome = Genome("gencode.v44")  # or whatever annotation you built

def find_transcript_by_gene_name(genome, gene_name):
    return [t for t in genome.transcripts if t.gene.name == gene_name]

def get_transcript_seq(transcript, genome):
    return "".join(genome.dna(exon) for exon in transcript.exons)

def build_cds_track(transcript):
    """1 at the first base of each codon (every 3rd position from the CDS start: 0, 3, 6, ...), 0 in UTRs."""
    exons = transcript.exons
    L = sum(len(e) for e in exons)
    cds = transcript.cdss
    if not cds:
        return np.zeros(L, dtype=np.float32)

    strand = transcript.strand
    if strand == "+":
        sorted_cds = sorted(cds, key=lambda c: c.start)
        sorted_exons = sorted(exons, key=lambda e: e.start)
        first_cds = sorted_cds[0]
    else:
        sorted_cds = sorted(cds, key=lambda c: c.end, reverse=True)
        sorted_exons = sorted(exons, key=lambda e: e.end, reverse=True)
        first_cds = sorted_cds[0]

    cds_len = sum(len(c) for c in sorted_cds)

    five_utr = 0
    for ex in sorted_exons:
        if strand == "+":
            if ex.end <= first_cds.start:
                five_utr += len(ex)
            elif ex.overlaps(first_cds):
                five_utr += max(0, first_cds.start - ex.start)
                break
            else:
                break
        else:
            if ex.start >= first_cds.end:
                five_utr += len(ex)
            elif ex.overlaps(first_cds):
                five_utr += max(0, ex.end - first_cds.end)
                break
            else:
                break

    three_utr = max(0, L - (five_utr + cds_len))
    body = np.zeros(cds_len, dtype=np.float32)
    body[0::3] = 1.0
    return np.concatenate([
        np.zeros(five_utr, dtype=np.float32),
        body,
        np.zeros(three_utr, dtype=np.float32),
    ])

def build_splice_track(transcript):
    """1 at the last base of each exon, 0 elsewhere."""
    exons = transcript.exons
    L = sum(len(e) for e in exons)
    track = np.zeros(L, dtype=np.float32)
    cumulative = 0
    for ex in exons:
        cumulative += len(ex)
        track[cumulative - 1] = 1.0
    return track

t = find_transcript_by_gene_name(genome, "BCL2L1")[0]
sequence = get_transcript_seq(t, genome)
cds = build_cds_track(t)
splice = build_splice_track(t)

seq_ohe = model.seq_to_oh(sequence).numpy()              # (L, 4)
oh = np.hstack([seq_ohe, cds[:, None], splice[:, None]])  # (L, 6)
oh = torch.tensor(oh, device=device).unsqueeze(0)
lengths = torch.tensor([oh.shape[1]], device=device)

with torch.no_grad():
    emb = model.representation(oh, lengths, channel_last=True)
# emb.shape == (1, D)
```

If you already have CDS and splice arrays from another source (e.g. a UCSC genePred table), you can skip GenomeKit and just `np.hstack` them with `seq_to_oh` output.

## Un-pooled (per-position) representation

```python
with torch.no_grad():
    hidden = model.representation_unpooled(oh, channel_last=True)
# hidden.shape == (1, L, D)
# Useful for: local scoring at a specific transcript position, attention
# probing, downstream sequence-tagging tasks.
```

## Fine-tuning

Configuration files and training scripts for fine-tuning, linear probing, and homology-aware splitting live in the [Orthrus GitHub repo](https://github.com/bowang-lab/Orthrus). All fine-tuning data and pre-computed splits are mirrored on [Zenodo](https://zenodo.org/records/13910050).

## Citation

```bibtex
@article{fradkinShi2026,
  title = {Orthrus: toward evolutionary and functional RNA foundation models},
  ISSN = {1548-7105},
  url = {http://dx.doi.org/10.1038/s41592-026-03064-3},
  DOI = {10.1038/s41592-026-03064-3},
  journal = {Nature Methods},
  publisher = {Springer Science and Business Media LLC},
  author = {Fradkin, Philip and Shi, Ruian "Ian" and Dalal, Taykhoom and Isaev, Keren and Frey, Brendan J. and Lee, Leo J. and Morris, Quaid and Wang, Bo},
  year = {2026},
  month = Apr
}
```

## License

MIT
