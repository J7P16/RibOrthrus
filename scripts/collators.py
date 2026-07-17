import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from orthrus.gk_utils import create_six_track_encoding

class PrecomputedEmbeddingCollator:
    def __init__(self, embedding_dir="embeddings", fixed_sequence_length=16384):
        self.embedding_dir = embedding_dir
        self.fixed_sequence_length = fixed_sequence_length

    def __call__(self, batch):
        embeddings = []
        y_values = []
        masks = []
        lengths = []
        transcript_ids = []

        for item in batch:
            tx_id = item["transcript_id"]
            enst_id = str(tx_id).split("|")[0]

            path = os.path.join(self.embedding_dir, f"{enst_id}.pt")
            saved = torch.load(path, map_location="cpu", weights_only=False)

            hidden = saved["embedding"].float()  # [L, 512]
            y = item["y_true"].float()
             
            fixed_len = self.fixed_sequence_length
            true_len = min(y.shape[0], hidden.shape[0], fixed_len)

            hidden = hidden[:fixed_len]
            y = y[:fixed_len]

            hidden_pad = fixed_len - hidden.shape[0]
            if hidden_pad > 0:
                hidden = nn.functional.pad(
                    hidden,
                    (0, 0, 0, hidden_pad),
                )

            y_pad = fixed_len - y.shape[0]
            if y_pad > 0:
                y = nn.functional.pad(
                    y,
                    (0, 0, 0, y_pad),
                )

            mask = torch.zeros(fixed_len, dtype=torch.bool)
            mask[:true_len] = True

            embeddings.append(hidden)
            y_values.append(y)
            masks.append(mask)
            lengths.append(true_len)
            transcript_ids.append(tx_id)

        return {
            "transcript_ids": transcript_ids,
            "x": torch.stack(embeddings, dim=0),  # [B, L, 512]
            "y_true": torch.stack(y_values, dim=0),
            "mask": torch.stack(masks, dim=0),
            "lengths": torch.tensor(lengths),
        }

class FineTuneCollator:
    def __init__(self, genome, fixed_sequence_length=16384):
        self.genome = genome
        self.fixed_sequence_length = fixed_sequence_length

    def __call__(self, batch):
        six_track_values = []
        y_values = []
        masks = []
        lengths = []
        transcript_ids = []
        
        fixed_len = self.fixed_sequence_length

        for item in batch:
            tx_id = item["transcript_id"]
            enst_id = str(tx_id).split("|")[0]

            transcript = self.genome.transcripts[enst_id]
            
            six_track = create_six_track_encoding(transcript, self.genome)
            six_track = torch.as_tensor(six_track).float()
            
            y = item["y_true"].float()
            
            original_len = min(six_track.shape[1], y.shape[0])
            true_len = min(original_len, fixed_len)

            six_track = six_track[:, :true_len]
            y = y[:true_len]

            pad_len = fixed_len - true_len
            if pad_len > 0:
                six_track = F.pad(six_track, (0, pad_len), value=0)
                y = F.pad(y, (0, 0, 0, pad_len), value=0)

            mask = torch.zeros(fixed_len, dtype=torch.bool)
            mask[:true_len] = True

            six_track_values.append(six_track)
            y_values.append(y)
            masks.append(mask)
            lengths.append(true_len)
            transcript_ids.append(tx_id)

        return {
            "transcript_ids": transcript_ids,
            "six_track": torch.stack(six_track_values, dim=0),
            "y_true": torch.stack(y_values, dim=0),
            "mask": torch.stack(masks, dim=0),
            "lengths": torch.tensor(lengths, dtype=torch.long),
        }
