import os
import torch
import torch.nn as nn

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
            L = y.shape[0]
            true_len = min(L, fixed_len)
            
            hidden = hidden[:fixed_len]
            y = y[:fixed_len]
            
            hidden_pad = fixed_len - hidden.shape[0]
            if hidden_pad > 0:
                hidden = nn.functional.pad(hidden, (0, 0, 0, hidden_pad))
            y_pad = fixed_len - y.shape[0]
            if y_pad > 0:
                y = nn.functional.pad(y, (0, 0, 0, y_pad))

            """
            if L > fixed_len:
                y = y[:fixed_len]
            else:
                pad_len = fixed_len - L
                y = nn.functional.pad(y, (0, 0, 0, pad_len))
            """

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
