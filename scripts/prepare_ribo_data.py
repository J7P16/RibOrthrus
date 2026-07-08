from ribopy import Ribo
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import Optional, Sequence
from genome_kit import Genome
import numpy as np

class MultiCellRiboDataset(Dataset):
    """
    Dataset that contains y_true tensors with shape: [L, 2 * NUM_OF_CELL_LINES] for each transcript 
    For context, advisor told me there are two prediction tracks for the Transcriptome data.

    Track Layout:
        channel 0: cell_line_0 track_0
        channel 1: cell_line_0 track_1
        channel 2: cell_line_1 track_0
        channel 3: cell_line_1 track_1

    *** Just duplicating track_0 to track_1 until I email advisor for specifics of the outputs ***
    """
    
    def __init__(self,
            ribo_dir: str, 
            read_length_min: int, 
            read_length_max: int,
            transcript_ids: Optional[Sequence[str]]=None,
            normalize_per_track: bool=False
    ):
        self.ribo_dir = Path(ribo_dir)
        self.read_length_min = read_length_min
        self.read_length_max = read_length_max
        self.normalize_per_track = normalize_per_track
        self.ribo_paths = sorted(self.ribo_dir.glob("*.ribo"))
        self.cell_lines = [path.stem for path in self.ribo_paths]
        
        # data/feature lists of our generated Dataset
        self.ribos = {}
        self.rnas = {}
        self.experiments = {}
        self.coverages = {}
        self.transcript_lengths = {}

        for cell_line, ribo_path in zip(self.cell_lines, self.ribo_paths):
            ribo = Ribo(str(ribo_path))
            experiment = ribo.experiments[0]
            min_read = max(self.read_length_min, ribo.minimum_length)
            max_read = min(self.read_length_max, ribo.maximum_length)
            coverage = ribo.get_coverage(experiment, min_read, max_read, False)    
            
            # Count-Per-Million (CPM) Normalization
            """
            library_size = sum(arr.sum() for arr in coverage.values())
            scale = 1000000.0 / max(library_size, 1.0)
            coverage = {tx: np.log1p(arr * scale) for tx, arr in coverage.items()}
            """
            # log1p Normalization
            coverage = {tx: np.log1p(arr) for tx, arr in coverage.items()}
            
            if ribo.has_rnaseq(experiment):
                rnaseq = ribo.get_rnaseq(experiment)
            else:
                rnaseq = None
                print(f"Warning: {cell_line} has no RNA-seq data")

            self.ribos[cell_line] = ribo
            self.rnas[cell_line] = rnaseq
            self.experiments[cell_line] = experiment
            self.coverages[cell_line] = coverage
            self.transcript_lengths[cell_line] = ribo.transcript_lengths

        if transcript_ids is None:
            common = set(self.ribos[self.cell_lines[0]].transcript_names)
            for cell_line in self.cell_lines[1:]:
                common &= set(self.ribos[cell_line].transcript_names)
            transcript_ids = sorted(common) 
        self.transcript_ids = list(transcript_ids)
        
        print("Building GenomeKit transcript set...")
        genome_tx_ids = {str(t.id) for t in Genome("gencode.v41").transcripts}
        valid = []
        for tx in self.transcript_ids:
            enst_id = str(tx).split("|")[0]
            if enst_id in genome_tx_ids:
                valid.append(tx)
        self.transcript_ids = valid

        max_transcript_len = 20000

        length_filtered = []
        skipped_long = 0

        for tx in self.transcript_ids:
            lengths = [
                self.transcript_lengths[cell_line][tx]
                for cell_line in self.cell_lines
                if tx in self.transcript_lengths[cell_line]
            ]

            L = max(lengths)

            if L > max_transcript_len:
                skipped_long += 1
                continue

            length_filtered.append(tx)

        self.transcript_ids = length_filtered

        print("Finished building GenomeKit transcript set!")
        print(f"Before GenomeKit filter: {len(self.transcript_ids)}")
        print(f"After GenomeKit filter: {len(self.transcript_ids)}")
        print("Example ribo IDs:", [str(x).split('|')[0] for x in self.transcript_ids[:5]])
        print(f"Shared Transcripts: {len(self.transcript_ids)}")
        print(f"Loaded {len(self.cell_lines)} Cell Lines:")
        for cell_line in self.cell_lines:
            print(f"    {cell_line}: experiment={self.experiments[cell_line]}")
        print(f"Output Channels: {len(self.cell_lines)}")
    
    def __len__(self):
        return len(self.transcript_ids)
    
    def __getitem__(self, index):
        tx = self.transcript_ids[index]
        tracks = []
        length = None

        for cell_line in self.cell_lines:
            coverage = self.coverages[cell_line]
            rpf = torch.tensor(coverage[tx], dtype=torch.float32)
            expected_len = self.transcript_lengths[cell_line][tx]
            
            if length is None:
                length = rpf.numel()

            if self.normalize_per_track:
                total = rpf.sum()
                if total > 0:
                    rpf = rpf / total
            
            # First Track: RPF nucleotide coverage
            track_ribo = rpf # First Track - RPF coverage    
            
            # Second Track: RNA-seq transcript-level value repeated across transcript length
            rnaseq_df = self.rnas[cell_line]
            experiment = self.experiments[cell_line]
            if rnaseq_df is None:
                rna_value = 0.0
            else:
                try:
                    row = rnaseq_df.loc[(experiment, tx)]
                    #rna_value = float(row.sum()) # UTR5 + CDS + UTR3
                    # THIS NORMALIZATION STEP IS A MUST
                    rna_value = np.log1p(float(row.sum()))
                except KeyError:
                    rna_value = 0.0
            track_rna = torch.full_like(rpf, rna_value)

            tracks.extend([track_ribo, track_rna])

        y_true = torch.stack(tracks, dim=-1)

        return {
            "transcript_id": tx,
            "y_true": y_true,
            "length": length
        }

