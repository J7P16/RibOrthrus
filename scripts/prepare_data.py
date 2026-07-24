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
    Dataset containing Ribo, RNA, or interleaved Ribo/RNA targets for each transcript.
    """
    
    def __init__(self,
            ribo_dir: str, 
            read_length_min: int, 
            read_length_max: int,
            model_type: str = "ribo",
            transcript_ids: Optional[Sequence[str]]=None,
    ):
        self.ribo_dir = Path(ribo_dir)
        self.read_length_min = read_length_min
        self.read_length_max = read_length_max
        self.model_type = model_type
        self.ribo_paths = sorted(self.ribo_dir.glob("HEK293.ribo"))
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
            
            # log1p Normalization
            #coverage = {tx: np.log1p(arr) for tx, arr in coverage.items()}
            #coverage = {tx: 30.0 * arr for tx, arr in coverage.items()}
            coverage = {tx: arr for tx, arr in coverage.items()}

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
        
        print(f"Before GenomeKit filter: {len(self.transcript_ids)}")
        print("Building GenomeKit transcript set...")
        genome_tx_ids = {str(t.id) for t in Genome("gencode.v41").transcripts}
        valid = []
        for tx in self.transcript_ids:
            enst_id = str(tx).split("|")[0]
            if enst_id in genome_tx_ids:
                valid.append(tx)
        self.transcript_ids = valid
        print("Finished building GenomeKit transcript set!")
        print(f"After GenomeKit filter: {len(self.transcript_ids)}")
        
        print("Example ribo IDs:", [str(x).split('|')[0] for x in self.transcript_ids[:5]])
        print(f"Loaded {len(self.cell_lines)} Cell Lines:")
        for cell_line in self.cell_lines:
            print(f"    {cell_line}: experiment={self.experiments[cell_line]}")
        output_channels = len(self.cell_lines) * (2 if self.model_type == "mixed" else 1)
        print(f"Total Output Channels: {output_channels}")
    
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
                    # THIS NORMALIZATION STEP IS PRETTY MUCH A MUST
                    #rna_value = np.log1p(float(row.sum()))
                    rna_value = float(row.sum())
                except KeyError:
                    rna_value = 0.0
            track_rna = torch.full_like(rpf, rna_value)

            if self.model_type == "ribo":
                tracks.append(track_ribo)
            elif self.model_type == "rna":
                tracks.append(track_rna)
            else:
                tracks.extend([track_ribo, track_rna])

        y_true = torch.stack(tracks, dim=-1)
    
        enst_id = str(tx).split("|")[0]
        
        return {
            "transcript_id": tx,
            "y_true": y_true,
            "length": length
        }
