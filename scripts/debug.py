from ribopy import Ribo
from genome_kit import Genome
from pathlib import Path

ribo_paths = sorted(Path("data").glob("*.ribo"))
cell_lines = [path.stem for path in ribo_paths]
for cell_line, ribo_path in zip(cell_lines, ribo_paths):
    ribo = Ribo(str(ribo_path))
    print(cell_line, ribo.minimum_length, ribo.maximum_length)
