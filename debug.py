from genome_kit import Genome

genome = Genome("gencode.v41")

print("ENST00000637513.2" in genome.transcripts)
print("ENST00000637513" in genome.transcripts)
