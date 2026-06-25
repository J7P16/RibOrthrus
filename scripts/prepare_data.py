from __future__ import print_function

# importing necessary libraries
import argparse
import gzip
import os
import random
import re
import subprocess
import sys
import tempfile

DEFAULT_GENCODE_VERSION = 48
GENCODE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
    "release_{version}/gencode.v{version}.transcripts.fa.gz"
)
# CCLE K562-tuned A-site offsets (same as iXnos CCLE runs)
SHIFT_DICT = {
    27: {0: 14, 1: False, 2: False},
    28: {0: 14, 1: False, 2: 15},
    29: {0: 14, 1: False, 2: 15},
    30: {0: 14, 1: False, 2: 15},
    31: {0: 14, 1: False, 2: 15},
}

# ---- reference building (from BAM @SQ + GENCODE) ----
def run_cmd(cmd):
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return out.decode("utf-8")

def parse_sq_line(line):
    if not line.startswith("@SQ"):
        return None
    ref, ln = None, None
    for field in line.split("\t"):
        if field.startswith("SN:"):
            ref = field[3:]
        elif field.startswith("LN:"):
            ln = int(field[3:])
    return (ref, ln) if ref else None

def parse_region(field):
    m = re.match(r"(UTR5|CDS|UTR3):(\d+)-(\d+)", field)
    return (int(m.group(3)) - int(m.group(2)) + 1) if m else None

def parse_ref_lengths(ref_name, total_len=None):
    utr5 = cds = utr3 = 0
    for part in ref_name.split("|"):
        if part.startswith("UTR5:"):
            utr5 = parse_region(part)
        elif part.startswith("CDS:"):
            cds = parse_region(part)
        elif part.startswith("UTR3:"):
            utr3 = parse_region(part)
    if cds is None:
        raise ValueError("No CDS in reference: {0}".format(ref_name[:80]))
    return utr5 or 0, cds, utr3 or 0

def parse_ref_metadata(ref_name):
    parts = ref_name.split("|")
    return parts[0], parts[5] if len(parts) > 5 else ""

def get_bam_sq_records(bam):
    records = []
    for line in run_cmd(["samtools", "view", "-H", bam]).splitlines():
        parsed = parse_sq_line(line)
        if parsed:
            records.append(parsed)
    return records

def ensure_gencode(cache_dir, version, download=True):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "gencode.v{0}.transcripts.fa.gz".format(version))
    if os.path.isfile(path):
        return path
    if not download:
        raise IOError("Missing GENCODE: {0}".format(path))
    url = GENCODE_URL.format(version=version)
    print("Downloading {0}".format(url))
    try:
        import urllib.request
        urllib.request.urlretrieve(url, path)
    except Exception:
        import urllib
        urllib.urlretrieve(url, path)
    return path

def load_gencode(path):
    seqs = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as fh:
        name, chunks = None, []
        for raw in fh:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(chunks)
                name = line[1:].split("|")[0]
                chunks = []
            else:
                chunks.append(line.upper())
        if name:
            seqs[name] = "".join(chunks)
    return seqs

def write_reference(out_dir, records, gencode):
    os.makedirs(out_dir, exist_ok=True)
    dna_path = os.path.join(out_dir, "transcripts.fa")
    rna_path = os.path.join(out_dir, "transcripts.rna.fa")
    len_path = os.path.join(out_dir, "transcript.lengths.txt")
    missing = []
    with open(dna_path, "w") as dna, open(rna_path, "w") as rna, open(len_path, "w") as lens:
        for ref_name, total_len in records:
            enst, _ = parse_ref_metadata(ref_name)
            utr5, cds, utr3 = parse_ref_lengths(ref_name, total_len)
            lens.write("{0}\t{1}\t{2}\t{3}\n".format(ref_name, utr5, cds, utr3))
            if enst not in gencode:
                missing.append(enst)
                continue
            seq = gencode[enst]
            dna.write(">{0}\n".format(ref_name))
            rna.write(">{0}\n".format(ref_name))
            for i in range(0, len(seq), 60):
                chunk = seq[i:i + 60]
                dna.write(chunk + "\n")
                rna.write(chunk.replace("T", "U") + "\n")
    print("Wrote {0} transcripts ({1} missing from GENCODE)".format(
        len(records) - len(missing), len(missing)))
    return dna_path, rna_path, len_path, missing

def validate_reads(bam, fasta, n=20, seed=0):
    seqs = {}
    with open(fasta) as fh:
        name, chunks = None, []
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if name:
            seqs[name] = "".join(chunks)
    random.seed(seed)
    frac = max(1e-4, n * 50 / 1e7)
    proc = subprocess.Popen(
        ["samtools", "view", "-s", str(frac), bam],
        stdout=subprocess.PIPE)
    lines = []
    for raw in proc.stdout:
        line = raw.decode().strip()
        if line and not line.startswith("@"):
            lines.append(line)
        if len(lines) >= n * 10:
            break
    proc.stdout.close()
    proc.wait()
    sample = random.sample(lines, min(n, len(lines)))
    ok = 0
    for line in sample:
        f = line.split("\t")
        ref, pos, read = f[2], int(f[3]), f[9]
        if ref not in seqs:
            continue
        start = pos - 1
        if start < 0 or start + len(read) > len(seqs[ref]):
            continue
        mm = sum(a != b for a, b in zip(read, seqs[ref][start:start + len(read)]))
        nm = next((int(x.split(":")[-1]) for x in f[11:] if x.startswith("NM:")), 2)
        if mm <= nm + 1:
            ok += 1
    print("Validation: {0}/{1} reads OK".format(ok, len(sample)))

def cmd_build_reference(args):
    records = get_bam_sq_records(args.bam)
    gencode = load_gencode(ensure_gencode(
        os.path.join(args.out_dir, "gencode_cache"),
        args.gencode_version, download=not args.no_download))
    _, _, _, missing = write_reference(args.out_dir, records, gencode)
    if missing:
        print("Warning: {0} transcripts absent from GENCODE".format(len(missing)))
    if not args.skip_validation:
        validate_reads(args.bam, os.path.join(args.out_dir, "transcripts.fa"))

# ---- label extraction from BAM ----
def load_len_dict(path):
    d = {}
    with open(path) as fh:
        for line in fh:
            gene, u5, cds, u3 = line.strip().split("\t")
            d[gene] = {"utr5": int(u5), "cds": int(cds), "utr3": int(u3)}
    return d

def load_fasta(path):
    d = {}
    with open(path) as fh:
        name, chunks = None, []
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    d[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.upper())
        if name:
            d[name] = "".join(chunks)
    return d

def get_cds_dict(fasta, len_dict):
    full = load_fasta(fasta)
    return {
        g: full[g][d["utr5"]:d["utr5"] + d["cds"]]
        for g, d in len_dict.items() if g in full
    }

def add_simple_map_weights(sam_in, sam_out):
    """1/n weight per read mapping (iXnos simple_wts)."""
    with open(sam_in) as fin, open(sam_out, "w") as fout:
        buf = []
        for line in fin:
            if line.startswith("@"):
                fout.write(line)
                continue
            fields = line.rstrip("\n").split("\t")
            if len(buf) == 0 or fields[0] != buf[0][0]:
                if buf:
                    w = 1.0 / len(buf)
                    for f in buf:
                        fout.write("\t".join(f) + "\t{0}\n".format(w))
                buf = [fields]
            else:
                buf.append(fields)
        if buf:
            w = 1.0 / len(buf)
            for f in buf:
                fout.write("\t".join(f) + "\t{0}\n".format(w))

def bam_to_weighted_sam(bam):
  """BAM -> filtered SAM with simple mapping weights."""
  tmp_in = tempfile.NamedTemporaryFile(suffix=".sam", delete=False)
  tmp_in.close()
  tmp_out = tempfile.NamedTemporaryFile(suffix=".sam", delete=False)
  tmp_out.close()
  with open(tmp_in.name, "w") as fh:
      subprocess.check_call(["samtools", "view", "-h", "-F", "4", bam], stdout=fh)
  add_simple_map_weights(tmp_in.name, tmp_out.name)
  os.unlink(tmp_in.name)
  return tmp_out.name

def get_cts_by_codon(sam, cds_dict, len_dict, min_fp, max_fp):
    cts = {g: [0.0] * (len(s) // 3) for g, s in cds_dict.items()}
    with open(sam) as fh:
        for line in fh:
            if line.startswith("@"):
                continue
            f = line.strip().split("\t")
            read = f[9]
            if len(read) < min_fp or len(read) > max_fp:
                continue
            gene = f[2]
            if gene not in cts:
                continue
            try:
                wt = float(f[-1])
            except ValueError:
                wt = 1.0
            u5 = len_dict[gene]["utr5"]
            map_nt = int(f[3]) - 1 - u5
            shift = SHIFT_DICT[len(read)].get(map_nt % 3)
            if not shift:
                continue
            codon = (map_nt + shift) // 3
            if 0 <= codon < len(cts[gene]):
                cts[gene][codon] += wt
    return cts

def scaled_densities(cts, trunc_5p, trunc_3p):
    """iXnos get_outputs: zero ends, divide by gene-mean in interior."""
    out = {}
    for gene, vals in cts.items():
        v = list(vals)
        n = len(v)
        interior = n - trunc_5p - trunc_3p
        if interior <= 0:
            continue
        for i in list(range(trunc_5p)) + list(range(n - trunc_3p, n)):
            v[i] = 0.0
        total = sum(v)
        if total == 0:
            continue
        avg = float(total) / interior
        out[gene] = [x / avg for x in v]
    return out

def gene_passes_qc(cts, trunc_5p, trunc_3p, min_cts, min_cod):
    interior = cts[trunc_5p:-trunc_3p] if len(cts) > trunc_5p + trunc_3p else []
    return sum(interior) >= min_cts and sum(1 for x in interior if x > 0) >= min_cod

def write_labels(out_path, cts, scaled, len_dict, cds_dict,
                 trunc_5p, trunc_3p, interior_only, min_cts, min_cod):
    n_rows = 0
    with open(out_path, "w") as out:
        out.write(
            "transcript\tenst\tgene_symbol\tcodon_idx\t"
            "transcript_nt_pos\tcodon_triplet\t"
            "raw_count\tscaled_density\tis_interior\n")
        for gene in sorted(scaled):
            if not gene_passes_qc(cts[gene], trunc_5p, trunc_3p, min_cts, min_cod):
                continue
            enst, sym = parse_ref_metadata(gene)
            u5 = len_dict[gene]["utr5"]
            cds = cds_dict.get(gene, "")
            for i, density in enumerate(scaled[gene]):
                is_interior = (trunc_5p <= i < len(scaled[gene]) - trunc_3p)
                if interior_only and not is_interior:
                    continue
                triplet = cds[i * 3:(i + 1) * 3] if cds else ""
                nt_pos = u5 + i * 3 + 1  # 1-based start in full transcript
                out.write(
                    "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(
                        gene, enst, sym, i, nt_pos, triplet,
                        cts[gene][i], density, int(is_interior)))
                n_rows += 1
    return n_rows

def cmd_extract_labels(args):
    ref = args.reference_dir
    rna_fa = os.path.join(ref, "transcripts.rna.fa")
    dna_fa = os.path.join(ref, "transcripts.fa")
    len_file = os.path.join(ref, "transcript.lengths.txt")
    fasta = rna_fa if os.path.isfile(rna_fa) else dna_fa
    for p in (fasta, len_file):
        if not os.path.isfile(p):
            sys.exit("Missing: {0}".format(p))
    os.makedirs(args.out_dir, exist_ok=True)
    sam = bam_to_weighted_sam(args.bam)
    try:
        len_dict = load_len_dict(len_file)
        cds_dict = get_cds_dict(fasta.replace(".rna.fa", ".fa") if fasta.endswith(".rna.fa") else fasta, len_dict)
        if not cds_dict and os.path.isfile(dna_fa):
            cds_dict = get_cds_dict(dna_fa, len_dict)
        cts = get_cts_by_codon(
            sam, cds_dict, len_dict, args.min_fp, args.max_fp)
        scaled = scaled_densities(cts, args.cod_trunc_5p, args.cod_trunc_3p)
        labels_path = os.path.join(args.out_dir, "labels.tsv")
        n = write_labels(
            labels_path, cts, scaled, len_dict, cds_dict,
            args.cod_trunc_5p, args.cod_trunc_3p,
            args.interior_only, args.min_cts, args.min_cod)
        print("Wrote {0} rows to {1}".format(n, labels_path))
    finally:
        os.unlink(sam)

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("build-reference")
    r.add_argument("--bam", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--gencode-version", type=int, default=DEFAULT_GENCODE_VERSION)
    r.add_argument("--no-download", action="store_true")
    r.add_argument("--skip-validation", action="store_true")
    r.set_defaults(func=cmd_build_reference)
    e = sub.add_parser("extract-labels")
    e.add_argument("--bam", required=True)
    e.add_argument("--reference-dir", required=True)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--min-fp", type=int, default=27)
    e.add_argument("--max-fp", type=int, default=31)
    e.add_argument("--cod-trunc-5p", type=int, default=20)
    e.add_argument("--cod-trunc-3p", type=int, default=20)
    e.add_argument("--min-cts", type=int, default=200,
                     help="Min interior footprint counts per gene")
    e.add_argument("--min-cod", type=int, default=100,
                     help="Min interior codons with count > 0")
    e.add_argument("--interior-only", action="store_true", default=True)
    e.add_argument("--all-codons", action="store_true",
                   help="Include terminal codons (overrides --interior-only)")
    e.set_defaults(func=cmd_extract_labels)
    args = p.parse_args()
    if getattr(args, "all_codons", False):
        args.interior_only = False
    args.func(args)
    
if __name__ == "__main__":
    main()