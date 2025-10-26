from Bio import SeqIO
def read_fasta(fasta_file):
    return [(r.id, str(r.seq)) for r in SeqIO.parse(fasta_file, "fasta")]
