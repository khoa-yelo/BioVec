import typer
from .embedders.protein_embedder import ProteinEmbedder
from .index.biovector_db import BioVectorDB

app = typer.Typer(help="BioVector CLI — embed and search protein sequences")

@app.command()
def build(fasta: str,
          model: str = typer.Option("esm2_t33_650M", "--model", "-m"),
          out: str = typer.Option("biovector_index", "--out", "-o")):
    """Build a BioVector index from a FASTA file."""
    emb = ProteinEmbedder(model=model)
    db = BioVectorDB()
    db.add_fasta(fasta, emb)
    db.save(out)
    typer.echo(f"[BioVector] Indexed {db.ntotal} sequences → {out}.faiss / {out}.meta.json")

@app.command()
def query(seq: str = typer.Option(None, "--seq", "-s"),
          model: str = typer.Option("esm2_t33_650M", "--model", "-m"),
          index: str = typer.Option("biovector_index", "--index", "-i"),
          top: int = typer.Option(10, "--top", "-k")):
    """Query a BioVector index with an amino acid sequence string."""
    if not seq:
        raise typer.BadParameter("Please provide --seq with an amino acid sequence.")
    emb = ProteinEmbedder(model=model)
    db = BioVectorDB()
    db.load(index)
    hits = db.search(seq, emb, top_k=top)
    for h in hits:
        typer.echo(f"{h['rank']:>2}. {h['id']}  dist={h['distance']:.6f}")

if __name__ == "__main__":
    app()
