import typer
from pathlib import Path

from BFA.utils import load_cfg
from BFA.forced_aligner import ForcedAligner

app = typer.Typer(add_completion=False)


@app.command()
def align(
    wav_dir: Path = typer.Option(..., exists=True, readable=True),
    lab_dir: Path = typer.Option(..., exists=True, readable=True),
    out_dir: Path = typer.Option("out/", writable=True),
    cfg: Path = typer.Option("config.yaml", exists=True)
):
    """Aligne des fichiers WAV + LAB et écrit des TextGrid."""
    aligner = ForcedAligner.from_config(load_cfg(cfg))
    aligner.align_corpus(wav_dir, lab_dir, out_dir)


if __name__ == "__main__":
    app()