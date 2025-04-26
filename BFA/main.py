import typer
from pathlib import Path

from BFA.utils import Failure, load_cfg
from BFA.forced_aligner import ForcedAligner

app = typer.Typer(add_completion=False)


@app.command()
def align(
	wav_dir: Path = typer.Option(..., exists=True, readable=True),
	lab_dir: Path = typer.Option(..., exists=True, readable=True),
	out_dir: Path = typer.Option("out/", writable=True),
	cfg: Path = typer.Option("config.yaml", exists=True)
):
	root = Path(__file__).parent
	config = load_cfg(cfg, root)
	
	# Check if the config is valid
	if isinstance(config, Failure):
		typer.echo(f"Error loading config: {config}")
		raise typer.Exit(code=1)

	aligner = ForcedAligner(config)
	aligner.align_corpus(wav_dir, lab_dir, out_dir)


if __name__ == "__main__":
	app()