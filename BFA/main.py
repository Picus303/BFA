import typer
from pathlib import Path
from typing import Literal

from BFA.utils import Failure, load_cfg
from BFA.forced_aligner import ForcedAligner

CONFIG_PATH = "config.yaml"

app = typer.Typer(add_completion=False)


@app.command()
def align(
	wav_dir: Path = typer.Option(..., exists=True, readable=True),
	lab_dir: Path = typer.Option(..., exists=True, readable=True),
	out_dir: Path = typer.Option("out/", writable=True),
	dtype: Literal["words", "phonemes"] = typer.Option("words"),
	ptype: Literal["IPA", "Misaki"] = typer.Option("IPA"),
	language: Literal["EN-GB", "EN-US"] = typer.Option("EN-GB"),
	n_jobs: int = typer.Option(-1),
):
	# Get project root directory
	root = Path(__file__).parent
	config = load_cfg(CONFIG_PATH, root)

	# Check if the config is valid
	if isinstance(config, Failure):
		typer.echo(f"Error loading config: {config}")
		raise typer.Exit(code=1)

	aligner = ForcedAligner(language, config)
	aligner.align_corpus(wav_dir, lab_dir, out_dir, dtype, ptype, n_jobs)


if __name__ == "__main__":
	app()