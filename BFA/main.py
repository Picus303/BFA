import typer
from pathlib import Path
from typing import Literal

from BFA.utils import Failure, load_cfg
from BFA.forced_aligner import ForcedAligner

DEFAULT_CONFIG_PATH = Path("config.yaml")

app = typer.Typer(add_completion=False)


@app.command()
def align(
	audio_dir: Path = typer.Option(..., exists=True, readable=True),
	text_dir: Path = typer.Option(..., exists=True, readable=True),
	out_dir: Path = typer.Option("out/", writable=True),
	dtype: Literal["words", "phonemes"] = typer.Option("words"),
	ptype: Literal["IPA", "Misaki"] = typer.Option("IPA"),
	language: Literal["EN-GB", "EN-US"] = typer.Option("EN-GB"),
	n_jobs: int = typer.Option(-1),
	config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, exists=True, readable=True),
):
	# Get project root directory
	root = Path(__file__).parent if config_path == DEFAULT_CONFIG_PATH else Path('.')
	config = load_cfg(config_path, root)

	# Check if the config is valid
	if isinstance(config, Failure):
		typer.echo(f"Error loading config. Exiting...")
		typer.echo(f"Error: {config}")
		raise typer.Exit(code=1)

	# Align corpus with config
	aligner = ForcedAligner(language, config)
	aligner.align_corpus(audio_dir, text_dir, out_dir, dtype, ptype, n_jobs)


if __name__ == "__main__":
	app()