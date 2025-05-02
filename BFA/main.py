from pathlib import Path
from typing import Literal, Annotated

from cyclopts import App, Parameter
from .utils import Failure, load_cfg
from .forced_aligner import ForcedAligner

ROOT_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"
DEFAULT_OUTPUT_PATH = Path("out/")

app = App()


@app.command
def align(
    audio_dir: Annotated[Path, Parameter(help="Path to audio directory")],
    text_dir: Annotated[Path, Parameter(help="Path to text directory")],
    out_dir: Annotated[Path, Parameter(help="Path to output directory")] = DEFAULT_OUTPUT_PATH,
    dtype: Literal["words", "phonemes"] = "words",
    ptype: Literal["IPA", "Misaki"] = "IPA",
    language: Literal["EN-GB", "EN-US"] = "EN-GB",
    n_jobs: int = -1,
    config_path: Path = DEFAULT_CONFIG_PATH,
):
    # Load config
    config = load_cfg(config_path, ROOT_DIR)
    if isinstance(config, Failure):
        print("Error loading config. Exiting...")
        print(f"Error: {config}")
        exit(1)

    # Align
    aligner = ForcedAligner(language, config)
    aligner.align_corpus(audio_dir, text_dir, out_dir, dtype, ptype, n_jobs)


if __name__ == "__main__":
    app()