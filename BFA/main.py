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
    audio_dir:          Annotated[Path, Parameter(help="Path to audio directory")],
    text_dir:           Annotated[Path, Parameter(help="Path to text directory")],
    out_dir:            Annotated[Path, Parameter(help="Path to output directory")] = DEFAULT_OUTPUT_PATH,
    dtype:              Annotated[Literal["words", "phonemes"], Parameter(help="Type of data contained in the text files")] = "words",
    ptype:              Annotated[Literal["IPA", "Misaki"], Parameter(help="Type of phoneme set used in the text files")] = "IPA",
    language:           Annotated[Literal["EN-GB", "EN-US"], Parameter(help="Language of the audio files")] = "EN-GB",
    n_jobs:             Annotated[int, Parameter(help="Number of parallel jobs to run. -1 for all available cores")] = -1,
    ignore_ram_usage:   Annotated[bool, Parameter(help="Don't check if you have enough RAM to run the number of jobs requested")] = False,
    config_path:        Annotated[Path, Parameter(help="Path to config file. Should not be used if you're not a contributor")] = DEFAULT_CONFIG_PATH,
):

    # Load config
    config = load_cfg(config_path, ROOT_DIR)
    if isinstance(config, Failure):
        print("Error loading config. Exiting...")
        print(f"Error: {config}")
        exit(1)

    # Align
    aligner = ForcedAligner(language, ignore_ram_usage, config)
    aligner.align_corpus(audio_dir, text_dir, out_dir, dtype, ptype, n_jobs)


if __name__ == "__main__":
    app()