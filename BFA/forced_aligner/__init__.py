import os
import typer
from typing import Literal

from .model import InferenceEngine
from .text_preprocessor import TextPreprocessor
from .audio_preprocessor import AudioPreprocessor
from .path_tracer import constrained_viterbi

from BFA.io import TextGridWriter
from BFA.utils import Failure, Alignment, get_logger



class ForcedAligner:
	def __init__(self, language: Literal["EN-GB", "EN-US"], config: dict) -> None:
		self.config = config

		try:
			# Initialize components
			self.logger = get_logger(config["logger"])
			self.text_preprocessor = TextPreprocessor(language, config["text_preprocessor"])
			self.audio_preprocessor = AudioPreprocessor(config["audio_preprocessor"])
			self.inference_engine = InferenceEngine(config["inference_engine"])
			self.textgrid_writer = TextGridWriter(config["textgrid_writer"])

		except Exception as e:
			typer.echo(f"Failed to initialize Forced Aligner. Exiting...")
			typer.echo(f"Error: {e}")
			typer.Exit(code=1)

		self.logger.info("Forced Aligner initialized successfully.")
		self.logger.info(f"Language: {language}")


	def align_corpus(
		self, wav_dir: str, lab_dir: str, out_dir: str,
		dtype: Literal["words", "phonemes"],
		ptype: Literal["IPA", "Misaki"],
		n_jobs: int,
	) -> None:

		pass