import os
import typer
from typing import Literal

from BFA.forced_aligner.model import InferenceEngine
from BFA.forced_aligner.text_preprocessor import TextPreprocessor
from BFA.forced_aligner.audio_preprocessor import AudioPreprocessor

from BFA.forced_aligner.path_tracer import constrained_viterbi

from BFA.io import TextGridWriter
from BFA.utils import Failure, Alignment, get_logger



class ForcedAligner:
	def __init__(self, language: Literal["EN-GB", "EN-US"], config: dict) -> None:
		self.config = config

		# Initialize logger
		self.logger = get_logger(config["logger"])
		if isinstance(self.logger, Failure):
			raise RuntimeError(self.logger)

		# Initialize components
		self.text_preprocessor = TextPreprocessor(language, config["text_preprocessor"])
		self.audio_preprocessor = AudioPreprocessor(config["audio_preprocessor"])
		self.inference_engine = InferenceEngine(config["inference_engine"])
		self.textgrid_writer = TextGridWriter(config["textgrid_writer"])

		self.logger.info("Forced Aligner initialized successfully.")
		self.logger.info(f"Language: {language}")


	def align_corpus(
		self, wav_dir: str, lab_dir: str, out_dir: str,
		dtype: Literal["words", "phonemes"],
		ptype: Literal["Misaki", "ARPAbet", "IPA"],
		n_jobs: int,
	) -> None:

		pass