import typer
from tqdm import tqdm
from pathlib import Path
from functools import partial
from typing import Literal, List, Dict
from multiprocessing import Pool, cpu_count

from .model import InferenceEngine
from .text_preprocessor import TextPreprocessor
from .audio_preprocessor import AudioPreprocessor
from .path_tracer import constrained_viterbi

from BFA.io import IOManager
from BFA.utils import (
	Failure,
	RawAlignment,
	TranslatedAlignment,
	get_logger
)



class ForcedAligner:
	def __init__(self, language: Literal["EN-GB", "EN-US"], config: dict) -> None:
		self.config = config

		try:
			# Initialize components
			self.logger = get_logger(config["logger"])
			self.text_preprocessor = TextPreprocessor(language, config["text_preprocessor"])
			self.audio_preprocessor = AudioPreprocessor(config["audio_preprocessor"])
			self.inference_engine = InferenceEngine(config["inference_engine"])
			self.io_manager = IOManager(config["io_manager"])

		except Exception as e:
			typer.echo(f"Failed to initialize Forced Aligner. Exiting...")
			typer.echo(f"Error: {e}")
			raise typer.Exit(code=1)

		self.logger.info("Forced Aligner initialized successfully.")
		self.logger.info(f"Language: {language}")


	def align_corpus(
		self, audio_dir: Path, text_dir: Path, out_dir: Path,
		dtype: Literal["words", "phonemes"],
		ptype: Literal["IPA", "Misaki"],
		n_jobs: int,
	) -> None:

		try:
			# Find audio/annotation pairs
			file_pairs, unpaired_audios, unpaired_text = self.io_manager.get_pairs(audio_dir, text_dir)
			file_pairs: List[Dict[str, Path]]

			if (unpaired_audios > 0) or (unpaired_text > 0):
				self.logger.warning("Some files were not paired:")
				self.logger.warning(f"Audio files without annotations: {unpaired_audios}")
				self.logger.warning(f"Annotation files without audio: {unpaired_text}")
			else:
				self.logger.info("All files were successfully paired.")

			# Create output directory if it doesn't exist
			if not out_dir.exists():
				out_dir.mkdir(parents=True, exist_ok=True)

			# Start processing in parallel
			assert -1 <= n_jobs <= cpu_count(), "Invalid number of jobs specified."
			n_jobs = cpu_count() if n_jobs == -1 else n_jobs
			with Pool(n_jobs) as pool:
				processed_files = 0
				failures = 0

				# Define the function to align a single pair
				align_pair_partial = partial(
					self.align_pair,
					dtype = dtype,
					ptype = ptype,
					out_dir = out_dir,
				)

				generator = pool.imap(align_pair_partial, file_pairs)

				# Wait for results
				with tqdm(generator, total=len(file_pairs)) as pbar:
					for i, result in enumerate(pbar):
						processed_files += 1
						if isinstance(result, Failure):
							# Log individual errors to the log file only
							audio, annotation = file_pairs[i]["audio"], file_pairs[i]["annotation"]
							self.logger.error(f"Failed to align file pair {audio} | {annotation}. Cause: {result}", extra={"hidden": True})
							failures += 1

						# Update progress bar
						pbar.set_postfix(f"failed alignments: {failures} | success rate: {failures/processed_files}")

			# Log the results
			if failures > 0:
				self.logger.warning(f"Alignment completed with {failures} failures out of {processed_files} files.")
			else:
				self.logger.info("All files were successfully aligned.")

			typer.echo(f"Alignment completed with {failures} failures out of {processed_files} files.")
			raise typer.Exit(code=0)

		except Exception as e:
			self.logger.error(f"Failed to Align Corpus. Cause: {e}")
			raise typer.Exit(code=1)


	def align_pair(
		self,
		files: Dict[str, Path],
		dtype: Literal["words", "phonemes"],
		ptype: Literal["IPA", "Misaki"],
		out_dir: Path,
	) -> None:

		raise NotImplementedError("The align_pair method is not implemented yet.")