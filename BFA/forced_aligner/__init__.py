import torch
import resource
from tqdm import tqdm
from torch import Tensor
from pathlib import Path
from functools import partial
from psutil import virtual_memory
from typing import Literal, Union, List
from multiprocessing import get_context, cpu_count

from .inference_engine import InferenceEngine
from .text_preprocessor import TextPreprocessor
from .audio_preprocessor import AudioPreprocessor
from .path_tracer import constrained_viterbi

from ..io import IOManager
from ..utils import (
	Failure,
	FilePair,
	RawAlignment,
	TranslatedAlignment,
	SharedLogger,
)

# For thread safety, prevent pytorch from using multiple threads
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Increase maximum file descriptors
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
new_soft = min(hard, 8192)
resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))

GO_RATIO = 1024**3



class ForcedAligner:
	def __init__(
		self,
		language: Literal["EN-GB", "EN-US"],
		ignore_ram_usage: bool,
		config: dict
	) -> None:

		self.config = config

		try:
			# Initialize components
			self.logger = SharedLogger.get_instance(config["logger"], reset=True)
			self.text_preprocessor = TextPreprocessor(language, config["text_preprocessor"])
			self.audio_preprocessor = AudioPreprocessor(config["audio_preprocessor"])
			self.inference_engine = InferenceEngine(config["inference_engine"])
			self.io_manager = IOManager(config["io_manager"])

		except Exception as e:
			print(f"Failed to initialize Forced Aligner. Exiting...")
			print(f"Error: {e}")
			exit(code=1)

		# Check available RAM
		max_jobs = int((virtual_memory().total / GO_RATIO) // self.config["ram_usage_per_thread"])
		self.max_jobs = cpu_count() if ignore_ram_usage else max_jobs

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
			file_pairs, unpaired_audios, unpaired_texts = self.io_manager.get_pairs(audio_dir, text_dir, out_dir)
			file_pairs: List[FilePair]

			if (unpaired_audios > 0) or (unpaired_texts > 0):
				self.logger.warning("Some files were not paired:")
				self.logger.warning(f"Audio files without annotations: {unpaired_audios}")
				self.logger.warning(f"Annotation files without audio: {unpaired_texts}")
			else:
				self.logger.info("All files were successfully paired.")

			# Start processing in parallel
			assert -1 <= n_jobs <= cpu_count(), "Invalid number of jobs specified."
			n_jobs = cpu_count() if n_jobs == -1 else n_jobs
			n_jobs = min(n_jobs, len(file_pairs), self.max_jobs)

			self.logger.info(f"Using {n_jobs} jobs for processing.")

			ctx = get_context("spawn")	# Use spawn for Windows compatibility
			with ctx.Pool(n_jobs, initializer=self.init_worker, initargs=(self.config,)) as pool:
				processed_files = 0
				failures = 0

				# Define the function to align a single pair
				align_pair_partial = partial(
					self.align_pair,
					dtype = dtype,
					ptype = ptype,
					out_dir = out_dir,
				)

				generator = pool.imap_unordered(align_pair_partial, file_pairs, chunksize=self.config["chunk_size"])

				# Wait for results
				with tqdm(generator, total=len(file_pairs)) as pbar:
					for i, result in enumerate(pbar):
						processed_files += 1

						if not result:
							failures += 1

						# Update progress bar
						pbar.set_postfix_str(f"failed alignments: {failures} | success rate: {(100 * (1 - failures/processed_files)):.2f}%")

			# Log the results
			if failures > 0:
				self.logger.warning(f"Alignment completed with {failures} failures out of {processed_files} files.")
			else:
				self.logger.info("All files were successfully aligned.")

			exit(code=0)

		except Exception as e:
			self.logger.error(f"Failed to Align Corpus. Cause: {e}")
			exit(code=1)


	def align_pair(
		self,
		files: FilePair,
		dtype: Literal["words", "phonemes"],
		ptype: Literal["IPA", "Misaki"],
		out_dir: Path,
	) -> bool:

		try:
			# 1) Preprocess text and audio files
			phonemes_tensor, phonemes_tensor_length = self.text_preprocessor.process_text(files["annotation"], dtype, ptype)
			audio_tensor, audio_tensor_length, audio_duration = self.audio_preprocessor.process_audio(files["audio"])

			# 1.5) Get word labels if necessary
			word_labels = self.text_preprocessor.get_word_labels(files["annotation"]) if dtype == "words" else None

			# 2) Predict alignments
			alignement_scores: Union[Tensor, Failure] = self.inference_engine.inference(
				audio_tensor,
				phonemes_tensor,
				audio_tensor_length,
				phonemes_tensor_length,
			)

			# 3) Trace alignment path
			alignment_path: Union[RawAlignment, Failure] = constrained_viterbi(
				alignement_scores[0],
				phonemes_tensor[0, 1:]
			)
			if isinstance(alignment_path, Failure):
				self.logger.error(f"Failed to trace alignment path for {files['audio']}. Cause: {alignment_path}", extra={"hidden": True})
				raise RuntimeError(f"Failed to trace alignment path for {files['audio']}. Cause: {alignment_path}")

			# 4) Translate aligned tokens back to phonemes
			translated_alignment: TranslatedAlignment = self.text_preprocessor.detokenize_alignment(alignment_path, ptype)

			# 5) Save the alignment to textgrid
			frame_duration = self.audio_preprocessor.frame_duration

			self.io_manager.alignment_to_textgrid(
				translated_alignment,
				audio_duration,
				frame_duration,
				files["output"],
				word_labels,
			)

			# All steps completed successfully
			self.logger.info(f"Alignment for {files['audio']} completed successfully.")
			return True

		except Exception as e:
			self.logger.error(f"Failed to align pair {files['audio']} and {files['annotation']}. Cause: {e}", extra={"hidden": True}, exc_info=e)
			return False


	@staticmethod
	def init_worker(config: dict) -> None:
		"""Initialize the worker process."""
		# Set the logger for the worker
		logger = SharedLogger.get_instance(config["logger"])
		logger.propagate = False