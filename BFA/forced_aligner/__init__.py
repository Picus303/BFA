import torch
from tqdm import tqdm
from torch import Tensor
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Literal, Optional, Union, List

from .model import InferenceEngine
from .text_preprocessor import TextPreprocessor
from .audio_preprocessor import AudioPreprocessor
from .path_tracer import constrained_viterbi

from ..io import IOManager
from ..utils import (
	Failure,
	FilePair,
	RawAlignment,
	TranslatedAlignment,
	get_logger
)

# For thread safety, prevent pytorch from using multiple threads
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


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
			print(f"Failed to initialize Forced Aligner. Exiting...")
			print(f"Error: {e}")
			exit(code=1)

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
			n_jobs = min(n_jobs, len(file_pairs))
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

				generator = pool.imap(align_pair_partial, file_pairs, chunksize=self.config["chunk_size"])

				# Wait for results
				with tqdm(generator, total=len(file_pairs)) as pbar:
					for i, result in enumerate(pbar):
						processed_files += 1

						if isinstance(result, Failure):
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
	) -> Optional[Failure]:

		# to do: allow modules to access the logger and remove all these "isinstance" checks

		try:
			# Preprocess text and audio files
			text_preprocessing_result = self.text_preprocessor.process_text(files["annotation"], dtype, ptype)
			audio_preprocessing_result = self.audio_preprocessor.process_audio(files["audio"])

			# Check if text preprocessing was successful
			if isinstance(text_preprocessing_result, Failure):
				self.logger.error(f"Failed to process text file {files['annotation']}. Cause: {text_preprocessing_result}", extra={"hidden": True})
				return text_preprocessing_result
			else:
				phonemes_tensor, phonemes_tensor_length = text_preprocessing_result

			# Check if audio preprocessing was successful
			if isinstance(audio_preprocessing_result, Failure):
				self.logger.error(f"Failed to process audio file {files['audio']}. Cause: {audio_preprocessing_result}", extra={"hidden": True})
				return audio_preprocessing_result
			else:
				audio_tensor, audio_tensor_length, audio_duration = audio_preprocessing_result

			# Get word labels if necessary
			# Note: Reuses code from text preprocessing so don't need to check if it was successful
			word_labels = self.text_preprocessor.get_word_labels(files["annotation"]) if dtype == "words" else None

			# Predict alignments
			alignement_scores: Union[Tensor, Failure] = self.inference_engine.inference(
				audio_tensor,
				phonemes_tensor,
				audio_tensor_length,
				phonemes_tensor_length,
			)
			if isinstance(alignement_scores, Failure):
				self.logger.error(f"Failed to predict alignments for {files['audio']}. Cause: {alignement_scores}", extra={"hidden": True})
				return alignement_scores

			# Trace alignment path
			alignment_path: Union[RawAlignment, Failure] = constrained_viterbi(
				alignement_scores[0],
				phonemes_tensor[0, 1:]
			)
			if isinstance(alignment_path, Failure):
				self.logger.error(f"Failed to trace alignment path for {files['audio']}. Cause: {alignment_path}", extra={"hidden": True})
				return alignment_path

			# Translate aligned tokens to phonemes
			translated_alignment: Union[TranslatedAlignment, Failure] = self.text_preprocessor.detokenize_alignment(alignment_path, ptype)
			if isinstance(translated_alignment, Failure):
				self.logger.error(f"Failed to translate alignment for {files['audio']}. Cause: {translated_alignment}", extra={"hidden": True})
				return translated_alignment

			# Save the alignment to textgrid
			frame_duration = self.audio_preprocessor.frame_duration

			export_result = self.io_manager.alignment_to_textgrid(
				translated_alignment,
				audio_duration,
				frame_duration,
				files["output"],
				word_labels,
			)
			if isinstance(export_result, Failure):
				self.logger.error(f"Failed to export alignment to textgrid for {files['audio']}. Cause: {export_result}", extra={"hidden": True})
				return export_result

			# All steps completed successfully
			self.logger.info(f"Alignment for {files['audio']} completed successfully.")
			return None

		except Exception as e:
			self.logger.error(f"Failed to align pair {files['audio']} and {files['annotation']}. Cause: {e}", extra={"hidden": True})
			return Failure(f"Failed to align pair {files['audio']} and {files['annotation']}. Cause: {e}")