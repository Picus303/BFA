import typer
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

from BFA.io import IOManager
from BFA.utils import (
	Failure,
	FilePair,
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
			file_pairs, unpaired_audios, unpaired_texts = self.io_manager.get_pairs(audio_dir, text_dir)
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
							failures += 1

						# Update progress bar
						pbar.set_postfix_str(f"failed alignments: {failures} | success rate: {failures/processed_files}")

			# Log the results
			if failures > 0:
				self.logger.warning(f"Alignment completed with {failures} failures out of {processed_files} files.")
			else:
				self.logger.info("All files were successfully aligned.")

			raise typer.Exit(code=0)

		except Exception as e:
			self.logger.error(f"Failed to Align Corpus. Cause: {e}")
			raise typer.Exit(code=1)


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
				audio_tensor, audio_tensor_length = audio_preprocessing_result

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
				alignement_scores,
				phonemes_tensor[0, 1:]
			)
			if isinstance(alignment_path, Failure):
				self.logger.error(f"Failed to trace alignment path for {files['audio']}. Cause: {alignment_path}", extra={"hidden": True})
				return alignment_path

			# Translate aligned tokens to phonemes
			translated_alignment: TranslatedAlignment = []
			for t, u, emit in alignment_path:
				if emit is not None:
					translated_phoneme = self.text_preprocessor.detokenize(emit, ptype)
					if isinstance(translated_phoneme, Failure):
						self.logger.error(f"Failed to detokenize phoneme {emit}. Cause: {translated_phoneme}", extra={"hidden": True})
						return translated_phoneme

					translated_alignment.append((t, u, translated_phoneme))
				else:
					translated_alignment.append((t, u, None))

			# Save the alignment to textgrid
			frame_duration = 1 / self.config["audio_preprocessor"]["sample_rate"]
			audio_duration = audio_tensor_length.item() * frame_duration
			output_path = out_dir / (files["audio"].stem + ".TextGrid")

			export_result = self.io_manager.alignment_to_textgrid(
				translated_alignment,
				audio_duration,
				frame_duration,
				output_path,
				# to do: word_labels
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