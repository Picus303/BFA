from pathlib import Path
from logging import Logger
from typing import Optional, List, Tuple, Dict

from .utils import (
	FilePair,
	TranslatedAlignment,
	SharedLogger,
)

Durations = List[Tuple[str, float]]
Intervals = List[Tuple[float, float, str]]


class IOManager:
	def __init__(self, config: dict):
		self.config = config
		self.logger: Logger = SharedLogger.get_instance()
		self.logger.info("IO manager initialized successfully.")


	def get_pairs(self, audio_dir: Path, text_dir: Path, out_dir: Path) -> Tuple[List[FilePair], int, int]:
		"""
		Recursively traverses audio_dir and text_dir, and constructs the list of
		audio/annotation pairs based on the relative structure and supported extensions.
		It also produces the output path for each pair.

		Also returns the numbers of file that couldn't be paired.
		"""

		def collect(base_dir: Path, exts: set) -> Dict[Path, Path]:
			"""Returns a dict mapping "rel/path/without_ext" → "abs_path.ext"."""
			mapping = {}
			for file in base_dir.rglob("*"):
				if file.is_file() and file.suffix.lower() in exts:
					# Key = path without the extension
					key = file.relative_to(base_dir).with_suffix("")
					mapping[key] = file.resolve()

			return mapping

		# Collect files in both directories
		audio_map = collect(audio_dir, self.config["supported_audio_formats"])
		text_map = collect(text_dir, self.config["supported_annotation_formats"])

		keys_audio = set(audio_map.keys())
		keys_text = set(text_map.keys())

		# Intersections et differences
		common_keys	= keys_audio & keys_text
		only_audio = keys_audio - keys_text
		only_text = keys_text - keys_audio

		# Building the list of pairs
		pairs = [
			{"audio": audio_map[k], "annotation": text_map[k], "output": out_dir / k.with_suffix(self.config["output_format"])}
			for k in sorted(common_keys)
		]

		return pairs, len(only_audio), len(only_text)


	def alignment_to_durations(self, alignment: TranslatedAlignment, frame_duration: float) -> Durations:
		# Count frames for each phoneme
		counts = []
		current_frame_phonemes = []
		current_phoneme = self.config["sos_token"]
		current_phoneme_count = 0
		for t, u, emitted_token in alignment:
			if emitted_token is not None:
				# Don't have a phoneme that lasts 0 frames
				if current_phoneme_count > 0:
					current_frame_phonemes = [emitted_token]
					counts.append((current_phoneme, current_phoneme_count))
				else:
					current_frame_phonemes.append(emitted_token)

				current_phoneme = emitted_token
				current_phoneme_count = 0
			else:
				# Changing frame
				if len(current_frame_phonemes) > 1:
					# Split the frame between the phonemes in it
					splited_duration = 1 / len(current_frame_phonemes)
					current_phoneme_count += splited_duration

					for phoneme in current_frame_phonemes[:-1]:
						counts.append((phoneme, splited_duration))

					current_frame_phonemes = [current_phoneme]

				else:
					current_phoneme_count += 1

		# Add the last phoneme
		counts.append((current_phoneme, current_phoneme_count))

		# If the last phoneme lasts 0 frames -> alignment error
		if current_phoneme_count == 0:
			raise RuntimeError("Alignment error: The last phoneme has a duration of 0 frames.")

		# Compute the duration of each phoneme
		durations = []
		for phoneme, count in counts:
			duration = round(count * frame_duration, 5)
			durations.append((phoneme, duration))
		
		return durations



	def duration_to_intervals(self, durations: Durations, audio_duration: float, word_labels: Optional[List[str]] = None) -> Tuple[Intervals, Intervals]:
		# Calculate the phoneme intervals
		phoneme_intervals = []
		t = 0.0
		for phn, dur in durations:
			xmin = t
			t += dur
			xmax = t
			phoneme_intervals.append((xmin, xmax, phn))

		# Set end of EOS token to the end of the audio
		phoneme_intervals[-1] = (phoneme_intervals[-1][0], audio_duration, phoneme_intervals[-1][2])

		# Regroup phonemes by words
		word_segments = []
		segment = []
		for xmin, xmax, phn in phoneme_intervals:
			if phn in self.config["special_tokens"]:
				if segment:
					word_segments.append(segment)
					segment = []
			else:
				segment.append((xmin, xmax, phn))
		if segment:
			word_segments.append(segment)

		# Calculate the word intervals
		word_intervals = []
		for idx, seg in enumerate(word_segments):
			start = seg[0][0]
			end = seg[-1][1]
			if word_labels and idx < len(word_labels):
				label = word_labels[idx]
			else:
				# By default, concatenate phoneme labels to form the word label
				label = "".join(p for _, _, p in seg)
			word_intervals.append((start, end, label))

		return phoneme_intervals, word_intervals


	def alignment_to_textgrid(
		self,
		alignment: TranslatedAlignment,
		audio_duration: float,
		frame_duration: float,
		path: Path,
		word_labels: Optional[List[str]] = None
	) -> None:

		try:
			# Convert the alignment to intervals
			durations = self.alignment_to_durations(alignment, frame_duration)
			phoneme_intervals, word_intervals = self.duration_to_intervals(durations, audio_duration, word_labels)

			# Create the directory if it doesn't exist
			output_dir = Path(path).parent
			output_dir.mkdir(parents=True, exist_ok=True)

			# Write the TextGrid file
			with open(path, "w", encoding="utf-8") as f:
				f.write('File type = "ooTextFile"\n')
				f.write('Object class = "TextGrid"\n\n')
				f.write('xmin = 0\n')
				f.write(f'xmax = {audio_duration:.6f}\n')
				f.write('tiers? <exists>\n')
				f.write('size = 2\n')
				f.write('item []:\n')

				# Phoneme tier
				f.write('    item [1]:\n')
				f.write('        class = "IntervalTier"\n')
				f.write(f'        name = "phones"\n')
				f.write('        xmin = 0\n')
				f.write(f'        xmax = {audio_duration:.6f}\n')
				f.write(f'        intervals: size = {len(phoneme_intervals)}\n')
				for i, (xmin, xmax, phn) in enumerate(phoneme_intervals, 1):
					f.write(f'        intervals [{i}]:\n')
					f.write(f'            xmin = {xmin:.6f}\n')
					f.write(f'            xmax = {xmax:.6f}\n')
					f.write(f'            text = "{phn}"\n')

				# Word tier
				f.write('    item [2]:\n')
				f.write('        class = "IntervalTier"\n')
				f.write(f'        name = "words"\n')
				f.write('        xmin = 0\n')
				f.write(f'        xmax = {audio_duration:.6f}\n')
				f.write(f'        intervals: size = {len(word_intervals)}\n')
				for i, (xmin, xmax, label) in enumerate(word_intervals, 1):
					f.write(f'        intervals [{i}]:\n')
					f.write(f'            xmin = {xmin:.6f}\n')
					f.write(f'            xmax = {xmax:.6f}\n')
					f.write(f'            text = "{label}"\n')

		except Exception as e:
			self.logger.error(f"Error writing TextGrid file: {e}", extra={"hidden": True})
			raise RuntimeError(f"Error writing TextGrid file: {e}")