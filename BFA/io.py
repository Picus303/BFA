from pathlib import Path
from typing import Optional, List, Tuple

from BFA.utils import Failure, Durations, Intervals



class TextGridWriter:
	def __init__(self, config: dict):
		self.config = config


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


	def durations_to_textgrid(
		self,
		durations: Durations,
		audio_duration: float,
		path: str,
		word_labels: Optional[List[str]] = None
	) -> Optional[Failure]:

		try:
			phoneme_intervals, word_intervals = self.duration_to_intervals(durations, audio_duration, word_labels)

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

			return None	# Success

		except Exception as e:
			return Failure(f"Error writing TextGrid file: {e}")