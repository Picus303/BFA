import torch
import torchaudio
from torch import Tensor
from pathlib import Path
from typing import Tuple
from torchaudio.transforms import Resample, MelSpectrogram

from ..utils import SharedLogger



class AudioPreprocessor:
	def __init__(self, config: dict) -> None:
		self.config = config

		# Initialize transforms
		self.mel_transform = MelSpectrogram(
			sample_rate = self.config["sample_rate"],
			n_fft = self.config["n_fft"],
			hop_length = self.config["hop_size"],
			win_length = self.config["win_size"],
			n_mels = self.config["n_mels"],
			f_min = self.config["f_min"],
			f_max = self.config["f_max"],
		)

		self.sample_rate: int = self.config["sample_rate"]
		self.frame_duration: float = self.config["hop_size"] / self.config["sample_rate"]

		# Intialize logger
		self.logger = SharedLogger.get_instance()
		self.logger.info("Audio preprocessor initialized successfully.")


	def process_audio(self, audio_path: Path) -> Tuple[Tensor, Tensor, float]:
		try:
			# Load, resample and normalize the audio
			audio, sample_rate = torchaudio.load(audio_path)
			audio = audio.mean(dim=0, keepdim=True)	# Convert to mono

			if sample_rate != self.sample_rate:
				resampler = Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
				audio = resampler(audio)

			audio /= audio.abs().max()

			# Apply the mel spectrogram transform
			mel: Tensor = self.mel_transform(audio)
			mel = mel.transpose(-1, -2).unsqueeze(1)							# Convert to (batch, channel, time, frequency)
			l_mel = torch.tensor(mel.shape[2], dtype=torch.int32).unsqueeze(0)	# Length of the mel spectrogram

			# Calculate audio duration
			audio_duration: float = audio.shape[1] / self.sample_rate	# Duration in seconds

			return mel, l_mel, audio_duration

		except Exception as e:
			self.logger.error(f"Error during audio processing: {e}", extra={"hidden": True})
			raise RuntimeError(f"Error during audio processing: {e}")