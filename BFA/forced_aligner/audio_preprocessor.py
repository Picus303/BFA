import torch
import torchaudio
from torch import Tensor
from pathlib import Path
from typing import Union, List, Tuple
from torchaudio.transforms import Resample, MelSpectrogram

from ..utils import Failure



class AudioPreprocessor:
	def __init__(self, config: dict) -> None:
		self.config = config

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


	def process_audio(self, audio_path: Path) -> Union[Tuple[Tensor, Tensor], Failure]:
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

			return mel, l_mel

		except Exception as e:
			return Failure(f"Error during audio processing: {e}")