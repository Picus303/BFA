import torch
import torchaudio
from torch import Tensor
from typing import Union, List, Tuple
from torchaudio.transforms import Resample, MelSpectrogram

from BFA.forced_aligner.tokenizer import Tokenizer, load_tokenizer
from BFA.utils import Failure



class TensorPreprocessor:
	def __init__(self, config: dict) -> None:
		self.audio_config: dict = config["audio"]
		self.text_config: dict = config["text"]

		self.mel_transform = MelSpectrogram(
			sample_rate = self.audio_config["sample_rate"],
			n_fft = self.audio_config["n_fft"],
			hop_length = self.audio_config["hop_length"],
			win_length = self.audio_config["win_length"],
			n_mels = self.audio_config["n_mels"],
			f_min = self.audio_config["f_min"],
			f_max = self.audio_config["f_max"],
		)

		self.sample_rate: int = self.audio_config["sample_rate"]

		self.tokenizer: Tokenizer = load_tokenizer(self.text_config["tokenizer_path"])


	def process_audio(self, audio_path: str) -> Union[Tuple[Tensor, Tensor], Failure]:
		try:
			# Load, resample and normalize the audio
			audio, sample_rate = torchaudio.load(audio_path)
			audio = audio.mean(dim=0, keepdim=True)	# Convert to mono

			if sample_rate != self.sample_rate:
				resampler = Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
				audio = resampler(audio)

			audio /= audio.abs().max()

			# Apply the mel spectrogram transform
			mel = self.mel_transform(audio).transpose(-1, -2).unsqueeze(1)		# Convert to (batch, channel, time, frequency)
			l_mel = torch.tensor(mel.shape[2], dtype=torch.int32).unsqueeze(0)	# Length of the mel spectrogram

			return mel, l_mel

		except Exception as e:
			return Failure(f"Error during audio processing: {e}")


	def process_text(self, phonemes: List[str]) -> Union[Tuple[Tensor, Tensor], Failure]:
		try:
			# Convert the phonemes to indices
			phoneme_ids = self.tokenizer.encode(phonemes)

			# Convert to tensor
			phoneme_tensor = torch.tensor(phoneme_ids, dtype=torch.int32).unsqueeze(0)
			l_phoneme = torch.tensor(len(phoneme_ids), dtype=torch.int32).unsqueeze(0)

			return phoneme_tensor, l_phoneme

		except Exception as e:
			return Failure(f"Error during text processing: {e}")