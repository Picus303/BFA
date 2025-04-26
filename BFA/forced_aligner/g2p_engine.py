from misaki.en import G2P
from itertools import chain
from typing import Union, Literal, List, Set

from BFA.utils import Failure


class G2PEngine:
	def __init__(
		self,
		language: Literal["EN-GB", "EN-US"],
		vocab: Set[str],
		config: dict
	) -> None:

		self.model = G2P(british=(language == "EN-GB"), unk='❓')
		self.vocab = vocab
		self.special_tokens = config["special_tokens"]
		self.modifiers = config["modifiers"]
		self.ponctuation = config["punctuation"]

	def translate(self, text: str) -> Union[List[str]]:
		try:
			# Get the convertion tokens
			_, tokens = self.model(text)

			output = []

			for i, token in enumerate(tokens):
				# Filter out of vocabulary phonemes in tokens
				phonemes = [p if p in self.vocab else self.special_tokens["unknown"]
							for p in token.phonemes
							if not (p in self.modifiers or p in self.ponctuation)]

				# Ignore fully unknown tokens (especially targets ponctuation)
				if set(phonemes) == {self.special_tokens["unknown"]}:
					continue

				# Add the phoneme to the output
				output.append(phonemes)

				# Add a silences between tokens
				if i != len(tokens) - 1:
					output.append([self.special_tokens["silence"]])

			# Add the start and end of sequence tokens
			output = [self.special_tokens["start_of_sequence"]] + output + [self.special_tokens["end_of_sequence"]]
			output = list(chain.from_iterable(output))

			return output
		
		except Exception as e:
			return Failure(f"Error during G2P conversion: {e}")