import pickle
from typing import Dict


class Tokenizer:
	def __init__(self, vocab, special_tokens: Dict[str, int] = {}):
		self.vocab = vocab
		self.char2idx = {char: idx+len(special_tokens) for idx, char in enumerate(vocab)}
		self.idx2char = {idx+len(special_tokens): char for idx, char in enumerate(vocab)}

		for token, idx in special_tokens.items():
			self.char2idx[token] = idx
			self.idx2char[idx] = token

	def encode(self, text):
		return [self.char2idx[char] for char in text]

	def decode(self, sequence):
		return ''.join([self.idx2char[idx] for idx in sequence])


def load_tokenizer(path: str) -> Tokenizer:
	"""Load a tokenizer from a pickle file."""
	with open(path, "rb") as file:
		tokenizer = pickle.load(file)

	return tokenizer