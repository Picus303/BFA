# bfa/tokenizer.py
import json
from pathlib import Path
from typing import Dict, Set, List, Union



class Tokenizer:
	def __init__(
		self,
		vocab: List[str],
		special_tokens: List[str],
		char2idx: Dict[str, int],
		idx2char: Dict[str, str],
	) -> None:

		self.vocab: Set[str] = set(vocab)
		self.special_tokens: Set[str] = set(special_tokens)
		self.char2idx: Dict[str, int] = char2idx
		self.idx2char: Dict[str, str] = idx2char

	def encode(self, text: List[str]) -> List[int]:
		return [self.char2idx[char] for char in text]

	def decode(self, sequence: List[int]) -> str:
		return "".join(self.idx2char[str(idx)] for idx in sequence)

	@classmethod
	def from_json(cls, path: Path) -> "Tokenizer":
		with path.open("r", encoding="utf-8") as file:
			data = json.load(file)

		return cls(
			vocab = data["vocab"],
			special_tokens = data["special_tokens"],
			char2idx = data["char2idx"],
			idx2char = data["idx2char"],
		)