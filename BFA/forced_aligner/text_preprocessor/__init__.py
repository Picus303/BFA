import re
import json
import torch
from torch import Tensor
from pathlib import Path
from misaki.en import G2P
from itertools import chain
from typing import (
	Literal,
	List,
	Tuple,
	Dict,
	Set,
)

from .tokenizer import Tokenizer
from ...utils import (
	RawAlignment,
	TranslatedAlignment,
	SharedLogger,
)



class TextPreprocessor:
	def __init__(
		self,
		language: Literal["EN-GB", "EN-US"],
		config: dict
	) -> None:

		# Initialize text to phoneme model (Misaki format)
		self.g2p = G2P(british=(language == "EN-GB"), unk='❓')
		self.special_tokens = config["special_tokens"]
		self.modifiers = config["modifiers"]
		self.punctuation = config["punctuation"]

		# Load the tokenizer
		self.tokenizer = Tokenizer.from_json(config["tokenizer_path"])
		self.vocab: Set[str] = set(self.tokenizer.vocab)

		# Load IPA mapping
		with open(config["ipa_mapping_path"], "r") as file:
			ipa_mapping: Dict[str, Dict[str, str]] = json.load(file)

		self.ipa_to_misaki_mapping: Dict[str, str] = ipa_mapping["ipa_to_misaki"]
		self.misaki_to_ipa_mapping: Dict[str, str] = ipa_mapping["misaki_to_ipa"]

		self.sorted_ipa_keys: List[str] = sorted(self.ipa_to_misaki_mapping, key=len, reverse=True)
		self.stress_pattern = re.compile(r"[ˈˌ]")

		# Initialize logger
		self.logger = SharedLogger.get_instance()
		self.logger.info("Text preprocessor initialized successfully.")

	#==================================================================
	# Special tokens
	#==================================================================

	def add_special_tokens(self, phonemes: List[List[str]]) -> List[str]:
		"""Add special tokens to the phonemes list."""
		# Add silence tokens between each word
		for i in range(len(phonemes) - 1):
			phonemes.insert(i * 2 + 1, [self.special_tokens["silence"]])
		
		# Add start and end of sequence tokens
		phonemes = [[self.special_tokens["start_of_sequence"]]] + phonemes + [[self.special_tokens["end_of_sequence"]]]
		phonemes_chain = list(chain.from_iterable(phonemes))

		return phonemes_chain

	#==================================================================
	# G2P conversion
	#==================================================================

	def translate(self, text: str) -> List[str]:
		"""Convert raw text to phonemes (Misaki format)."""
		try:
			# Get the convertion tokens
			_, tokens = self.g2p(text)

			output = []

			for i, token in enumerate(tokens):
				# Filter out of vocabulary phonemes in tokens
				if token.phonemes is None:
					continue

				phonemes = [p if p in self.vocab else self.special_tokens["unknown"]
							for p in token.phonemes
							if not (p in self.modifiers or p in self.punctuation)]

				# Ignore fully unknown or empty tokens
				if not phonemes or (set(phonemes) == {self.special_tokens["unknown"]}):
					continue

				# Add the phoneme to the output
				output.append(phonemes)

			# Add special tokens
			return self.add_special_tokens(output)
		
		except Exception as e:
			self.logger.error(f"Error during G2P conversion: {e}", extra={"hidden": True})
			raise RuntimeError(f"Error during G2P conversion: {e}")

	#==================================================================
	# Tokenization
	#==================================================================

	def tokenize(self, phonemes: List[str]) -> Tuple[Tensor, Tensor]:
		"""Convert phonemes to indices for the model embeddings."""
		try:
			# Convert the phonemes to indices
			phoneme_ids = self.tokenizer.encode(phonemes)

			# Convert to tensor
			phoneme_tensor = torch.tensor(phoneme_ids, dtype=torch.int32).unsqueeze(0)
			l_phoneme = torch.tensor(len(phoneme_ids), dtype=torch.int32).unsqueeze(0)

			return phoneme_tensor, l_phoneme

		except Exception as e:
			self.logger.error(f"Error during tokenization: {e}", extra={"hidden": True})
			raise RuntimeError(f"Error during tokenization: {e}")


	def detokenize(self, phoneme: int, ptype: Literal["IPA", "Misaki"]) -> str:
		"""Convert phoneme index to string."""
		misaki_phoneme = self.tokenizer.decode([phoneme])
		if ptype == "Misaki":
			return misaki_phoneme
		elif ptype == "IPA":
			return self.misaki_to_ipa(misaki_phoneme)
		else:
			raise ValueError(f"Unknown ptype: {ptype}")


	def detokenize_alignment(self, raw_alignment: RawAlignment, ptype: Literal["IPA", "Misaki"]) -> TranslatedAlignment:
		"""Convert the alignment to phonemes."""
		try:
			translated_alignment: TranslatedAlignment = []
			for t, u, emit in raw_alignment:
				if emit is not None:
					translated_phoneme = self.detokenize(emit, ptype)
					translated_alignment.append((t, u, translated_phoneme))
				else:
					translated_alignment.append((t, u, None))

			return translated_alignment

		except Exception as e:
			self.logger.error(f"Error during detokenization: {e}", extra={"hidden": True})
			raise RuntimeError(f"Error during detokenization: {e}")

	#==================================================================
	# IPA to Misaki conversion
	#==================================================================

	def segment_ipa_word(self, word: str) -> List[str]:
		"""
		Split a single word (string of IPA symbols) into symbol list.

		A *longest-match* strategy ensures diphthongs & affriqués (e.g.
		`eɪ`, `aʊ`, `tʃ`) are captured as single tokens.
		"""
		tokens: List[str] = []
		i = 0
		L = len(word)
		while i < L:
			matched = False

			# Search for the longest match first
			for key in self.sorted_ipa_keys:
				if word.startswith(key, i):
					tokens.append(key)
					i += len(key)
					matched = True
					break

			# If no match is found, take the single character
			if not matched:
				tokens.append(word[i])
				i += 1

		return tokens


	def segment_ipa_text(self, text: str) -> List[List[str]]:
		"""Segment a string of IPA symbols into a list of words."""
		words = self.stress_pattern.sub("", text).split()
		return [self.segment_ipa_word(word) for word in words]


	def ipa_to_misaki(self, text: str) -> List[str]:
		"""
		Convert a string of IPA symbols to Misaki symbols.
		Only used before alignment.
		"""
		try:
			segmented = self.segment_ipa_text(text)
			ipa_text = []

			ukn_token = self.special_tokens["unknown"]

			for word in segmented:
				ipa_word = []
				for phoneme in word:
					# Add converted phoneme or unknown token
					ipa_word.append(self.ipa_to_misaki_mapping.get(phoneme, ukn_token))

				ipa_text.append(ipa_word)

			# Add special tokens
			return self.add_special_tokens(ipa_text)

		except Exception as e:
			self.logger.error(f"Error during IPA to Misaki conversion: {e}", extra={"hidden": True})
			raise RuntimeError(f"Error during IPA to Misaki conversion: {e}")
	

	def misaki_to_ipa(self, phoneme: str) -> str:
		"""
		Convert a single Misaki symbol to IPA.
		Only used after alignment.
		"""
		return self.misaki_to_ipa_mapping.get(phoneme, phoneme)

	#==================================================================
	# Main functions
	#==================================================================

	def process_misaki(self, text: str) -> List[str]:
		"""Process the text file and return the phonemes in Misaki format."""
		try:
			# Split the text into words
			words = [[c if c in self.vocab else self.special_tokens["unknown"]
						for c in word if c not in self.modifiers and c not in self.punctuation]
						for word in text.split()]

			# Ignore fully unknown or empty words
			words = [word for word in words if word and (set(word) != {self.special_tokens["unknown"]})]
			phonemes = self.add_special_tokens(words)

			return phonemes

		except Exception as e:
			self.logger.error(f"Error during text processing (Misaki): {e}", extra={"hidden": True})
			raise RuntimeError(f"Error during text processing (Misaki): {e}")


	def process_text(
		self,
		text_path: Path,
		dtype: Literal["words", "phonemes"],
		ptype: Literal["IPA", "Misaki"]
	) -> Tuple[Tensor, Tensor]:
		"""Process the text file and return the phonemes."""

		try:
			# Read the text file
			with open(text_path, "r") as file:
				text = file.read().strip()

			if not text:
				raise ValueError("Empty text file")

			# Select the processing method
			match dtype:
				case "words":
					# Ignore dtype (output format only) and use G2P
					phonemes = self.translate(text)

				case "phonemes":
					match ptype:
						case "IPA":
							# Convert IPA to Misaki
							phonemes = self.ipa_to_misaki(text)
						case "Misaki":
							# Extract phonemes from the text
							phonemes = self.process_misaki(text)
						case _:
							raise ValueError(f"Unknown ptype: {ptype}")

				case _:
					raise ValueError(f"Unknown dtype: {dtype}")

			# Tokenize the phonemes
			return self.tokenize(phonemes)

		except Exception as e:
			self.logger.error(f"Error during text processing: {e}", extra={"hidden": True})
			raise RuntimeError(f"Error during text processing: {e}")


	def get_word_labels(self, text_path: Path) -> List[str]:
		"""Get the word labels from the text file."""

		# Read the text file
		with open(text_path, "r") as file:
			text = file.read().strip()

		if not text:
			raise ValueError("Empty text file")

		# Remove ponctuation
		for p in self.punctuation:
			text = text.replace(p, "")

		# Split the text into words
		words = [word for word in text.split() if word]

		return words