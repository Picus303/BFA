import os
import yaml
import logging
from logging import (
	Logger,
	LogRecord,
	StreamHandler,
	FileHandler,
	Formatter,
	Filter,
)

from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple, Any	

PATHS = (
	("model", "weights_paths", "encoder"),
	("model", "weights_paths", "decoder"),
	("model", "weights_paths", "joint_network"),
	("tensor_preprocessor", "text", "tokenizer_path"),
	("logger", "log_file"),
)



Failure = str
FilePair = Dict[str, Path]
RawAlignment = List[Tuple[int, int, Optional[int]]]
TranslatedAlignment = List[Tuple[int, int, Optional[str]]]


def read_dict(data: dict, keys: Tuple[str, ...]) -> Union[Any, Failure]:
	"""Access a value of a nested dictionary using a tuple of keys."""
	for key in keys:
		data = data[key]
	return data



def write_dict(data: dict, keys: Tuple[str, ...], value: Any) -> None:
	"""Write a value to a nested dictionary using a tuple of keys."""
	for key in keys[:-1]:
		data = data[key]
	data[keys[-1]] = value



def load_cfg(cfg_path: Path, root: Path) -> Union[dict, Failure]:
	"""Load the configuration file."""
	try:
		# Read the config file
		cfg_path = root / cfg_path
		with cfg_path.open("r") as file:
			cfg = yaml.safe_load(file)

		# Convert lists to sets
		cfg["supported_audio_formats"] = set(cfg["supported_audio_formats"])
		cfg["supported_annotation_formats"] = set(cfg["supported_annotation_formats"])
		cfg["g2p_engine"]["modifiers"] = set(cfg["g2p_engine"]["modifiers"])
		cfg["g2p_engine"]["ponctuation"] = set(cfg["g2p_engine"]["ponctuation"])
		cfg["textgrid_writer"]["special_tokens"] = set(cfg["textgrid_writer"]["special_tokens"])

		# Convert paths to absolute paths
		for path in PATHS:
			value = read_dict(cfg, path)
			if isinstance(value, str):
				abs_path = (root / value).resolve()
				write_dict(cfg, path, str(abs_path))
			else:
				# Error in config file or while accessing the value
				raise RuntimeError("Invalid path in config file")

		return cfg

	except Exception as e:
		return Failure(f"Failed to load config file: {e}")



def get_logger(config: dict) -> Logger:
	"""Initialize the logger."""

	# Create log directory if it doesn't exist
	log_dir = Path(config["log_file"]).parent
	log_dir.mkdir(parents=True, exist_ok=True)

	# Set up logging
	logger = logging.getLogger(config["name"])
	logger.setLevel(config["base_log_level"])

	# Create file handler
	file_handler = FileHandler(config["log_file"])
	file_handler.setLevel(config["file_log_level"])

	# Create console handler
	console_handler = StreamHandler()
	console_handler.setLevel(config["console_log_level"])

	# Create formatter
	formatter = Formatter(config["log_format"])

	file_handler.setFormatter(formatter)
	console_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(console_handler)

	# Create a filter to hide certain log messages
	class VerboseFilter(Filter):
		def filter(self, record: LogRecord) -> bool:
			return not getattr(record, "hidden", False)

	console_handler.addFilter(VerboseFilter())

	logger.info("Logger initialized")
	return logger