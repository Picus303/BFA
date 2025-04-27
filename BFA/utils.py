import os
import yaml
import logging
from logging import (
	Logger,
	StreamHandler,
	FileHandler,
	Formatter,
)

from pathlib import Path
from typing import Optional, Union, List, Tuple, Any	

PATHS = (
	("model", "weights_paths", "encoder"),
	("model", "weights_paths", "decoder"),
	("model", "weights_paths", "joint_network"),
	("tensor_preprocessor", "text", "tokenizer_path"),
	("logger", "log_file"),
)



Failure = str
Alignment = List[Tuple[int, int, Optional[int]]]
Durations = List[Tuple[str, float]]
Intervals = List[Tuple[float, float, str]]



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



def load_cfg(cfg_path: str, root: Path) -> Union[dict, Failure]:
	"""Load the configuration file."""
	try:
		# Read the config file
		cfg_path = os.join(root, cfg_path)
		with open(cfg_path, "r") as file:
			cfg = yaml.safe_load(file)

		# Convert lists to sets
		cfg["g2p_engine"]["modifiers"] = set(cfg["g2p_engine"]["modifiers"])
		cfg["g2p_engine"]["ponctuation"] = set(cfg["g2p_engine"]["ponctuation"])

		# Convert paths to absolute paths
		for path in PATHS:
			value = read_dict(cfg, path)
			if isinstance(value, str):
				abs_path = os.path.abspath(os.path.join(root, value))
				write_dict(cfg, path, abs_path)
			else:
				# Error in config file or while accessing the value
				raise RuntimeError("Invalid path in config file")

		return cfg

	except Exception as e:
		return Failure(f"Failed to load config file: {e}")



def get_logger(config: dict) -> Logger:
	"""Initialize the logger."""

	# Create log directory if it doesn't exist
	log_dir = os.path.dirname(config["log_file"])
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

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

	logger.info("Logger initialized")
	return logger