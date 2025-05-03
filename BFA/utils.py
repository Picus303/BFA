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
from typing import (
	Optional,
	Union,
	Dict,
	List,
	Tuple,
)	



class Failure(str):
	pass


FilePair = Dict[str, Path]
RawAlignment = List[Tuple[int, int, Optional[int]]]
TranslatedAlignment = List[Tuple[int, int, Optional[str]]]


def load_cfg(cfg_path: Path, root: Path) -> Union[dict, Failure]:
	"""Load the configuration file."""
	try:
		# Read the config file
		with cfg_path.open("r") as file:
			cfg = yaml.safe_load(file)

		# Convert lists to sets
		cfg["io_manager"]["supported_audio_formats"] = set(cfg["io_manager"]["supported_audio_formats"])
		cfg["io_manager"]["supported_annotation_formats"] = set(cfg["io_manager"]["supported_annotation_formats"])
		cfg["text_preprocessor"]["modifiers"] = set(cfg["text_preprocessor"]["modifiers"])
		cfg["text_preprocessor"]["punctuation"] = set(cfg["text_preprocessor"]["punctuation"])
		cfg["io_manager"]["special_tokens"] = set(cfg["io_manager"]["special_tokens"])

		# Convert paths to absolute paths
		cfg["inference_engine"]["weights_paths"]["encoder"] = root / cfg["inference_engine"]["weights_paths"]["encoder"]
		cfg["inference_engine"]["weights_paths"]["decoder"] = root / cfg["inference_engine"]["weights_paths"]["decoder"]
		cfg["inference_engine"]["weights_paths"]["joint_network"] = root / cfg["inference_engine"]["weights_paths"]["joint_network"]
		cfg["text_preprocessor"]["tokenizer_path"] = root / cfg["text_preprocessor"]["tokenizer_path"]
		cfg["text_preprocessor"]["ipa_mapping_path"] = root / cfg["text_preprocessor"]["ipa_mapping_path"]
		cfg["logger"]["log_file"] = root / cfg["logger"]["log_file"]

		return cfg

	except Exception as e:
		return Failure(f"Failed to load config file: {e}")



class SharedLogger:

	_instance: Optional[Logger] = None

	@classmethod
	def get_instance(cls, config: Optional[dict] = None, reset: bool = False) -> Logger:
		if not cls._instance is None:
			return cls._instance
		if config is None:
			raise ValueError("Logger not initialized. Please provide a config.")

		cls._instance = cls.get_logger(config, reset)
		return cls._instance

	@classmethod
	def get_logger(cls, config: dict, reset: bool = False) -> Logger:
		"""Initialize the logger."""

		# Create log directory if it doesn't exist
		log_dir = Path(config["log_file"]).parent
		log_dir.mkdir(parents=True, exist_ok=True)

		# Set up logging
		logger = logging.getLogger(config["name"])
		logger.setLevel(config["base_log_level"])

		# Create file handler
		write_mode = "w" if reset else "a"
		file_handler = FileHandler(config["log_file"], mode=write_mode)
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