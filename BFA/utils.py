import yaml


def load_cfg(cfg_path: str) -> dict:
	"""Load the configuration file."""
	with open(cfg_path, "r") as f:
		cfg = yaml.safe_load(f)

	return cfg