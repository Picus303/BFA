import yaml


class Failure(str):
	pass


def load_cfg(cfg_path: str) -> dict:
	"""Load the configuration file."""
	with open(cfg_path, "r") as f:
		cfg = yaml.safe_load(f)

	# Convert sets
	cfg["g2p_engine"]["modifiers"] = set(cfg["g2p_engine"]["modifiers"])
	cfg["g2p_engine"]["ponctuation"] = set(cfg["g2p_engine"]["ponctuation"])

	return cfg