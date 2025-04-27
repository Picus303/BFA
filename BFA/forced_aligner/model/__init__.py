import torch
from torch import Tensor
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import EncoderOnlyTransformer, build_encoder_only_transformer
from .joint_network import JointNetwork



class RNNT:
	def __init__(self, config: dict) -> None:
		self.config: dict = config
		self.device = torch.device("cpu")	# Small model: faster on CPU -> no transfers

		build_args = config["build_args"]
		weights_paths = config["weights_paths"]

		# Build Modules
		self.encoder: Encoder = Encoder(build_args["encoder"])
		self.decoder: EncoderOnlyTransformer = build_encoder_only_transformer(build_args["decoder"])
		self.joint_network: JointNetwork = JointNetwork(build_args["joint_network"])

		# Load Model Weights
		self.encoder.load_state_dict(torch.load(weights_paths["encoder"], weights_only=True))
		self.decoder.load_state_dict(torch.load(weights_paths["decoder"], weights_only=True))
		self.joint_network.load_state_dict(torch.load(weights_paths["joint_network"], weights_only=True))

		# Use Eval Mode
		self.encoder.eval()
		self.decoder.eval()
		self.joint_network.eval()


	def inference(
		self,
		spectrogram: Tensor,
		text: Tensor,
		spectrogram_length: Tensor,
		text_length: Tensor
	) -> Tensor:

		with torch.no_grad():
			# Build Attention Mask:
			# This should note be necessary, but I messed up the indices for mask selection during training
			# Because of that, the EOS token must be masked -> Will be fixed if I retrain the model
			mask: Tensor = torch.zeros((1, text_length, text_length), device=self.device)
			mask[:, :, :-1] = 1

			# Pass Forward
			encoder_output = encoder(spectrogram, spectrogram_length.cpu())
			decoder_output = decoder(text, mask)
			joint_output = joint_network(encoder_output, decoder_output)

			return joint_output