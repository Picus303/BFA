import torch
from torch import Tensor
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import EncoderOnlyTransformer, build_encoder_only_transformer
from .joint_network import JointNetwork



class RNNT:
	def __init__(self, config: dict):
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


	def forward(
		self,
		spectrogram: Tensor,
		text: Tensor,
		spectrogram_length: Tensor,
		text_length: Tensor
	) -> Tensor:

		# Build Attention Mask:
		# This should note be necessary, but I messed up the indexes for mask selection during training
		# Because of that, the EOS token must be masked -> Will be fixed if I retrain the model
		mask = torch.zeros((1, text_length+1, text_length+1), device=self.device)
		mask[:, :, :-1] = 1

		encoder_output = encoder(spectrogram, spectrogram_length.cpu())
		decoder_output = decoder(text, mask)
		joint_output = joint_network(encoder_output, decoder_output)

		joint_output = F.log_softmax(joint_output, dim=-1)
		return joint_output