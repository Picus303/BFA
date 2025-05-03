import torch
import torch.nn as nn

class JointNetwork(nn.Module):
	def __init__(self, dim: int, vocab_size: int):
		super(JointNetwork, self).__init__()

		self.projection = nn.Linear(dim, vocab_size)

	def forward(self, x: torch.Tensor, y: torch.Tensor):	# x: (B, T, 64), y: (B, U, 64)
		x = x.unsqueeze(2)		# x: (B, T, 1, 64)
		y = y.unsqueeze(1)		# y: (B, 1, U, 64)
		z = x * y				# z: (B, T, U, 64)

		z = self.projection(z)	# z: (B, T, U, 64)
		return z