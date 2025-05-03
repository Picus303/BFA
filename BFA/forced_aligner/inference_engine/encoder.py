import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import (
	pack_padded_sequence,
	pad_packed_sequence,
)


class Encoder(nn.Module):
	def __init__(self,
		mel_dim: int = 40,
		num_layers: int = 2,
		hidden_size: int = 256,
		output_size: int = 64,
	):
		super(Encoder, self).__init__()
		
		self.conv1 = nn.Conv2d(
			1, 16, 
			kernel_size = (5, 5),
			stride = (1, 1),
			padding = (2, 2),
		)

		self.conv2 = nn.Conv2d(
			16, 32,
			kernel_size = (5, 5),
			stride = (1, 1),
			padding = (2, 2),
		)

		self.norm1 = nn.BatchNorm2d(16)
		self.norm2 = nn.BatchNorm2d(32)

		self.pool1 = nn.MaxPool2d(
			kernel_size = (1, 2),
			stride = (1, 2),
		)

		self.pool2 = nn.MaxPool2d(
			kernel_size = (1, 4),
			stride = (1, 4),
		)

		self.GELU = nn.GELU()
		self.dropout = nn.Dropout(0.1)

		self.lstm = nn.LSTM(
			input_size = 4*mel_dim,
			hidden_size = hidden_size,
			num_layers = num_layers,
			batch_first = True,
			bidirectional = True,
		)

		self.h0 = nn.Parameter(torch.zeros(2*num_layers, 1, hidden_size))
		self.c0 = nn.Parameter(torch.zeros(2*num_layers, 1, hidden_size))

		self.projection = nn.Linear(2*hidden_size, output_size)

	def forward(self, x: Tensor, l: Tensor):	# x: (B, 1, T, F)
		batch_size = x.size(0)

		# Part 1: CNN

		x = self.conv1(x)		# x: (B, 16, T, F)
		x = self.norm1(x)
		x = self.GELU(x)
		x = self.pool1(x)		# x: (B, 16, T, F/2)
		x = self.dropout(x)

		x = self.conv2(x)		# x: (B, 32, T, F/2)
		x = self.norm2(x)
		x = self.GELU(x)
		x = self.pool2(x)		# x: (B, 32, T, F/8)
		x = self.dropout(x)

		# Part 2: LSTM

		x = x.permute(0, 2, 1, 3)							# x: (B, T, 32, F/8)
		x = x.contiguous().view(x.size(0), x.size(1), -1)	# x: (B, T, 4*F)

		sorted_lens, sorted_idx = l.sort(descending=True)
		x = x[sorted_idx]
		x_seq = pack_padded_sequence(x, sorted_lens, batch_first=True, enforce_sorted=False)

		h0 = self.h0.expand(-1, batch_size, -1).contiguous()
		c0 = self.c0.expand(-1, batch_size, -1).contiguous()

		x_seq, _ = self.lstm(x_seq, (h0, c0))	# x: (B, T, 512)
		x, _ = pad_packed_sequence(x_seq, batch_first=True)
		_, original_idx = sorted_idx.sort()
		x = x[original_idx]

		x = self.projection(x)	# x: (B, T, 64)
		return x