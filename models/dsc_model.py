import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
import numpy as np


class Sentence_CNN(nn.Module):
	def __init__(self, vocab_size, embed_size, num_classes):
		super(Sentence_CNN, self).__init__()

		self.out_channels=64
		self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_size, padding_idx=0, max_norm=5.0)
		self.conv1 = nn.Conv1d(in_channels=embed_size, out_channels=self.out_channels, kernel_size=3, dilation=1, padding=1, groups=self.out_channels)
		self.conv2 = nn.Conv1d(in_channels=embed_size, out_channels=self.out_channels, kernel_size=3, dilation=2, padding=2, groups=self.out_channels)
		self.conv3 = nn.Conv1d(in_channels=embed_size, out_channels=self.out_channels, kernel_size=3, dilation=3, padding=3, groups=self.out_channels)
		self.conv4 = nn.Conv1d(in_channels=embed_size, out_channels=self.out_channels, kernel_size=3, dilation=4, padding=4, groups=self.out_channels)

		self.conv1x1=nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)
		self.relu = nn.ReLU()

		self.fc = nn.Linear(256, num_classes)

	def forward(self, input_ids):

		x = self.embedding(input_ids)
		x_permute = x.permute(0, 2, 1)

		conv1 = self.conv1(x_permute)
		conv2 = self.conv2(x_permute)
		conv3 = self.conv3(x_permute)
		conv4 = self.conv4(x_permute)

		cat_feats = torch.cat((conv1, conv2, conv3, conv4), dim=1)
		conv_cat = self.relu(self.conv1x1(cat_feats))

		max1 = F.max_pool1d(conv_cat, kernel_size=conv_cat.size(2))		
		feats = torch.mean(max1, dim=2)

		logits = self.fc(feats)

		return logits
