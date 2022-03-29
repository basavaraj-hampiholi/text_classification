import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
import numpy as np


class Sentence_CNN(nn.Module):
	def __init__(self, vocab_size, embed_size, num_classes):
		super(Sentence_CNN, self).__init__()

		self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_size, padding_idx=0, max_norm=5.0)

		self.conv1 = nn.Conv1d(in_channels=300,out_channels=128, kernel_size=3)#, groups=out_channels)
		self.conv2 = nn.Conv1d(in_channels=300,out_channels=128, kernel_size=4)#, groups=out_channels)
		self.conv3 = nn.Conv1d(in_channels=300,out_channels=128, kernel_size=5)#, groups=out_channels)

		self.relu = nn.ReLU()

		self.fc = nn.Linear(384, num_classes)


	def forward(self, input_ids):
		x = self.embedding(input_ids)
		x_permute = x.permute(0, 2, 1)

		conv1 = self.relu(self.conv1(x_permute))
		conv2 = self.relu(self.conv2(x_permute))
		conv3 = self.relu(self.conv3(x_permute))

		max1 = F.max_pool1d(conv1, kernel_size=conv1.size(2))
		max2 = F.max_pool1d(conv2, kernel_size=conv2.size(2))
		max3 = F.max_pool1d(conv3, kernel_size=conv3.size(2))

		cat_features = torch.cat((max1, max2, max3), dim=1)
		feats = torch.mean(cat_features, dim=2)

		logits = self.fc(feats)

		return logits
