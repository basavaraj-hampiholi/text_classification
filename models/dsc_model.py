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

		self.conv1 = nn.Conv1d(in_channels=300,out_channels=128, kernel_size=2)
		self.conv2 = nn.Conv1d(in_channels=300,out_channels=128, kernel_size=3, padding=1)
		self.conv3 = nn.Conv1d(in_channels=300,out_channels=128, kernel_size=4, padding=1)

		self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
		self.relu = nn.ReLU()

		self.fc = nn.Linear(384, num_classes)


	def forward(self, input_ids):
		x = self.embedding(input_ids)
		x_permute = x.permute(0, 2, 1)

		conv1 = self.pool(self.relu(self.conv1(x_permute)))
		conv2 = self.pool(self.relu(self.conv2(x_permute)))
		conv3 = self.pool(self.relu(self.conv3(x_permute)))

		cat_features = torch.cat((conv1, conv2, conv3), dim=1)
		feats = torch.mean(cat_features, dim=2)

		logits = self.fc(feats)

		return logits







