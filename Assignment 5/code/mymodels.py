import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		#self.input_layer = nn.Linear(178, 16) #unimproved model
		#self.output_layer = nn.Linear(16, 5) #unimproved model
		self.input_layer = nn.Linear(178, 50) #improved model
		self.output_layer = nn.Linear(50, 5) #improved model
		self.improvement1 = nn.Dropout(p = 0.5) #improved model
		self.improvement2 = nn.BatchNorm1d(178) #improved model

	def forward(self, x):
		#x = torch.sigmoid(self.input_layer(x)) #unimproved model #improved model
		#x = self.output_layer(x) #unimproved model #improved model
		x = F.relu(self.improvement1(self.input_layer(self.improvement2(x)))) #improved model
		x = self.output_layer(x) #improved model
		return x

class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		#self.layer1 = nn.Conv1d(in_channels = 1, out_channels= 6, kernel_size = 5) #unimproved model
		#self.layer2 = nn.Conv1d(6, 16, 5) #unimproved model
		#self.pool = nn.MaxPool1d(kernel_size = 2) #unimproved model
		#self.input_layer = nn.Linear(in_features=16 * 41, out_features=128) #unimproved model
		#self.output_layer = nn.Linear(128, 5) #unimproved model
		self.layer1 = nn.Conv1d(in_channels = 1, out_channels= 6, kernel_size = 5) #improved model
		self.layer2 = nn.Conv1d(6, 16, 5) #improved model
		self.pool = nn.MaxPool1d(kernel_size = 2) #improved model
		self.input_layer = nn.Linear(in_features=16 * 41, out_features=128) #improved model
		self.output_layer = nn.Linear(128, 5) #improved model
		self.improvement1 = nn.Dropout(p = 0.2) #improved model

	def forward(self, x):
		#x = self.pool(F.relu(self.layer1(x))) #unimproved model
		#x = self.pool(F.relu(self.layer2(x))) #unimproved model
		#x = x.view(-1, 16 * 41) #unimproved model
		#x = F.relu(self.input_layer(x)) #unimproved model
		#x = self.output_layer(x) #unimproved model
		x = self.pool(F.relu(self.improvement1(self.layer1(x)))) #improved model
		x = self.pool(F.relu(self.improvement1(self.layer2(x)))) #improved model
		x = x.view(-1, 16 * 41) #improved model
		x = F.relu(self.improvement1(self.input_layer(x))) #improved model
		x = self.output_layer(x) #improved model
		return x

class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		#self.rnn_model = nn.GRU(input_size= 1, hidden_size= 16, num_layers = 1, batch_first = True) #unimproved model
		#self.input_layer = nn.Linear(in_features = 16, out_features = 5) #unimproved model
		self.rnn_model = nn.GRU(input_size= 1, hidden_size = 16, num_layers = 1, batch_first = True, dropout = 0.3) #improved model
		self.input_layer = nn.Linear(in_features = 16, out_features = 5) #improved model

	def forward(self, x):
		#x, _ = self.rnn_model(x) #unimproved model
		#x = self.input_layer(x[:, -1, :]) #unimproved model
		x, _ = self.rnn_model(x) #improved model
		x = F.relu(x[:, -1, :]) #improved model
		x = self.input_layer(x) #improved model
		return x

#class MyVariableRNN(nn.Module):
	#def __init__(self, dim_input):
		#super(MyVariableRNN, self).__init__()
		#self.input_layer1 = nn.Linear(in_features=dim_input, out_features=32) #unimproved model
		#self.rnn_model = nn.GRU(input_size=32, hidden_size=16, num_layers=1, batch_first=True) #unimproved model
		#self.input_layer2 = nn.Linear(in_features=16, out_features=2) #unimproved model

	#def forward(self, input_tuple):
		#seqs, lengths = input_tuple
		#seqs = torch.tanh(self.input_layer1(seqs)) #unimproved model
		#seqs = pack_padded_sequence(seqs, lengths, batch_first=True) #unimproved model
		#seqs, h = self.rnn_model(seqs) #unimproved model
		#seqs, _ = pad_packed_sequence(seqs, batch_first=True) #unimproved model
		#seqs = self.input_layer2(seqs[:, -1, :]) #unimproved model
		#return seqs

class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()

		self.batch_first =True
		self.layer1 = nn.Sequential(nn.Dropout(p=0.8),nn.Linear(dim_input, 128, bias=False),nn.Dropout(p=0.5))
		self.rnn1 = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
		self.rnnl1 = nn.Linear(in_features=128, out_features=1)
		self.rnnl1.bias.data.zero_()
		self.rnn2 = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
		self.rnnl2 = nn.Linear(in_features=128, out_features=128)
		self.rnnl2.bias.data.zero_()
		self.rnno = nn.Sequential(nn.Dropout(p=0.5),nn.Linear(in_features=128, out_features=2))
		self.rnno[1].bias.data.zero_()

	def forward(self, input_tuple):
		seqs, lengths = input_tuple
		b1, m1 = seqs.size()[:2]
		x = self.layer1(seqs)
		pi = pack_padded_sequence(x, lengths, batch_first=self.batch_first)
		a, _ = self.rnn1(pi)
		b, _ = pad_packed_sequence(a, batch_first=self.batch_first)
		c = Variable(torch.FloatTensor([[1.0 if i < lengths[idx] else 0.0 for i in range(m1)] for idx in range(b1)]).unsqueeze(2), requires_grad=False)
		e = self.rnnl1(b)
		def max(x, c):
				exp = torch.exp(x)
				msp = exp * c
				sth = torch.sum(msp, dim=1, keepdim=True)
				return msp / sth
		alpha = max(e, c)
		h, _ = self.rnn2(pi)
		gps, _ = pad_packed_sequence(h, batch_first=self.batch_first)
		out = torch.tanh(self.rnnl2(gps))
		context = torch.bmm(torch.transpose(alpha, 1, 2), out * x).squeeze(1)
		rnno = self.rnno(context)
		return rnno