import numpy as np
import pandas as pd
from functools import reduce
from scipy import sparse
from scipy.sparse import coo_matrix
import torch
from torch.utils.data import TensorDataset, Dataset


def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	# TODO: Read a csv file from path.
	# TODO: Please refer to the header of the file to locate X and y.
	# TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	# TODO: Remove the header of CSV file of course.
	# TODO: Do Not change the order of rows.
	# TODO: You can use Pandas if you want to.
	data_csv = pd.read_csv(path)
	if model_type == 'MLP':
		data = torch.tensor(data_csv.drop('y', axis = 1).values.astype(np.float32))
		target = torch.tensor((data_csv['y'] - 1).values)
		dataset = TensorDataset(data, target)
	elif model_type == 'CNN':
		data = data_csv.loc[:, 'X1':'X178'].values
		target = torch.tensor((data_csv['y'] - 1).values)
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(1), target)
	elif model_type == 'RNN':
		data = data_csv.loc[:, 'X1':'X178'].values
		target = torch.tensor((data_csv['y'] - 1).values)
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(2), target)
	else:
		raise AssertionError("Wrong Model Type!")
	return dataset

def calculate_num_features(seqs):
	features = reduce(lambda a, b: a + b, seqs)
	features = reduce(lambda a, b: a + b, features)
	return max(features) + 2

class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")
		self.labels = labels
		answers = []
		for x in seqs:
			a = len(x)
			b = num_features
			mtx = np.zeros((a, b))
			each_line = 0
			for ets in x:
				for et in ets:
					mtx[each_line, et] = 1
				each_line = each_line + 1
			answers.append(mtx)
			self.seqs = answers

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]

def visit_collate_fn(batch):
	x = 0
	lines = []
	for a, b in batch:
		lines.append((a.shape[0], x))
		x = x + 1
	lines.sort(key = lambda s: s[0], reverse=True)
	line_row = lines[0][0]
	line_col = batch[0][0].shape[1]
	listOne = []
	listTwo = []
	listThree = []
	for i in list(map(lambda s: s[1], lines)):
		patient = batch[i]
		listTwo.append(patient[1])
		listThree.append(patient[0].shape[0])
		d = np.zeros((line_row, line_col))
		d[0:patient[0].shape[0], 0:patient[0].shape[1]] = patient[0]
		listOne.append(d)
        
	seqs_tensor = torch.FloatTensor(listOne)
	lengths_tensor = torch.LongTensor(listThree)
	labels_tensor = torch.LongTensor(listTwo)
	#print(listTwo)

	return (seqs_tensor, lengths_tensor), labels_tensor