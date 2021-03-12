import numpy as np
import torch
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
	def __init__(self, data_npy, label_npy, redux=0):
		super(MyDataset, self).__init__()
		self.data_list = np.load(data_npy)
		if label_npy is None:
			self.label_list = None
		else:
			self.label_list = torch.from_numpy(np.load(label_npy))
		if redux:
			self.data_list = self.data_list[:redux]
			self.label_list = self.label_list[:redux]

	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, idx):
		data = tvF.to_tensor(self.data_list[idx]).float()
		if self.label_list is None:
			return data
		label = self.label_list[idx]
		return data, label


def load_dataset(data_npy, label_npy, redux, params, shuffled=False, single=False):
	dataset = MyDataset(data_npy, label_npy, redux)
	if single:
		return DataLoader(dataset, batch_size=1, shuffle=shuffled)
	else:
		return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)