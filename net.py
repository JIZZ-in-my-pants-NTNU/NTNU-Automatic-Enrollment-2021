import torch
import torch.nn as nn
from torchsummary import summary


class Net(nn.Module):
	def __init__(self, in_channels=1, num_classes=39):
		super(Net, self).__init__()
		self._block1 = nn.Sequential(
			nn.Conv2d(in_channels, 4, 3, stride=1, padding=1),
			nn.BatchNorm2d(4),
			nn.LeakyReLU(0.01),
			nn.MaxPool2d(2))
		self._block2 = nn.Sequential(
			nn.Conv2d(4, 16, 3, stride=1, padding=1),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(0.01),
			nn.MaxPool2d(2))
		self._block3 = nn.Sequential(
			nn.Linear(16*5*5, 64),
			nn.ReLU(),
			nn.BatchNorm1d(64),
			nn.Linear(64, num_classes))
		self._init_weights()

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight.data)
				m.bias.data.zero_()

	def forward(self, x):
		pool1 = self._block1(x)
		pool2 = self._block2(pool1)
		flatten = pool2.view(pool2.size(0), -1)
		return self._block3(flatten)


if __name__ == '__main__':
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = Net().to(device)
	summary(model, (1, 20, 20))