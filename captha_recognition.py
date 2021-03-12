import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from net import Net
from utils import *


class CapthaRecognition(object):
	def __init__(self, params, trainable):
		self.p = params
		self.trainable = trainable
		self._compile()

	def _compile(self):
		print('Captha Recognition')
		self.model = Net(in_channels=1, num_classes=39)
		if self.trainable:
			self.optim = Adam(self.model.parameters(),
							  lr=self.p.learning_rate,
							  betas=self.p.adam[:2],
							  eps=self.p.adam[2])
			self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, 
							 patience=self.p.nb_epochs/4, factor=0.5, verbose=True)
			if self.p.loss == 'cel':
				self.loss = nn.CrossEntropyLoss()
		self.use_cuda = torch.cuda.is_available() and self.p.cuda
		if self.use_cuda:
			self.model = self.model.cuda()
			if self.trainable:
				self.loss = self.loss.cuda()

	def _print_params(self):
		print('Training parameters: ')
		self.p.cuda = self.use_cuda
		param_dict = vars(self.p)
		pretty = lambda x: x.replace('_', ' ').capitalize()
		print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
		print()

	def save_model(self, epoch, stats, first=False):
		if first:
			ckpt_dir_name = f'{datetime.now():%H%M}'
			self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
			if not os.path.isdir(self.p.ckpt_save_path):
				os.mkdir(self.p.ckpt_save_path)
			if not os.path.isdir(self.ckpt_dir):
				os.mkdir(self.ckpt_dir)
		valid_loss = stats['valid_loss'][epoch]
		fname_net = '{}/epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
		print('Saving checkpoint to: {}\n'.format(fname_net))
		torch.save(self.model.state_dict(), fname_net)
		fname_dict = '{}/stats.json'.format(self.ckpt_dir)
		with open(fname_dict, 'w') as f:
			json.dump(stats, f, indent=2)

	def load_model(self, ckpt_fname):
		print('Loading checkpoint from: {}'.format(ckpt_fname))
		if self.use_cuda:
			self.model.load_state_dict(torch.load(ckpt_fname))
		else:
			self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))

	def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
		print('\rTesting model on validation set... ', end='')
		epoch_time = time_elapsed_since(epoch_start)[0]
		valid_loss, valid_acc, valid_time = self.eval(valid_loader)
		show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_acc)
		self.scheduler.step(valid_loss)
		stats['train_loss'].append(train_loss)
		stats['valid_loss'].append(valid_loss)
		stats['valid_acc'].append(valid_acc)
		self.save_model(epoch, stats, epoch == 0)
		if self.p.plot_stats:
			loss_str = f'{self.p.loss.upper()} loss'
			plot_per_epoch(self.ckpt_dir, 'Train loss', stats['train_loss'], loss_str)
			plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], loss_str)
			plot_per_epoch(self.ckpt_dir, 'Valid acc', stats['valid_acc'], 'Accuracy')

	def train(self, train_loader, valid_loader):
		self.model.train(True)
		self._print_params()
		num_batches = len(train_loader)
		stats = {'train_loss': [], 'valid_loss': [], 'valid_acc': []}
		train_start = datetime.now()
		for epoch in range(self.p.nb_epochs):
			print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))
			epoch_start = datetime.now()
			train_loss_meter = AvgMeter()
			loss_meter = AvgMeter()
			time_meter = AvgMeter()
			for batch_idx, (data, label) in enumerate(train_loader):
				batch_start = datetime.now()
				progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)
				if self.use_cuda:
					data = data.cuda()
					label = label.cuda()
				pred = self.model(data)
				loss = self.loss(pred, label)
				loss_meter.update(loss.item())
				self.optim.zero_grad()
				loss.backward()
				self.optim.step()
				time_meter.update(time_elapsed_since(batch_start)[1])
				if ((batch_idx + 1) % self.p.report_interval == 0 and batch_idx) or (batch_idx == len(train_loader) - 1):
					show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
					train_loss_meter.update(loss_meter.avg, loss_meter.count)
					loss_meter.reset()
					time_meter.reset()
			self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
			train_loss_meter.reset()
		train_time = time_elapsed_since(train_start)[0]
		print('Training done! Total elapsed time: {}\n'.format(train_time))

	def eval(self, valid_loader):
		self.model.train(False)
		valid_start = datetime.now()
		valid_loss_meter = AvgMeter()
		valid_acc_meter = AvgMeter()
		for batch_idx, (data, label) in enumerate(valid_loader):
			if self.use_cuda:
				data = data.cuda()
				label = label.cuda()
			pred = self.model(data)
			loss = self.loss(pred, label)
			valid_loss_meter.update(loss.item(), data.size(0))
			pred_label = pred.max(dim=1)[1]
			valid_acc_meter.update(((pred_label == label).sum() / data.size(0)).item(), data.size(0))
		valid_time = time_elapsed_since(valid_start)[0]
		return valid_loss_meter.avg, valid_acc_meter.avg, valid_time

	def test(self, test_loader):
		self.model.train(False)
		test_start = datetime.now()
		pred_list = []
		for batch_idx, data in enumerate(test_loader):
			if self.use_cuda:
				data = data.cuda()
			pred = self.model(data)
			pred_list.append(pred.max(dim=1)[1].item())
		test_time = time_elapsed_since(test_start)[0]
		print('Testing done! Total elapsed time: {}\n'.format(test_time))
		return pred_list