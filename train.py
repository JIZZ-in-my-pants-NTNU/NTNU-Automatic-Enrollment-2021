import torch
import torch.nn as nn
from argparse import ArgumentParser
from datasets import load_dataset
from captha_recognition import CapthaRecognition


def parse_args():
	parser = ArgumentParser(description='Captha Recognition')
	parser.add_argument('--train-data-npy', help='training data path', default='./train_data.npy')
	parser.add_argument('--train-label-npy', help='training label path', default='./train_label.npy')
	parser.add_argument('--valid-data-npy', help='validation data path', default='./valid_data.npy')
	parser.add_argument('--valid-label-npy', help='validation label path', default='./valid_label.npy')
	parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./ckpts')
	parser.add_argument('--report-interval', help='batch report interval', default=100, type=int)
	parser.add_argument('--train-size', help='size of train dataset', type=int)
	parser.add_argument('--valid-size', help='size of valid dataset', type=int)
	parser.add_argument('--learning-rate', help='learning rate', default=0.001, type=float)
	parser.add_argument('--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
	parser.add_argument('--batch-size', '-b', help='minibatch size', default=32, type=int)
	parser.add_argument('--nb-epochs', '-e', help='number of epochs', default=3, type=int)
	parser.add_argument('--loss', help='loss function', choices=['cel'], default='cel')
	parser.add_argument('--cuda', help='use cuda', action='store_true')
	parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')
	return parser.parse_args()


if __name__ == '__main__':
	params = parse_args()
	train_loader = load_dataset(params.train_data_npy, params.train_label_npy, params.train_size, params, shuffled=True)
	valid_loader = load_dataset(params.valid_data_npy, params.valid_label_npy, params.valid_size, params, shuffled=False)
	cr = CapthaRecognition(params, trainable=True)
	cr.train(train_loader, valid_loader)