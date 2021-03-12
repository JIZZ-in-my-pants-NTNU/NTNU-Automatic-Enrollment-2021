import numpy as np
import pandas as pd
from utils import label_to_idx
from argparse import ArgumentParser


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--train-ratio', '-r', help='ratio of training data', default=0.8, type=float)
	parser.add_argument('--load-dataset', '-l', help='captha dataset path', default='captha.csv')
	return parser.parse_args()


if __name__ == '__main__':
	params = parse_args()
	# label to idx
	captha_df = pd.read_csv(params.load_dataset)
	captha_df['label'] = captha_df['label'].apply(label_to_idx)
	# train-valid split
	train_num = int(params.train_ratio * captha_df.shape[0])
	train_df = captha_df[:train_num]
	train_data = train_df.drop('label', axis=1).to_numpy(dtype=np.uint8).reshape((-1, 20, 20))
	train_label = train_df['label'].to_numpy(dtype=np.int64).reshape((-1, ))
	np.save('train_data.npy', train_data)
	np.save('train_label.npy', train_label)
	valid_df = captha_df[train_num:]
	valid_data = valid_df.drop('label', axis=1).to_numpy(dtype=np.uint8).reshape((-1, 20, 20))
	valid_label = valid_df['label'].to_numpy(dtype=np.int64).reshape((-1, ))
	np.save('valid_data.npy', valid_data)
	np.save('valid_label.npy', valid_label)