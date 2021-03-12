import os
import torch
import torch.nn as nn
from argparse import ArgumentParser, Namespace
from datasets import load_dataset
from captha_recognition import CapthaRecognition
from utils import *


def parse_args():
	parser = ArgumentParser(description='Captha Recognition')
	parser.add_argument('--test-img', help='testing image path', default='./captha.jpg')
	parser.add_argument('--load-ckpt', help='load model checkpoint')
	parser.add_argument('--cuda', help='use cuda', action='store_true')
	return parser.parse_args()


if __name__ == '__main__':
	params = parse_args()
	img = cv2.imread(params.test_img, cv2.IMREAD_COLOR)
	crop_imgs = captha_segmentation(img)
	if crop_imgs is None:
		print('None')
	else:
		test_data = np.array(crop_imgs, dtype=np.uint8).reshape(-1, 20, 20)
		np.save('test_data.npy', test_data)
		test_loader = load_dataset('test_data.npy', None, 0, params, shuffled=False, single=True)
		os.remove('test_data.npy')
		cr = CapthaRecognition(params, trainable=False)
		cr.load_model(params.load_ckpt)
		pred_idx = cr.test(test_loader)
		pred_label = [idx_to_label(i) for i in pred_idx]
		print(''.join(pred_label))