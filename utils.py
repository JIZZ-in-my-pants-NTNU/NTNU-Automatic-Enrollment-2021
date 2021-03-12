import os
import cv2
import json
import numpy as np
from math import log10
from datetime import datetime
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def label_to_idx(ch):
	if '1' <= ch <= '9':
		return ord(ch) - ord('1')
	elif ch == '*':
		return 9
	elif ch == '+':
		return 10
	elif ch == '-':
		return 11
	elif ch == '=':
		return 12
	else:
		return ord(ch) - ord('a') + 13

def idx_to_label(idx):
	if 0 <= idx <= 8:
		return chr(idx + ord('1'))
	elif idx == 9:
		return '*'
	elif idx == 10:
		return '+'
	elif idx == 11:
		return '-'
	elif idx == 12:
		return '='
	else:
		return chr(idx - 13 + ord('a'))

def captha_segmentation(img):
	height, width = img.shape[:2]
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
	edge_img = cv2.Canny(blur_img, 20, 160)
	contours, _ = cv2.findContours(edge_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) != 4:
		return None
	coords = []
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		center_x, center_y = x + w//2, y + h //2
		top_left_x, top_left_y = center_x - 10, center_y - 10
		bot_right_x, bot_right_y = center_x + 10, center_y + 10
		if top_left_x < 0:
			bot_right_x -= top_left_x
			top_left_x = 0
		if top_left_y < 0:
			bot_right_y -= top_left_y
			top_left_y = 0
		if bot_right_x >= width:
			top_left_x -= (bot_right_x - width + 1)
			bot_right_x = width - 1
		if bot_right_y >= height:
			top_left_y -= (bot_right_y - height + 1)
			bot_right_y = height - 1
		coords.append(((top_left_x, top_left_y), (bot_right_x, bot_right_y)))
	coords.sort(key = lambda s: s[0][0])
	crop_imgs = []
	for coord in coords:
		crop_imgs.append(gray_img[coord[0][1]:coord[1][1], coord[0][0]:coord[1][0]])
	return crop_imgs

def str_to_dict(s):
	try:
		d = json.loads(s)
	except json.decoder.JSONDecodeError:
		s = s.replace('\"', '')
		d = {}
		for pair in s[1:-1].split(','):
			key, value = pair.split(':')
			if value == 'true':
				d[key] = True
			elif value == 'false':
				d[key] = False
			else:
				d[key] = value
	finally:
		return d

def clear_line():
	print('\r{}'.format(' ' * 80), end='\r')

def progress_bar(batch_idx, num_batches, report_interval, train_loss):
	dec = int(np.ceil(np.log10(num_batches)))
	bar_size = 21 + dec
	progress = (batch_idx % report_interval) / report_interval
	fill = int(progress * bar_size) + 1
	print('\rBatch {:>{dec}d} [{}{}] Train loss: {:>1.5f}'.format(batch_idx + 1, '=' * fill + '>', ' ' * (bar_size - fill), train_loss, dec=str(dec)), end='')

def time_elapsed_since(start):
	timedelta = datetime.now() - start
	string = str(timedelta)[:-7]
	ms = int(timedelta.total_seconds() * 1000)
	return string, ms

def show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_acc):
	clear_line()
	print('Train time: {} | Valid time: {} | Valid loss: {:>1.5f} | Valid accuracy: {:>1.5f}'.format(epoch_time, valid_time, valid_loss, valid_acc))

def show_on_report(batch_idx, num_batches, loss, elapsed):
	clear_line()
	dec = int(np.ceil(np.log10(num_batches)))
	print('Batch {:>{dec}d} / {:d} | Avg loss: {:>1.5f} | Avg train time / batch: {:d} ms'.format(batch_idx + 1, num_batches, loss, int(elapsed), dec=dec))

def plot_per_epoch(ckpt_dir, title, measurements, y_label):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(range(1, len(measurements) + 1), measurements)
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.set_xlabel('Epoch')
	ax.set_ylabel(y_label)
	ax.set_title(title)
	plt.tight_layout()
	fname = '{}.png'.format(title.replace(' ', '-').lower())
	plot_fname = os.path.join(ckpt_dir, fname)
	plt.savefig(plot_fname, dpi=200)
	plt.close()


class AvgMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0.
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count