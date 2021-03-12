import os
import cv2
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils import captha_segmentation
from sklearn.cluster import MiniBatchKMeans


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--data-size', '-d', help='size of dataset', default=40000, type=int)
	parser.add_argument('--k-means', '-k', help='number of clusters', default=200, type=int)
	parser.add_argument('--batch-size', '-b', help='size of minibatch', default=1000, type=int)
	parser.add_argument('--load-captha', '-l', help='captha images path', default='captha_imgs.npy')
	return parser.parse_args()


if __name__ == '__main__':
	params = parse_args()
	# download
	if not os.path.isfile(params.load_captha):
		captha_imgs = []
		while True:
			img_url = f'https://cos{np.random.randint(low=1, high=5)}s.ntnu.edu.tw/AasEnrollStudent/RandImage'
			response = requests.get(img_url)
			with open('tmp.jpg', 'wb') as f:
				f.write(response.content)
			img = cv2.imread('tmp.jpg', cv2.IMREAD_COLOR)
			os.remove('tmp.jpg')
			crop_imgs = captha_segmentation(img)
			if crop_imgs is None:
				continue
			captha_imgs.append(crop_imgs)
			print(len(captha_imgs)*4)
			if len(captha_imgs) >= params.data_size//4:
				break
		captha_imgs = np.array(captha_imgs, dtype=np.uint8).reshape(-1, 20, 20)
		np.save(params.load_captha, captha_imgs)
	else:
		captha_imgs = np.load(params.load_captha)
	# K-Means
	data = np.array(captha_imgs, dtype=np.float32).reshape(-1, 20*20) / 255.0
	model = MiniBatchKMeans(n_clusters=params.k_means, batch_size=params.batch_size)
	model.fit(data)
	print(f'Number of clusters: {model.n_clusters}')
	print(f'Inertia: {model.inertia_}')
	# labeling
	label_to_ch = np.zeros((params.k_means, ), dtype=np.str)
	for k in range(params.k_means):
		fig, ax = plt.subplots(1, 5)
		ax_idx = 0
		for idx, label in enumerate(model.labels_):
			if label == k:
				ax[ax_idx].imshow(captha_imgs[idx], cmap='gray')
				ax_idx += 1
			if ax_idx >= 5:
				break
		plt.show(block=False)
		print(f'Input the label of the {k+1}-cluster:', end=' ')
		while True:
			ch = input()
			if len(ch) != 1:
				print('Incorrect!')
			else:
				break
		label_to_ch[k] = ch
		plt.close('all')
	# save csv
	captha_idx = 0
	cols = ['label']
	for i in range(20):
		for j in range(20):
			cols.append(f'{i+1}x{j+1}')
	if not os.path.isfile('captha.csv'):
		captha_df = pd.DataFrame(columns=cols)
		captha_df.to_csv('captha.csv', header=True, index=False)
	else:
		captha_df = pd.read_csv('captha.csv')
		captha_idx = captha_df.shape[0]
	if not os.path.exists('captha'):
		os.mkdir('captha')
	new_captha = np.insert(captha_imgs.reshape(-1, 20*20).astype(np.str), 0, label_to_ch[model.labels_], axis=1)
	captha_df = pd.DataFrame(new_captha, columns=cols)
	captha_df.to_csv('captha.csv', mode='a', header=False, index=False)
	# save images
	for idx, label in enumerate(model.labels_):
		print(idx)
		if not os.path.exists(f'captha/{label_to_ch[label]}'):
			os.mkdir(f'captha/{label_to_ch[label]}')
		cv2.imwrite(f'captha/{label_to_ch[label]}/{captha_idx}.png', captha_imgs[idx])
		captha_idx += 1
	print('Done!')