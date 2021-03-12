import os
import cv2
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import captha_segmentation


if __name__ == '__main__':
	captha_idx = 0
	if not os.path.isfile('captha.csv'):
		cols = ['label']
		for i in range(20):
			for j in range(20):
				cols.append(f'{i+1}x{j+1}')
		captha_df = pd.DataFrame(columns=cols)
		captha_df.to_csv('captha.csv', mode='w', header=True, index=False)
	else:
		captha_df = pd.read_csv('captha.csv')
		captha_idx = captha_df.shape[0]
	if not os.path.exists('captha'):
		os.mkdir('captha')
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
		plt.imshow(img)
		plt.show(block=False)
		print(captha_idx // 4)
		while True:
			input_str = input()
			if len(input_str) != 4:
				print('Incorrect!')
			else:
				break
		if input_str == 'pass':
			print('Pass!')
			continue
		if input_str == 'quit':
			print('Quit!')
			break
		for idx, crop_img in enumerate(crop_imgs):
			new_dict = {'label': input_str[idx]}
			for i in range(20):
				for j in range(20):
					new_dict[f'{i+1}x{j+1}'] = crop_img[i, j]
			new_captha = pd.DataFrame(new_dict, index=[0])
			new_captha.to_csv('captha.csv', mode='a', header=False, index=False)
			if not os.path.exists(f'captha/{input_str[idx]}'):
				os.mkdir(f'captha/{input_str[idx]}')
			cv2.imwrite(f'captha/{input_str[idx]}/{captha_idx}.png', crop_img)
			captha_idx += 1
		plt.close('all')