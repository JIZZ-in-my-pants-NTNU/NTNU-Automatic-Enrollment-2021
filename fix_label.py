import os
import numpy as np
import pandas as pd


if __name__ == '__main__':
	captha_df = pd.read_csv('captha.csv')
	while True:
		print('Input #.png wrong_label correct_label:', end=' ')
		input_arr = input().split()
		if len(input_arr) != 3:
			if input_arr[0] == 'quit':
				captha_df.to_csv('captha.csv', header=True, index=False)
				break
			continue
		num = int(input_arr[0])
		wrong_label = input_arr[1]
		correct_label = input_arr[2]
		os.rename(f'captha/{wrong_label}/{num}.png', f'captha/{correct_label}/{num}.png')
		captha_df.loc[num, 'label'] = correct_label
	print('Done!')