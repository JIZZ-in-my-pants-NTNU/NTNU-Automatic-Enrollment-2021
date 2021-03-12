import os
import time
import requests
import numpy as np
from argparse import ArgumentParser
import torch
import torch.nn as nn
from datasets import load_dataset
from captha_recognition import CapthaRecognition
from utils import *


def parse_args():
	parser = ArgumentParser(description='Automatic Enrollment')
	parser.add_argument('--user-id', help='user id')
	parser.add_argument('--password', help='password')
	parser.add_argument('--serial-no', help='serial number')
	parser.add_argument('--domain-no', help='domain number', default='0000')
	parser.add_argument('--load-ckpt', help='load model checkpoint', default='captha.pt')
	parser.add_argument('--temp-filename', help='temporary filename', default='captha')
	parser.add_argument('--max-delay', help='maximum delay(s)', type=int, default=0)
	parser.add_argument('--cuda', help='use cuda', action='store_true')
	return parser.parse_args()

def login(server_rng):
	global my_cookies
	# First
	login_url = f'https://cos{server_rng}s.ntnu.edu.tw/AasEnrollStudent/LoginCheckCtrl?language=TW'
	response = requests.get(login_url)
	# print(response.content.decode())
	str_idx = response.content.decode().find('LoginCheckCtrl?action=login&id=')
	login_rng = response.content.decode()[str_idx:str_idx+100].split('\'')[2]
	my_cookies = response.cookies
	# Second
	captha_url = f'https://cos{server_rng}s.ntnu.edu.tw/AasEnrollStudent/RandImage'
	captha_header = {'Host': f'cos{server_rng}s.ntnu.edu.tw',
				 'Connection': 'keep-alive',
				 'sec-ch-ua': '''"Google Chrome";v="87", " Not;A Brand";v="99", "Chromium";v="87"''',
				 'sec-ch-ua-mobile': '?0',
				 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
				 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
				 'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
				 'Origin': f'https://cos{server_rng}s.ntnu.edu.tw',
				 'Sec-Fetch-Site': 'same-origin',
				 'Sec-Fetch-Mode': 'cors',
				 'Sec-Fetch-Dest': 'empty',
				 'Referer': f'https://cos{server_rng}.ntnu.edu.tw/AasEnrollStudent/LoginCheckCtrl?language=TW',
				 'Accept-Encoding': 'gzip, deflate, br',
				 'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,ja;q=0.5,es;q=0.4'}
	captha = get_captha_answer(server_rng, captha_url, captha_header)
	# Third
	login_url = f'https://cos{server_rng}s.ntnu.edu.tw/AasEnrollStudent/LoginCheckCtrl?action=login&id={login_rng}'
	my_header = {'Host': f'cos{server_rng}s.ntnu.edu.tw',
				 'Connection': 'keep-alive',
				 'sec-ch-ua': '''"Google Chrome";v="87", " Not;A Brand";v="99", "Chromium";v="87"''',
				 'X-Requested-With': 'XMLHttpRequest',
				 'sec-ch-ua-mobile': '?0',
				 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
				 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
				 'Accept': '*/*',
				 'Origin': f'https://cos{server_rng}s.ntnu.edu.tw',
				 'Sec-Fetch-Site': 'same-origin',
				 'Sec-Fetch-Mode': 'cors',
				 'Sec-Fetch-Dest': 'empty',
				 'Referer': f'https://cos{server_rng}.ntnu.edu.tw/AasEnrollStudent/LoginCheckCtrl?language=TW',
				 'Accept-Encoding': 'gzip, deflate, br',
				 'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,ja;q=0.5,es;q=0.4'}
	my_data = {'userid': params.user_id, 'password': params.password, 'validateCode': captha, 'CheckTW': 1}
	response = requests.post(login_url, headers=my_header, data=my_data, cookies=my_cookies)
	# print(response.content.decode())
	response_dict = str_to_dict(response.content.decode())
	if not response_dict['success']:
		print(response_dict['msg'])
		return False
	# Fourth
	login_url = f'https://cos{server_rng}s.ntnu.edu.tw/AasEnrollStudent/IndexCtrl?language=TW'
	my_header = {'Host': f'cos{server_rng}s.ntnu.edu.tw',
				 'Connection': 'keep-alive',
				 'sec-ch-ua': '''"Google Chrome";v="87", " Not;A Brand";v="99", "Chromium";v="87"''',
				 'sec-ch-ua-mobile': '?0',
				 'Upgrade-Insecure-Requests': '1',
				 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
				 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
				 'Sec-Fetch-Site': 'same-origin',
				 'Sec-Fetch-Mode': 'navigate',
				 'Sec-Fetch-User': '?1',
				 'Sec-Fetch-Dest': 'document',
				 'Referer': f'https://cos{server_rng}.ntnu.edu.tw/AasEnrollStudent/LoginCheckCtrl?language=TW',
				 'Accept-Encoding': 'gzip, deflate, br',
				 'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,ja;q=0.5,es;q=0.4'}
	response = requests.get(login_url, headers=my_header, cookies=my_cookies)
	# print(response.content.decode())
	str_idx = response.content.decode().find('stdName')
	my_name = response.content.decode()[str_idx:str_idx+500].split('\'')[8]
	# Fifth
	login_url = f'https://cos{server_rng}s.ntnu.edu.tw/AasEnrollStudent/LoginCtrl'
	my_header = {'Host': f'cos{server_rng}s.ntnu.edu.tw',
				 'Connection': 'keep-alive',
				 'sec-ch-ua': '''"Google Chrome";v="87", " Not;A Brand";v="99", "Chromium";v="87"''',
				 'X-Requested-With': 'XMLHttpRequest',
				 'sec-ch-ua-mobile': '?0',
				 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
				 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
				 'Accept': '*/*',
				 'Origin': f'https://cos{server_rng}s.ntnu.edu.tw',
				 'Sec-Fetch-Site': 'same-origin',
				 'Sec-Fetch-Mode': 'cors',
				 'Sec-Fetch-Dest': 'empty',
				 'Referer': f'https://cos{server_rng}.ntnu.edu.tw/AasEnrollStudent/IndexCtrl?language=TW',
				 'Accept-Encoding': 'gzip, deflate, br',
				 'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,ja;q=0.5,es;q=0.4'}
	my_data = {'userid': params.user_id, 'stdName': my_name, 'CheckTW': 1}
	response = requests.post(login_url, headers=my_header, data=my_data, cookies=my_cookies)
	# print(response.content.decode())
	response_dict = str_to_dict(response.content.decode())
	if not response_dict['success']:
		print(response_dict['msg'])
		return False
	# Sixth
	login_url = f'https://cos{server_rng}s.ntnu.edu.tw/AasEnrollStudent/EnrollCtrl?action=go'
	my_header = {'Host': f'cos{server_rng}s.ntnu.edu.tw',
				 'Connection': 'keep-alive',
				 'sec-ch-ua': '''"Google Chrome";v="87", " Not;A Brand";v="99", "Chromium";v="87"''',
				 'sec-ch-ua-mobile': '?0',
				 'Upgrade-Insecure-Requests': '1',
				 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
				 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
				 'Sec-Fetch-Site': 'same-origin',
				 'Sec-Fetch-Mode': 'navigate',
				 'Sec-Fetch-User': '?1',
				 'Sec-Fetch-Dest': 'document',
				 'Referer': f'https://cos{server_rng}.ntnu.edu.tw/AasEnrollStudent/IndexCtrl?language=TW',
				 'Accept-Encoding': 'gzip, deflate, br',
				 'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,ja;q=0.5,es;q=0.4'}
	response = requests.get(login_url, headers=my_header, cookies=my_cookies)
	# print(response.content.decode())
	# Seventh
	login_url = f'https://cos{server_rng}s.ntnu.edu.tw/AasEnrollStudent/StfseldListCtrl'
	my_header = {'Host': f'cos{server_rng}s.ntnu.edu.tw',
				 'Connection': 'keep-alive',
				 'sec-ch-ua': '''"Google Chrome";v="87", " Not;A Brand";v="99", "Chromium";v="87"''',
				 'sec-ch-ua-mobile': '?0',
				 'Upgrade-Insecure-Requests': '1',
				 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
				 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
				 'Sec-Fetch-Site': 'same-origin',
				 'Sec-Fetch-Mode': 'navigate',
				 'Sec-Fetch-Dest': 'iframe',
				 'Referer': f'https://cos{server_rng}.ntnu.edu.tw/AasEnrollStudent/EnrollCtrl?action=go',
				 'Accept-Encoding': 'gzip, deflate, br',
				 'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,ja;q=0.5,es;q=0.4'}
	response = requests.get(login_url, headers=my_header, cookies=my_cookies)
	# print(response.content.decode())
	# Eighth
	login_url = f'https://cos{server_rng}s.ntnu.edu.tw/AasEnrollStudent/CourseQueryCtrl?action=add'
	my_header = {'Host': f'cos{server_rng}s.ntnu.edu.tw',
				 'Connection': 'keep-alive',
				 'sec-ch-ua': '''"Google Chrome";v="87", " Not;A Brand";v="99", "Chromium";v="87"''',
				 'sec-ch-ua-mobile': '?0',
				 'Upgrade-Insecure-Requests': '1',
				 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
				 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
				 'Sec-Fetch-Site': 'same-origin',
				 'Sec-Fetch-Mode': 'navigate',
				 'Sec-Fetch-User': '?1',
				 'Sec-Fetch-Dest': 'iframe',
				 'Referer': f'https://cos{server_rng}.ntnu.edu.tw/AasEnrollStudent/StfseldListCtrl',
				 'Accept-Encoding': 'gzip, deflate, br',
				 'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,ja;q=0.5,es;q=0.4'}
	response = requests.get(login_url, headers=my_header, data=my_data, cookies=my_cookies)
	# print(response.content.decode())
	return True

def enroll(server_rng):
	enroll_url = f'https://cos{server_rng}s.ntnu.edu.tw/AasEnrollStudent/EnrollCtrl'
	my_header = {'Host': f'cos{server_rng}s.ntnu.edu.tw',
				 'Connection': 'keep-alive',
				 'sec-ch-ua': '''"Google Chrome";v="87", " Not;A Brand";v="99", "Chromium";v="87"''',
				 'X-Requested-With': 'XMLHttpRequest',
				 'sec-ch-ua-mobile': '?0',
				 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
				 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
				 'Accept': '*/*',
				 'Origin': f'https://cos{server_rng}s.ntnu.edu.tw',
				 'Sec-Fetch-Site': 'same-origin',
				 'Sec-Fetch-Mode': 'cors',
				 'Sec-Fetch-Dest': 'empty',
				 'Referer': f'https://cos{server_rng}.ntnu.edu.tw/AasEnrollStudent/CourseQueryCtrl?action=add',
				 'Accept-Encoding': 'gzip, deflate, br',
				 'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,ja;q=0.5,es;q=0.4'}
	# First
	my_data = {'action': 'checkCourseTime', 'serial_no': params.serial_no, 'direct': 1}
	response = requests.post(enroll_url, headers=my_header, data=my_data, cookies=my_cookies)
	# print(response.content.decode())
	response_dict = str_to_dict(response.content.decode())
	if not response_dict['success']:
		print(response_dict['msg'])
		return False
	# Second
	my_data = {'action': 'checkDomain', 'serial_no': params.serial_no, 'direct': 1}
	response = requests.post(enroll_url, headers=my_header, data=my_data, cookies=my_cookies)
	response_dict = str_to_dict(response.content.decode())
	# print(response.content.decode())
	if (not response_dict['success']) and (params.domain_no == '0000'):
		print(response_dict['courseCode'])
		return False
	# Third
	captha_url = f'https://cos{server_rng}s.ntnu.edu.tw/AasEnrollStudent/RandImage?type=q&{np.random.randint(low=1000000000000, high=10000000000000)}'
	captha_header = {'Host': f'cos{server_rng}s.ntnu.edu.tw',
				 'Connection': 'keep-alive',
				 'sec-ch-ua': '''"Google Chrome";v="87", " Not;A Brand";v="99", "Chromium";v="87"''',
				 'sec-ch-ua-mobile': '?0',
				 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
				 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
				 'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
				 'Origin': f'https://cos{server_rng}s.ntnu.edu.tw',
				 'Sec-Fetch-Site': 'same-origin',
				 'Sec-Fetch-Mode': 'no-cors',
				 'Sec-Fetch-Dest': 'image',
				 'Referer': f'https://cos{server_rng}.ntnu.edu.tw/AasEnrollStudent/CourseQueryCtrl?action=add',
				 'Accept-Encoding': 'gzip, deflate, br',
				 'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,ja;q=0.5,es;q=0.4'}
	captha = get_captha_answer(server_rng, captha_url, captha_header)
	# Human Delay
	time.sleep(np.random.uniform(low=0, high=params.max_delay))
	# Fourth
	my_data = {'action': 'add', 'serial_no': params.serial_no, 'direct': 1, 'validQuery': captha}
	if params.domain_no == '0000':
		my_data['guDomain'] = ''
	else:
		my_data['guDomain'] = params.domain_no
	response = requests.post(enroll_url, headers=my_header, data=my_data, cookies=my_cookies)
	# print(response.content.decode())
	response_dict = str_to_dict(response.content.decode())
	print(response_dict['msg'])
	if not response_dict['success']:
		return False
	return True

def logout(server_rng):
	logout_url = f'https://cos{server_rng}s.ntnu.edu.tw/AasEnrollStudent/LogoutCtrl'
	my_header = {'Host': f'cos{server_rng}s.ntnu.edu.tw',
				 'Connection': 'keep-alive',
				 'sec-ch-ua': '''"Google Chrome";v="87", " Not;A Brand";v="99", "Chromium";v="87"''',
				 'X-Requested-With': 'XMLHttpRequest',
				 'sec-ch-ua-mobile': '?0',
				 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
				 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
				 'Accept': '*/*',
				 'Origin': f'https://cos{server_rng}s.ntnu.edu.tw',
				 'Sec-Fetch-Site': 'same-origin',
				 'Sec-Fetch-Mode': 'cors',
				 'Sec-Fetch-Dest': 'empty',
				 'Referer': f'https://cos{server_rng}.ntnu.edu.tw/AasEnrollStudent/CourseQueryCtrl?action=add',
				 'Accept-Encoding': 'gzip, deflate, br',
				 'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,ja;q=0.5,es;q=0.4'}
	my_data =  {'action': 'logout'}
	response = requests.post(logout_url, headers=my_header, data=my_data, cookies=my_cookies)
	# print(response.content.decode())

def get_captha_answer(server_rng, captha_url, captha_header):
	while True:
		response = requests.get(captha_url, headers=captha_header, cookies=my_cookies)
		with open(f'{params.temp_filename}.jpg', 'wb') as f:
			f.write(response.content)
		pred_label = test_captha()
		if pred_label is None:
			continue
		os.remove(f'{params.temp_filename}.jpg')
		if pred_label[3] == '=':
			if '1' <= pred_label[0] <= '9' and '1' <= pred_label[2] <= '9':
				if pred_label[1] == '*' or pred_label[1] == 'x':
					return str(int(pred_label[0]) * int(pred_label[2]))
				elif pred_label[1] == '+':
					return str(int(pred_label[0]) + int(pred_label[2]))
				elif pred_label[1] == '-':
					return str(int(pred_label[0]) - int(pred_label[2]))
		else:
			if 'a' <= pred_label[0] <= 'z' and 'a' <= pred_label[1] <= 'z' and 'a' <= pred_label[2] <= 'z' and 'a' <= pred_label[3] <= 'z':
				return ''.join(pred_label)

def test_captha():
	img = cv2.imread(f'{params.temp_filename}.jpg', cv2.IMREAD_COLOR)
	crop_imgs = captha_segmentation(img)
	if crop_imgs is None:
		return None
	test_data = np.array(crop_imgs, dtype=np.uint8).reshape(-1, 20, 20)
	np.save(f'{params.temp_filename}.npy', test_data)
	test_loader = load_dataset(f'{params.temp_filename}.npy', None, 0, params, shuffled=False, single=True)
	os.remove(f'{params.temp_filename}.npy')
	pred_idx = cr.test(test_loader)
	pred_label = [idx_to_label(i) for i in pred_idx]
	return pred_label


if __name__ == '__main__':
	params = parse_args()
	cr = CapthaRecognition(params, trainable=False)
	cr.load_model(params.load_ckpt)
	while True:
		server_rng = np.random.randint(low=1, high=5)
		if not login(server_rng):
			continue
		login_start = datetime.now()
		while True:
			if enroll(server_rng):
				exit()
			login_time = time_elapsed_since(login_start)[1]
			if login_time >= 1000 * 1000:
				logout(server_rng)
				break