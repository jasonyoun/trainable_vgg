import os
import tensorflow as tf
import logging as log
import numpy as np

def dir_exists(dir_name):
	"""
	Check if directory exists.

	Inputs:
		- dir_name: directory name to check

	Returns:
		- True: directory exists
		- False: directory does not exist
	"""
	if os.path.exists(dir_name):
		# log.debug('Directory {} exists'.format(dir_name))
		return True
	else:
		log.debug('Directory {} does not exist'.format(dir_name))
		return False

def file_exists(file_name):
	"""
	Check if file exists.

	Inputs:
		- file_name: file name to check

	Returns:
		- True: file exists
		- False: file does not exist
	"""
	if os.path.isfile(file_name):
		# log.debug('File {} exists'.format(file_name))
		return True
	else:
		log.debug('File {} does not exist'.format(file_name))
		return False

def print_prob(prob, file_path):
	"""
	Print probability for ImageNet dataset.

	Inputs:
		- prob: probability
		- file_path: path to the sysnet.txt file

	Returns:
		- top1: top1 probability
	"""
	synset = [l.strip() for l in open(file_path).readlines()]

	# print prob
	pred = np.argsort(prob)[::-1]

	# Get top1 label
	top1 = synset[pred[0]]
	log.info('Top1: {} {}'.format(top1, prob[pred[0]]))
	# Get top5 label
	top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
	log.info('Top5: {}'.format(top5))

	return top1
