import os
import numpy as np
import tensorflow as tf
import logging as log
from utils import *
from image_utils import *
from PIL import Image

class ParseMnist:
	# global variables
	_NUM_CLASSES = 10
	_NUM_TRAIN_IMGS = 10000
	_MNIST_MEAN = [0, 0, 0]
	_batch_idx = None

	def __init__(self, batch_size):
		mnist_cluttered = np.load('D:\\Jason_Folder\\Study\\Coding\\stn_examples\\sample_data\\mnist_sequence1_sample_5distortions5x5.npz')

		self.X_train = mnist_cluttered['X_train']
		self.y_train = mnist_cluttered['y_train']

		# other inits
		self.batch_size = batch_size
		self._batch_idx = 0

	def get_mean(self):
		"""
		Get the mean of training images.

		Returns:
			- list containing mean of training images
		"""
		return self._MNIST_MEAN

	def get_num_classes(self):
		"""
		Get number of classes of the dataset.

		Returns:
			- number of classes
		"""
		return self._NUM_CLASSES

	def get_next_train_batch(self, augment=True, shuffle_after_each_epoch=True):
		# variables
		start = self._batch_idx
		end = self._batch_idx + self.batch_size
		is_last = False

		# dynamically allocate batch size depending
		# on the location of current batch index
		if end < self._NUM_TRAIN_IMGS:
			batch = np.zeros((self.batch_size, 40, 40, 1), dtype=np.float32)
			label = np.zeros(self.batch_size, dtype=np.int)
		else:
			batch = np.zeros((self._NUM_TRAIN_IMGS-start, 40, 40, 1), dtype=np.float32)
			label = np.zeros(self._NUM_TRAIN_IMGS-start, dtype=np.int)
			end = self._NUM_TRAIN_IMGS
			is_last = True

		# load images from the self._train_file_list[] and assign to batch
		for i in range(start, end):
			image_reshaped = np.expand_dims(np.reshape(self.X_train[i], (-1, 40, 40)), 3)

			batch[i-start,:,:,:] = np.array(image_reshaped, dtype=np.float32)

			# labels
			label[i-start] = self.y_train[i]

		# update the global static variable for batch index
		if is_last is True:
			self._batch_idx = 0
		else:
			self._batch_idx += self.batch_size

		# convert label to one-hot label format
		label_onehot = np.zeros((label.shape[0], self.get_num_classes()))
		label_onehot[np.arange(label.shape[0]), label] = 1

		return batch, label_onehot, is_last
