import os
import numpy as np
import logging as log
from PIL import Image
import tensorflow as tf

class ParseImageNet:
	# global variables
	_NUM_CLASSES = 1000
	_IMAGENET_MEAN = [123.68, 116.779, 103.939]
	_IMAGE_SIZE = [224, 224, 3]

	def __init__(self, batch_size):
		"""
		Constructor for parsing the ImageNet dataset.

		Inputs:
			- batch_size: batch size
		"""
		self.batch_size = batch_size

	def get_mean(self):
		"""
		Return the mean of training images.

		Returns:
			- list containing mean of training images
		"""
		return self._IMAGENET_MEAN

	def get_num_classes(self):
		"""
		Return number of classes of the dataset.

		Returns:
			- number of classes
		"""
		return self._NUM_CLASSES

	def get_image_size(self):
		"""
		Return the shape of the input image to be fed into the network.

		Returns:
			- shape in python list format
		"""
		return self._IMAGE_SIZE

	def get_next_train_batch(self):
		"""
		Get the next training batch of size batch_size.

		Returns:
			- batch: numpy array of size
					 [batch_size, self.crop_shape[0], self.crop_shape[1], 1]
			- label_onehot: label in one hot format [self.batch_size, self._NUM_CLASSES]
			- is_last: True if returned batch is the last batch
		"""
		is_last = False

		batch = np.zeros((self.batch_size, 224, 224, 3))
		label = np.zeros((self.batch_size, self._NUM_CLASSES))

		img = Image.open('laska.png').convert('RGB').resize((224,224))
		img = np.array(img, dtype=np.float32).reshape((1, 224, 224, 3))
		one_hot = [1 if i == 356 else 0 for i in range(1000)]  # 1-hot result for tiger

		for i in range(self.batch_size):
			batch[i, :, :, :] = img
			label[i, :] = np.asarray(one_hot).reshape(1,1000)

		return batch, label, is_last
