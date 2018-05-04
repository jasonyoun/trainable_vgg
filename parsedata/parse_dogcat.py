import os
import numpy as np
import tensorflow as tf
import logging as log
from utils import *
from image_utils import *
from PIL import Image

class ParseDogCat:
	# global variables
	_TRAIN_LIST = 'train_list.txt'
	_DOGCAT_FOLDER = 'kaggle_dogcat'
	_TRAIN_FOLDER = 'train'
	_TEST_FOLDER = 'test'
	_CAT_LABEL = 0
	_DOG_LABEL = 1
	_NUM_CLASSES = 2
	_TRAIN_DATA_PER_CLASS = 12500
	_TOTAL_TRAIN_DATA = 25000
	_file_list = None
	_label_list = None
	_DOGCAT_MEAN = [124.27288165, 115.83684239, 106.07509781]
	_batch_idx = None

	def __init__(self, dataset_dir, crop_shape, batch_size, resize=256, initial_load=False):
		"""
		Constructor for parsing the Kaggle Dog vs. Cat dataset.

		Inputs:
			- dataset_dir: path to the dataset parent directory
			- crop_shape: shape of batch to be randomly cropped to e.g. [224, 224]
			- batch_size: batch size
			- resize: isotropically resize the smallest side of images to this size before cropping
			- initial_load: Set to True if loading the dataset for the first time
		"""
		# generate and check paths
		self.dataset_path = os.path.join(dataset_dir, self._DOGCAT_FOLDER)
		self.train_path = os.path.join(self.dataset_path, self._TRAIN_FOLDER)
		self.test_path = os.path.join(self.dataset_path, self._TEST_FOLDER)
		assert dir_exists(self.train_path)
		assert dir_exists(self.test_path)
		self.train_list_path = os.path.join(self.dataset_path, self._TRAIN_LIST)

		# init numpy arrays
		self._file_list = np.empty(self._TOTAL_TRAIN_DATA, dtype=object)
		self._label_list = np.empty(self._TOTAL_TRAIN_DATA, dtype=np.int)

		# other inits
		self.crop_shape = crop_shape
		self.batch_size = batch_size
		self.resize = resize
		self._batch_idx = 0

		# load the list or make one if loading for the first time
		if initial_load is True:
			self._make_list()
		else:
			self._load_list()

		# get_image_mean(self.train_list_path, resize=self.resize)

	def _make_list(self):
		"""
		After reading the file names from the training dataset,
		shuffle the data and save file list and label list
		to a single .txt file.
		"""
		# variables
		count = 0

		log.info('Saving list for the Kaggle Dog Cat dataset ' +
			'to \'{}\'.'.format(self.train_list_path))

		# check if train_list.txt file exists
		if file_exists(self.train_list_path):
			log.warning('File \'{}\' will be overwritten!'.format(self.train_list_path))

		# cat
		for i in range(self._TRAIN_DATA_PER_CLASS):
			self._file_list[count] = os.path.join(self.train_path, 'cat.{:d}.jpg'.format(i))
			assert file_exists(self._file_list[count])
			self._label_list[count] = self._CAT_LABEL
			count += 1

		# dog
		for i in range(self._TRAIN_DATA_PER_CLASS):
			self._file_list[count] = os.path.join(self.train_path, 'dog.{:d}.jpg'.format(i))
			assert file_exists(self._file_list[count])
			self._label_list[count] = self._DOG_LABEL
			count += 1

		# shuffle lists
		shuffle_idx = np.arange(self._TOTAL_TRAIN_DATA)
		np.random.shuffle(shuffle_idx)
		self._file_list = self._file_list[shuffle_idx]
		self._label_list = self._label_list[shuffle_idx]

		# write to file
		with open(self.train_list_path, 'w') as text_file:
			for i in range(self._TOTAL_TRAIN_DATA):
				print('{:05d} {} {}'.format(i+1, self._file_list[i], self._label_list[i]), file=text_file)

		log.info('Finished saving list for the Kaggle Dog Cat dataset.')

	def _load_list(self):
		"""
		Load the shuffled file and label list from the .txt file.
		"""
		log.info('Loading list for the Kaggle Dog Cat dataset ' +
			'from \'{}\'.'.format(self.train_list_path))

		# check if train_list.txt file exists
		if not file_exists(self.train_list_path):
			log.error('File \'{}\' does not exist!'.format(self.train_list_path))

		# open from file
		with open(self.train_list_path, 'r') as text_file:
			for i, line in enumerate(text_file):
				self._file_list[i] = line.split()[1]
				self._label_list[i] = line.split()[2]

		log.info('Finished loading list for the Kaggle Dog Cat dataset.')

	def get_mean(self):
		"""
		Get the mean of training images.

		Returns:
			- list containing mean of training images
		"""
		return self._DOGCAT_MEAN

	def get_num_classes(self):
		"""
		Get number of classes of the dataset.

		Returns:
			- number of classes
		"""
		return self._NUM_CLASSES

	def get_next_train_batch(self):
		"""
		Get the next training batch of size self.batch_size.

		Returns:
			- batch: numpy array of size
					 [self.batch_size, self.crop_shape[0], self.crop_shape[1], 3]
			- label_onehot: label in one hot format [self.batch_size, self._NUM_CLASSES]
			- is_last: True if returned batch is the last batch
		"""
		# variables
		height, width = self.crop_shape
		start = self._batch_idx
		end = self._batch_idx + self.batch_size
		is_last = False

		# dynamically allocate batch size depending
		# on the location of current batch index
		if end <= self._TOTAL_TRAIN_DATA:
			batch = np.zeros((self.batch_size, height, width, 3), dtype=np.float32)
			label = np.zeros(self.batch_size, dtype=np.int)
		else:
			batch = np.zeros((self._TOTAL_TRAIN_DATA-start, height, width, 3), dtype=np.float32)
			label = np.zeros(self._TOTAL_TRAIN_DATA-start, dtype=np.int)
			end = self._TOTAL_TRAIN_DATA
			is_last = True

		# load images from the self._file_list[] and assign to batch
		for i in range(start, end):
			# images
			image = Image.open(self._file_list[i]).convert('RGB')
			image = random_crop(image, height, width, self.resize)
			batch[i-start,:,:,:] = np.array(image, dtype=np.float32)

			# labels
			label[i-start] = self._label_list[i]

		# update the global static variable for batch index
		if is_last is True:
			self._batch_idx = 0
		else:
			self._batch_idx += self.batch_size

		# convert label to one-hot label format
		label_onehot = np.zeros((label.shape[0], self.get_num_classes()))
		label_onehot[np.arange(label.shape[0]), label] = 1

		return batch, label_onehot, is_last
