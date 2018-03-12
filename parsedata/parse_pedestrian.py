import os
import numpy as np
import tensorflow as tf
import logging as log
from utils import *
from parsedata.image_utils import *
from PIL import Image

class ParsePedestrian:
	# global variables
	_TRAIN_LIST = 'train_list.txt'
	_TEST_LIST = 'test_list.txt'
	_PEDESTRIAN_FOLDER = 'pedestrian_xing'
	_TRAIN_FOLDER = 'train'
	_TEST_FOLDER = 'test'
	_NEG_FOLDER = 'neg'
	_POS_FOLDER = 'pos'
	_NEG_LABEL = 0
	_POS_LABEL = 1
	_NUM_CLASSES = 2
	_NUM_POS_TRAIN_DATA = 1237
	_NUM_NEG_TRAIN_DATA = 5000
	_TOTAL_TRAIN_DATA = 6237
	_NUM_POS_TEST_DATA = 1178
	_NUM_NEG_TEST_DATA = 4504
	_TOTAL_TEST_DATA = 5682
	_train_file_list = None
	_train_label_list = None
	_test_file_list = None
	_test_label_list = None
	_PEDESTRIAN_MEAN = [114.86996197, 114.63007618, 108.72589651]
	_batch_idx = None

	def __init__(self, dataset_dir, resize, crop_shape, batch_size, isotropical=False, initial_load=False):
		# generate and check paths
		self.dataset_path = os.path.join(dataset_dir, self._PEDESTRIAN_FOLDER)
		self.train_path = os.path.join(self.dataset_path, self._TRAIN_FOLDER)
		self.test_path = os.path.join(self.dataset_path, self._TEST_FOLDER)
		assert dir_exists(self.train_path)
		assert dir_exists(self.test_path)
		self.train_neg_path = os.path.join(self.train_path, self._NEG_FOLDER)
		self.train_pos_path = os.path.join(self.train_path, self._POS_FOLDER)
		self.test_neg_path = os.path.join(self.test_path, self._NEG_FOLDER)
		self.test_pos_path = os.path.join(self.test_path, self._POS_FOLDER)
		self.train_list_path = os.path.join(self.dataset_path, self._TRAIN_LIST)
		self.test_list_path = os.path.join(self.dataset_path, self._TEST_LIST)

		# init numpy arrays
		self._train_file_list = np.empty(self._TOTAL_TRAIN_DATA, dtype=object)
		self._train_label_list = np.empty(self._TOTAL_TRAIN_DATA, dtype=np.int)
		self._test_file_list = np.empty(self._TOTAL_TEST_DATA, dtype=object)
		self._test_label_list = np.empty(self._TOTAL_TEST_DATA, dtype=np.int)

		# other inits
		self.resize = resize
		self.crop_shape = crop_shape
		self.batch_size = batch_size
		self.isotropical = isotropical
		self._batch_idx = 0

		# load the list or make one if loading for the first time
		if initial_load is True:
			self._make_list()
		else:
			self._load_list()

		# get_image_mean(self.train_list_path, resize=self.resize, isotropical=self.isotropical)

	def _make_list(self):
		"""
		After reading the file names from the training dataset,
		shuffle the data and save file list and label list
		to a single .txt file.
		"""
		# variables
		count = 0

		log.info('Saving train list for the pedestrain dataset to \'{}\'.'.format(self.train_list_path))
		log.info('Saving test list for the pedestrain dataset to \'{}\'.'.format(self.test_list_path))

		# check if train_list.txt and test_list.txt file exists
		if file_exists(self.train_list_path):
			log.warning('File \'{}\' will be overwritten!'.format(self.train_list_path))

		if file_exists(self.test_list_path):
			log.warning('File \'{}\' will be overwritten!'.format(self.test_list_path))

		# train negative
		for i in range(self._NUM_NEG_TRAIN_DATA):
			self._train_file_list[count] = os.path.join(self.train_neg_path, '{:d}.png'.format(i+1))
			assert file_exists(self._train_file_list[count])
			self._train_label_list[count] = self._NEG_LABEL
			count += 1

		# train positive
		for i in range(self._NUM_POS_TRAIN_DATA):
			self._train_file_list[count] = os.path.join(self.train_pos_path, 'W{:d}.png'.format(i+1))
			assert file_exists(self._train_file_list[count])
			self._train_label_list[count] = self._POS_LABEL
			count += 1

		count = 0

		# test negative
		for file in os.listdir(self.test_neg_path):
			self._test_file_list[count] = os.path.join(self.test_neg_path, file)
			assert file_exists(self._test_file_list[count])
			self._test_label_list[count] = self._NEG_LABEL
			count += 1

		for file in os.listdir(self.test_pos_path):
			self._test_file_list[count] = os.path.join(self.test_pos_path, file)
			assert file_exists(self._test_file_list[count])
			self._test_label_list[count] = self._POS_LABEL
			count += 1

		# shuffle lists
		shuffle_idx = np.arange(self._TOTAL_TRAIN_DATA)
		np.random.shuffle(shuffle_idx)
		self._train_file_list = self._train_file_list[shuffle_idx]
		self._train_label_list = self._train_label_list[shuffle_idx]

		shuffle_idx = np.arange(self._TOTAL_TEST_DATA)
		np.random.shuffle(shuffle_idx)
		self._test_file_list = self._test_file_list[shuffle_idx]
		self._test_label_list = self._test_label_list[shuffle_idx]

		# write to file
		with open(self.train_list_path, 'w') as text_file:
			for i in range(self._TOTAL_TRAIN_DATA):
				print('{:05d} {} {}'.format(i+1, self._train_file_list[i], self._train_label_list[i]), file=text_file)

		with open(self.test_list_path, 'w') as text_file:
			for i in range(self._TOTAL_TEST_DATA):
				print('{:05d} {} {}'.format(i+1, self._test_file_list[i], self._test_label_list[i]), file=text_file)

		log.info('Finished saving list for the pedestrian dataset.')

	def _load_list(self):
		"""
		Load the shuffled file and label list from the .txt file.
		"""
		log.info('Loading train list for the pedestrian dataset from \'{}\'.'.format(self.train_list_path))
		log.info('Loading test list for the pedestrian dataset from \'{}\'.'.format(self.test_list_path))

		# check if train_list.txt and test_list.txt file exists
		if not file_exists(self.train_list_path):
			log.error('File \'{}\' does not exist!'.format(self.train_list_path))

		if not file_exists(self.test_list_path):
			log.error('File \'{}\' does not exist!'.format(self.test_list_path))

		# open from file
		with open(self.train_list_path, 'r') as text_file:
			for i, line in enumerate(text_file):
				self._train_file_list[i] = line.split()[1]
				self._train_label_list[i] = line.split()[2]

		with open(self.test_list_path, 'r') as text_file:
			for i, line in enumerate(text_file):
				self._test_file_list[i] = line.split()[1]
				self._test_label_list[i] = line.split()[2]

		log.info('Finished loading list for the pedestrian dataset.')

	def get_mean(self):
		"""
		Get the mean of training images.

		Returns:
			- list containing mean of training images
		"""
		return self._PEDESTRIAN_MEAN

	def get_num_classes(self):
		"""
		Get number of classes of the dataset.

		Returns:
			- number of classes
		"""
		return self._NUM_CLASSES

	def get_next_train_batch(self, augment=True, shuffle_after_each_epoch=True):
		# variables
		height, width = self.crop_shape
		start = self._batch_idx
		end = self._batch_idx + self.batch_size
		is_last = False

		# dynamically allocate batch size depending
		# on the location of current batch index
		if end < self._TOTAL_TRAIN_DATA:
			batch = np.zeros((self.batch_size, height, width, 3), dtype=np.float32)
			label = np.zeros(self.batch_size, dtype=np.int)
		else:
			batch = np.zeros((self._TOTAL_TRAIN_DATA-start, height, width, 3), dtype=np.float32)
			label = np.zeros(self._TOTAL_TRAIN_DATA-start, dtype=np.int)
			end = self._TOTAL_TRAIN_DATA
			is_last = True

		# load images from the self._train_file_list[] and assign to batch
		for i in range(start, end):
			# images
			image = Image.open(self._train_file_list[i]).convert('RGB')

			if self.isotropical is True:
				image_resized = isotropical_resize(image, min(self.resize), upscale=True)
			else:
				image_resized = image.resize(self.resize, resample=Image.ANTIALIAS)

			image_cropped = random_crop(image_resized, height, width)
			batch[i-start,:,:,:] = np.array(image_cropped, dtype=np.float32)

			# labels
			label[i-start] = self._train_label_list[i]

		# update the global static variable for batch index
		if is_last is True:
			self._batch_idx = 0
		else:
			self._batch_idx += self.batch_size

		# shuffle the self._train_file_list[] for each epoch
		if shuffle_after_each_epoch is True and is_last is True:
			shuffle_idx = np.arange(self._TOTAL_TRAIN_DATA)
			np.random.shuffle(shuffle_idx)
			self._train_file_list = self._train_file_list[shuffle_idx]
			self._train_label_list = self._train_label_list[shuffle_idx]

		# augment image if requested
		if augment is True:
			batch = augment_image_batch(
				batch,
				flr=0.5)

		# convert label to one-hot label format
		label_onehot = np.zeros((label.shape[0], self.get_num_classes()))
		label_onehot[np.arange(label.shape[0]), label] = 1

		return batch, label_onehot, is_last

	def get_next_test_batch(self):
		# variables
		height, width = self.crop_shape
		start = self._batch_idx
		end = self._batch_idx + self.batch_size
		is_last = False

		# dynamically allocate batch size depending
		# on the location of current batch index
		if end < self._TOTAL_TEST_DATA:
			batch = np.zeros((self.batch_size, height, width, 3), dtype=np.float32)
			label = np.zeros(self.batch_size, dtype=np.int)
		else:
			batch = np.zeros((self._TOTAL_TEST_DATA-start, height, width, 3), dtype=np.float32)
			label = np.zeros(self._TOTAL_TEST_DATA-start, dtype=np.int)
			end = self._TOTAL_TEST_DATA
			is_last = True

		# load images from the self._test_file_list[] and assign to batch
		for i in range(start, end):
			# images
			image = Image.open(self._test_file_list[i]).convert('RGB')

			if self.isotropical is True:
				image_resized = isotropical_resize(image, min(self.resize), upscale=True)
			else:
				image_resized = image.resize(self.resize, resample=Image.ANTIALIAS)

			image_cropped = random_crop(image_resized, height, width)
			batch[i-start,:,:,:] = np.array(image_cropped, dtype=np.float32)

			# labels
			label[i-start] = self._test_label_list[i]

		# update the global static variable for batch index
		if is_last is True:
			self._batch_idx = 0
		else:
			self._batch_idx += self.batch_size

		# convert label to one-hot label format
		label_onehot = np.zeros((label.shape[0], self.get_num_classes()))
		label_onehot[np.arange(label.shape[0]), label] = 1

		return batch, label_onehot, is_last

