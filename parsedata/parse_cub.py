import os
import numpy as np
import tensorflow as tf
import logging as log
from utils import *
from image_utils import *
from PIL import Image

class ParseCub:
	# global variables
	_TRAIN_LIST = 'train_list.txt'
	_VAL_LIST = 'val_list.txt'
	_TEST_LIST = 'test_list.txt'
	_IMAGES_TXT = 'images.txt'
	_IMAGE_CLASS_LABELS_TXT = 'image_class_labels.txt'
	_CUB_FOLDER = 'CUB_200_2011'
	_IMAGES_FOLDER = 'images'
	_NUM_CLASSES = 200
	_train_file_list = None
	_train_label_list = None
	_test_file_list = None
	_test_label_list = None
	_NUM_TOTAL_IMGS = 11788
	_NUM_TRAIN_IMGS = 8251
	_NUM_VAL_IMGS = 1768
	_NUM_TEST_IMGS = _NUM_TOTAL_IMGS - _NUM_TRAIN_IMGS - _NUM_VAL_IMGS
	_train_batch_idx = None
	_val_batch_idx = None
	_test_batch_idx = None

	def __init__(self, dataset_dir, resize, crop_shape, batch_size, isotropical=False, initial_load=False):
		self.dataset_path = os.path.join(dataset_dir, self._CUB_FOLDER)
		self.images_path = os.path.join(self.dataset_path, self._IMAGES_FOLDER)
		self.train_list_path = os.path.join(self.dataset_path, self._TRAIN_LIST)
		self.val_list_path = os.path.join(self.dataset_path, self._VAL_LIST)
		self.test_list_path = os.path.join(self.dataset_path, self._TEST_LIST)
		self.images_txt_path = os.path.join(self.dataset_path, self._IMAGES_TXT)
		self.image_class_labels_path = os.path.join(self.dataset_path, self._IMAGE_CLASS_LABELS_TXT)

		# init numpy arrays
		self._train_file_list = np.empty(self._NUM_TRAIN_IMGS, dtype=object)
		self._train_label_list = np.empty(self._NUM_TRAIN_IMGS, dtype=np.int)
		self._val_file_list = np.empty(self._NUM_VAL_IMGS, dtype=object)
		self._val_label_list = np.empty(self._NUM_VAL_IMGS, dtype=np.int)
		self._test_file_list = np.empty(self._NUM_TEST_IMGS, dtype=object)
		self._test_label_list = np.empty(self._NUM_TEST_IMGS, dtype=np.int)

		# other inits
		self.resize = resize
		self.crop_shape = crop_shape
		self.batch_size = batch_size
		self.isotropical = isotropical
		self._train_batch_idx = 0
		self._val_batch_idx = 0
		self._test_batch_idx = 0

		# load the list or make one if loading for the first time
		if initial_load is True:
			self._make_list()
		else:
			self._load_list()

	def _make_list(self):
		"""
		After reading the file names from the training dataset,
		shuffle the data and save file list and label list
		to a single .txt file.
		"""
		log.info('Saving train list for the CUB dataset to \'{}\'.'.format(self.train_list_path))
		log.info('Saving validation list for the CUB dataset to \'{}\'.'.format(self.val_list_path))
		log.info('Saving test list for the CUB dataset to \'{}\'.'.format(self.test_list_path))

		# check if train_list.txt and test_list.txt file exists
		if file_exists(self.train_list_path):
			log.warning('File \'{}\' will be overwritten!'.format(self.train_list_path))

		if file_exists(self.val_list_path):
			log.warning('File \'{}\' will be overwritten!'.format(self.val_list_path))

		if file_exists(self.test_list_path):
			log.warning('File \'{}\' will be overwritten!'.format(self.test_list_path))

		shuffle_idx = np.arange(self._NUM_TOTAL_IMGS)
		np.random.shuffle(shuffle_idx)
		train_idx = shuffle_idx[:self._NUM_TRAIN_IMGS]
		val_idx = shuffle_idx[self._NUM_TRAIN_IMGS:(self._NUM_TRAIN_IMGS+self._NUM_VAL_IMGS)]
		test_idx = shuffle_idx[(self._NUM_TRAIN_IMGS+self._NUM_VAL_IMGS):]

		with open(self.images_txt_path) as images, open(self.image_class_labels_path) as labels:
			images_lines = images.readlines()
			labels_lines = labels.readlines()

			for i, idx in enumerate(train_idx):
				self._train_file_list[i] = os.path.join(self.images_path, images_lines[idx].split()[1])
				self._train_label_list[i] = labels_lines[idx].split()[1]

			for i, idx in enumerate(val_idx):
				self._val_file_list[i] = os.path.join(self.images_path, images_lines[idx].split()[1])
				self._val_label_list[i] = labels_lines[idx].split()[1]

			for i, idx in enumerate(test_idx):
				self._test_file_list[i] = os.path.join(self.images_path, images_lines[idx].split()[1])
				self._test_label_list[i] = labels_lines[idx].split()[1]

		# write to file
		with open(self.train_list_path, 'w') as text_file:
			for i in range(self._NUM_TRAIN_IMGS):
				print('{:05d} {} {}'.format(i+1, self._train_file_list[i], self._train_label_list[i]), file=text_file)

		with open(self.val_list_path, 'w') as text_file:
			for i in range(self._NUM_VAL_IMGS):
				print('{:05d} {} {}'.format(i+1, self._val_file_list[i], self._val_label_list[i]), file=text_file)

		with open(self.test_list_path, 'w') as text_file:
			for i in range(self._NUM_TEST_IMGS):
				print('{:05d} {} {}'.format(i+1, self._test_file_list[i], self._test_label_list[i]), file=text_file)

		log.info('Finished saving train / val / test lists for the CUB dataset.')

	def _load_list(self):
		log.info('Loading train list for the CUB dataset from \'{}\'.'.format(self.train_list_path))
		log.info('Loading val list for the CUB dataset from \'{}\'.'.format(self.val_list_path))
		log.info('Loading test list for the CUB dataset from \'{}\'.'.format(self.test_list_path))

		# check if train_list.txt and test_list.txt file exists
		if not file_exists(self.train_list_path):
			log.error('File \'{}\' does not exist!'.format(self.train_list_path))

		if not file_exists(self.val_list_path):
			log.error('File \'{}\' does not exist!'.format(self.val_list_path))

		if not file_exists(self.test_list_path):
			log.error('File \'{}\' does not exist!'.format(self.test_list_path))

		# open from file
		with open(self.train_list_path, 'r') as text_file:
			for i, line in enumerate(text_file):
				self._train_file_list[i] = line.split()[1]
				self._train_label_list[i] = line.split()[2]

			assert self._train_file_list.shape[0] == (i+1)

		with open(self.val_list_path, 'r') as text_file:
			for i, line in enumerate(text_file):
				self._val_file_list[i] = line.split()[1]
				self._val_label_list[i] = line.split()[2]

			assert self._val_file_list.shape[0] == (i+1)

		with open(self.test_list_path, 'r') as text_file:
			for i, line in enumerate(text_file):
				self._test_file_list[i] = line.split()[1]
				self._test_label_list[i] = line.split()[2]

			assert self._test_file_list.shape[0] == (i+1)

		log.info('Finished loading lists for the CUB dataset.')

	def get_num_classes(self):
		"""
		Get number of classes of the dataset.

		Returns:
			- number of classes
		"""
		return self._NUM_CLASSES

	def get_next_train_batch(self, augment=True):
		# variables
		height, width = self.crop_shape
		start = self._train_batch_idx
		end = self._train_batch_idx + self.batch_size
		is_last = False

		# dynamically allocate batch size depending
		# on the location of current batch index
		if end < self._NUM_TRAIN_IMGS:
			batch = np.zeros((self.batch_size, height, width, 3), dtype=np.float32)
			label = np.zeros(self.batch_size, dtype=np.int)
		else:
			batch = np.zeros((self._NUM_TRAIN_IMGS-start, height, width, 3), dtype=np.float32)
			label = np.zeros(self._NUM_TRAIN_IMGS-start, dtype=np.int)
			end = self._NUM_TRAIN_IMGS
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
			# subtract one to force label to start from 0
			label[i-start] = self._train_label_list[i] - 1

		# update the global static variable for batch index
		if is_last is True:
			self._train_batch_idx = 0
		else:
			self._train_batch_idx += self.batch_size

		# shuffle the self._train_file_list[] for each epoch
		if is_last is True:
			shuffle_idx = np.arange(self._NUM_TRAIN_IMGS)
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

	def get_next_val_batch(self):
		# variables
		height, width = self.crop_shape
		start = self._val_batch_idx
		end = self._val_batch_idx + self.batch_size
		is_last = False

		# dynamically allocate batch size depending
		# on the location of current batch index
		if end < self._NUM_VAL_IMGS:
			batch = np.zeros((self.batch_size, height, width, 3), dtype=np.float32)
			label = np.zeros(self.batch_size, dtype=np.int)
		else:
			batch = np.zeros((self._NUM_VAL_IMGS-start, height, width, 3), dtype=np.float32)
			label = np.zeros(self._NUM_VAL_IMGS-start, dtype=np.int)
			end = self._NUM_VAL_IMGS
			is_last = True

		# load images from the self._val_file_list[] and assign to batch
		for i in range(start, end):
			# images
			image = Image.open(self._val_file_list[i]).convert('RGB')

			if self.isotropical is True:
				image_resized = isotropical_resize(image, min(self.resize), upscale=True)
			else:
				image_resized = image.resize(self.resize, resample=Image.ANTIALIAS)

			image_cropped = central_crop(image_resized, height, width)
			batch[i-start,:,:,:] = np.array(image_cropped, dtype=np.float32)

			# labels
			# subtract one to force label to start from 0
			label[i-start] = self._val_label_list[i] - 1

		# update the global static variable for batch index
		if is_last is True:
			self._val_batch_idx = 0
		else:
			self._val_batch_idx += self.batch_size

		# shuffle the self._val_file_list[] for each epoch
		if is_last is True:
			shuffle_idx = np.arange(self._NUM_VAL_IMGS)
			np.random.shuffle(shuffle_idx)
			self._val_file_list = self._val_file_list[shuffle_idx]
			self._val_label_list = self._val_label_list[shuffle_idx]

		# convert label to one-hot label format
		label_onehot = np.zeros((label.shape[0], self.get_num_classes()))
		label_onehot[np.arange(label.shape[0]), label] = 1

		return batch, label_onehot, is_last

	def get_next_test_batch(self):
		# variables
		height, width = self.crop_shape
		start = self._test_batch_idx
		end = self._test_batch_idx + self.batch_size
		is_last = False

		# dynamically allocate batch size depending
		# on the location of current batch index
		if end < self._NUM_TEST_IMGS:
			batch = np.zeros((self.batch_size, height, width, 3), dtype=np.float32)
			label = np.zeros(self.batch_size, dtype=np.int)
		else:
			batch = np.zeros((self._NUM_TEST_IMGS-start, height, width, 3), dtype=np.float32)
			label = np.zeros(self._NUM_TEST_IMGS-start, dtype=np.int)
			end = self._NUM_TEST_IMGS
			is_last = True

		# load images from the self._test_file_list[] and assign to batch
		for i in range(start, end):
			# images
			image = Image.open(self._test_file_list[i]).convert('RGB')

			if self.isotropical is True:
				image_resized = isotropical_resize(image, min(self.resize), upscale=True)
			else:
				image_resized = image.resize(self.resize, resample=Image.ANTIALIAS)

			image_cropped = central_crop(image_resized, height, width)
			batch[i-start,:,:,:] = np.array(image_cropped, dtype=np.float32)

			# labels
			# subtract one to force label to start from 0
			label[i-start] = self._test_label_list[i] - 1

		# update the global static variable for batch index
		if is_last is True:
			self._test_batch_idx = 0
		else:
			self._test_batch_idx += self.batch_size

		# convert label to one-hot label format
		label_onehot = np.zeros((label.shape[0], self.get_num_classes()))
		label_onehot[np.arange(label.shape[0]), label] = 1

		return batch, label_onehot, is_last

