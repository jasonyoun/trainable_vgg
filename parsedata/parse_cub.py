import os
import re
import numpy as np
import tensorflow as tf
import logging as log
from utils import *
from image_utils import *
from PIL import Image
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ParseCub:
	# global variables
	_TRAIN_LIST = 'train_list.txt'
	_VAL_LIST = 'val_list.txt'
	_TEST_LIST = 'test_list.txt'
	_IMAGES_TXT = 'images.txt'
	_IMAGE_CLASS_LABELS_TXT = 'image_class_labels.txt'
	_BOUNDING_BOXES_TXT = 'bounding_boxes.txt'
	_PARTS_LOCS_TXT = 'parts\\part_locs.txt'
	_CUB_FOLDER = 'CUB_200_2011'
	_IMAGES_FOLDER = 'images'
	_NUM_CLASSES = 200
	_NUM_TOTAL_IMGS = 11788
	_NUM_TRAIN_IMGS = 8251
	_NUM_VAL_IMGS = 1768
	_NUM_TEST_IMGS = _NUM_TOTAL_IMGS - _NUM_TRAIN_IMGS - _NUM_VAL_IMGS

	def __init__(self, dataset_dir, resize, crop_shape, batch_size, isotropical=False, initial_load=False):
		# some paths
		self.dataset_path = os.path.join(dataset_dir, self._CUB_FOLDER)
		self.images_path = os.path.join(self.dataset_path, self._IMAGES_FOLDER)
		self.train_list_path = os.path.join(self.dataset_path, self._TRAIN_LIST)
		self.val_list_path = os.path.join(self.dataset_path, self._VAL_LIST)
		self.test_list_path = os.path.join(self.dataset_path, self._TEST_LIST)
		self.images_txt_path = os.path.join(self.dataset_path, self._IMAGES_TXT)
		self.image_class_labels_path = os.path.join(self.dataset_path, self._IMAGE_CLASS_LABELS_TXT)
		self.bounding_boxes_path = os.path.join(self.dataset_path, self._BOUNDING_BOXES_TXT)
		self.parts_locs_path = os.path.join(self.dataset_path, self._PARTS_LOCS_TXT)

		# init numpy arrays
		self._train_id_list = np.empty(self._NUM_TRAIN_IMGS, dtype=np.int)
		self._train_file_list = np.empty(self._NUM_TRAIN_IMGS, dtype=object)
		self._train_label_list = np.empty(self._NUM_TRAIN_IMGS, dtype=np.int)
		self._train_bb_list = np.empty(self._NUM_TRAIN_IMGS, dtype=(np.float32, 4))
		self._train_pl_list = np.empty(self._NUM_TRAIN_IMGS, dtype=object)

		self._val_id_list = np.empty(self._NUM_VAL_IMGS, dtype=np.int)
		self._val_file_list = np.empty(self._NUM_VAL_IMGS, dtype=object)
		self._val_label_list = np.empty(self._NUM_VAL_IMGS, dtype=np.int)
		self._val_bb_list = np.empty(self._NUM_VAL_IMGS, dtype=(np.float32, 4))
		self._val_pl_list = np.empty(self._NUM_VAL_IMGS, dtype=object)

		self._test_id_list = np.empty(self._NUM_TEST_IMGS, dtype=np.int)
		self._test_file_list = np.empty(self._NUM_TEST_IMGS, dtype=object)
		self._test_label_list = np.empty(self._NUM_TEST_IMGS, dtype=np.int)
		self._test_bb_list = np.empty(self._NUM_TEST_IMGS, dtype=(np.float32, 4))
		self._test_pl_list = np.empty(self._NUM_TEST_IMGS, dtype=object)

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

		# get bounding box information from all file lists (train/val/test)
		# must come after either making or loading the list
		self._file_list_2_bb()

		# get part locations from all file lists (train/val/test)
		# must come after either making or loading the list
		self._file_list_2_part_locs()

		# using the part locations achieved above
		# update the part locations by adding parts like 'head' and so on
		self._update_part_locs()

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
				self._train_id_list[i] = images_lines[idx].split()[0]
				self._train_file_list[i] = os.path.join(self.images_path, images_lines[idx].split()[1])
				self._train_label_list[i] = labels_lines[idx].split()[1]

			for i, idx in enumerate(val_idx):
				self._val_id_list[i] = images_lines[idx].split()[0]
				self._val_file_list[i] = os.path.join(self.images_path, images_lines[idx].split()[1])
				self._val_label_list[i] = labels_lines[idx].split()[1]

			for i, idx in enumerate(test_idx):
				self._test_id_list[i] = images_lines[idx].split()[0]
				self._test_file_list[i] = os.path.join(self.images_path, images_lines[idx].split()[1])
				self._test_label_list[i] = labels_lines[idx].split()[1]

		# write to file
		with open(self.train_list_path, 'w') as text_file:
			for i in range(self._NUM_TRAIN_IMGS):
				print('{} {} {}'.format(self._train_id_list[i], self._train_file_list[i], self._train_label_list[i]), file=text_file)

		with open(self.val_list_path, 'w') as text_file:
			for i in range(self._NUM_VAL_IMGS):
				print('{} {} {}'.format(self._val_id_list[i], self._val_file_list[i], self._val_label_list[i]), file=text_file)

		with open(self.test_list_path, 'w') as text_file:
			for i in range(self._NUM_TEST_IMGS):
				print('{} {} {}'.format(self._test_id_list[i], self._test_file_list[i], self._test_label_list[i]), file=text_file)

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
				self._train_id_list[i] = line.split()[0]
				self._train_file_list[i] = line.split()[1]
				self._train_label_list[i] = line.split()[2]

			assert self._train_file_list.shape[0] == (i+1)

		with open(self.val_list_path, 'r') as text_file:
			for i, line in enumerate(text_file):
				self._val_id_list[i] = line.split()[0]
				self._val_file_list[i] = line.split()[1]
				self._val_label_list[i] = line.split()[2]

			assert self._val_file_list.shape[0] == (i+1)

		with open(self.test_list_path, 'r') as text_file:
			for i, line in enumerate(text_file):
				self._test_id_list[i] = line.split()[0]
				self._test_file_list[i] = line.split()[1]
				self._test_label_list[i] = line.split()[2]

			assert self._test_file_list.shape[0] == (i+1)

		log.info('Finished loading lists for the CUB dataset.')

	def _file_list_2_bb(self):
		log.info('Extracting bounding box information from file lists.')

		with open(self.bounding_boxes_path) as bb:
			bb_lines = bb.readlines()

			def _fill_bb_list(id_list, bb_list):
				for i, image_id in enumerate(id_list):
					bl = bb_lines[image_id-1]
					assert image_id == int(bl.split()[0])
					bb_list[i] = tuple([float(p) for p in bl.split()[1:]])

			# get bounding box information for each list
			_fill_bb_list(self._train_id_list, self._train_bb_list)
			_fill_bb_list(self._val_id_list, self._val_bb_list)
			_fill_bb_list(self._test_id_list, self._test_bb_list)

		log.info('Finished extracting bounding box information from file lists.')

	def _file_list_2_part_locs(self):
		log.info('Extracting part locations from file lists.')

		with open(self.parts_locs_path) as pl:
			pl_lines = pl.readlines()

			def _fill_pl_list(id_list, pl_list):
				for i, image_id in enumerate(id_list):
					offset = (image_id - 1) *  15
					assert image_id == int(pl_lines[offset].split()[0])

					all_part_locs = {}
					for j in range(15):
						line = pl_lines[offset+j]
						assert (j+1) == int(line.split()[1])
						if line.split()[4] == '1':
							all_part_locs[j+1] = tuple([float(p) for p in line.split()[2:4]])

					pl_list[i] = all_part_locs

			_fill_pl_list(self._train_id_list, self._train_pl_list)
			_fill_pl_list(self._val_id_list, self._val_pl_list)
			_fill_pl_list(self._test_id_list, self._test_pl_list)

		log.info('Finished extracting part locations from file lists.')

	def _update_part_locs(self):
		# private function to add head location to the part locations list
		def _update_head_2_pl_list(pl_list):
			to_delete = []

			for i in range(pl_list.size):
				# both left & right eye visible in the image -> use middle as head
				if 2 in pl_list[i] and 5 in pl_list[i] and 6 in pl_list[i] and 7 in pl_list[i] and 15 in pl_list[i]:
					x_beak, y_beak = pl_list[i][2]
					x_crown, y_crown = pl_list[i][5]
					x_forehead, y_forehead = pl_list[i][6]
					x_lefteye, y_lefteye = pl_list[i][7]
					x_throat, y_throat = pl_list[i][15]
					x_middle = float(x_beak + x_crown + x_forehead + x_lefteye + x_throat) / 5
					y_middle = float(y_beak + y_crown + y_forehead + y_lefteye + y_throat) / 5
					pl_list[i]['head'] = (x_middle, y_middle)
				elif 2 in pl_list[i] and 5 in pl_list[i] and 6 in pl_list[i] and 11 in pl_list[i] and 15 in pl_list[i]:
					x_beak, y_beak = pl_list[i][2]
					x_crown, y_crown = pl_list[i][5]
					x_forehead, y_forehead = pl_list[i][6]
					x_righteye, y_righteye = pl_list[i][11]
					x_throat, y_throat = pl_list[i][15]
					x_middle = float(x_beak + x_crown + x_forehead + x_righteye + x_throat) / 5
					y_middle = float(y_beak + y_crown + y_forehead + y_righteye + y_throat) / 5
					pl_list[i]['head'] = (x_middle, y_middle)
				elif 7 in pl_list[i] and 11 in pl_list[i]:
					x_left, y_left = pl_list[i][7]
					x_right, y_right = pl_list[i][11]
					x_middle = float(x_left + x_right) / 2
					y_middle = float(y_left + y_right) / 2
					pl_list[i]['head'] = (x_middle, y_middle)
				elif 7 in pl_list[i]:
					pl_list[i]['head'] = pl_list[i][7]
				elif 11 in pl_list[i]:
					pl_list[i]['head'] = pl_list[i][11]
				elif 5 in pl_list[i] and 15 in pl_list[i]:
					x_crown, y_crown = pl_list[i][5]
					x_throat, y_throat = pl_list[i][15]
					x_middle = float(x_crown + x_throat) / 2
					y_middle = float(y_crown + y_throat) / 2
					pl_list[i]['head'] = (x_middle, y_middle)
				elif 6 in pl_list[i] and 10 in pl_list[i]:
					x_forehead, y_forehead = pl_list[i][6]
					x_nape, y_nape = pl_list[i][10]
					x_middle = float(x_forehead + x_nape) / 2
					y_middle = float(y_forehead + y_nape) / 2
					pl_list[i]['head'] = (x_middle, y_middle)
				elif 15 in pl_list[i]:
					pl_list[i]['head'] = pl_list[i][15]
				elif 5 in pl_list[i]:
					pl_list[i]['head'] = pl_list[i][5]
				elif 2 in pl_list[i]:
					pl_list[i]['head'] = pl_list[i][2]
				else:
					to_delete.append(i)

			return to_delete

		# update head location to the part locations list for training set
		to_delete = _update_head_2_pl_list(self._train_pl_list)
		self._train_id_list = np.delete(self._train_id_list, to_delete, axis=0)
		self._train_file_list = np.delete(self._train_file_list, to_delete, axis=0)
		self._train_label_list = np.delete(self._train_label_list, to_delete, axis=0)
		self._train_bb_list = np.delete(self._train_bb_list, to_delete, axis=0)
		self._train_pl_list = np.delete(self._train_pl_list, to_delete, axis=0)
		self._NUM_TRAIN_IMGS -= len(to_delete)

		# update head location to the part locations list for validation set
		to_delete = _update_head_2_pl_list(self._val_pl_list)
		self._val_id_list = np.delete(self._val_id_list, to_delete, axis=0)
		self._val_file_list = np.delete(self._val_file_list, to_delete, axis=0)
		self._val_label_list = np.delete(self._val_label_list, to_delete, axis=0)
		self._val_bb_list = np.delete(self._val_bb_list, to_delete, axis=0)
		self._val_pl_list = np.delete(self._val_pl_list, to_delete, axis=0)
		self._NUM_VAL_IMGS -= len(to_delete)

		# update head location to the part locations list for test set
		to_delete = _update_head_2_pl_list(self._test_pl_list)
		self._test_id_list = np.delete(self._test_id_list, to_delete, axis=0)
		self._test_file_list = np.delete(self._test_file_list, to_delete, axis=0)
		self._test_label_list = np.delete(self._test_label_list, to_delete, axis=0)
		self._test_bb_list = np.delete(self._test_bb_list, to_delete, axis=0)
		self._test_pl_list = np.delete(self._test_pl_list, to_delete, axis=0)
		self._NUM_TEST_IMGS -= len(to_delete)

		print()
		print(len(to_delete))
		print()

	def get_num_classes(self):
		"""
		Get number of classes of the dataset.

		Returns:
			- number of classes
		"""
		return self._NUM_CLASSES

	def get_next_train_batch(self, augment=True):
		# variables
		width, height = self.crop_shape
		start = self._train_batch_idx
		end = self._train_batch_idx + self.batch_size
		is_last = False

		# dynamically allocate batch size depending
		# on the location of current batch index
		if end < self._NUM_TRAIN_IMGS:
			batch = np.zeros((self.batch_size, width, height, 3), dtype=np.float32)
			label = np.zeros(self.batch_size, dtype=np.int)
			bb = np.empty(self.batch_size, dtype=(np.float32, 4))
			pl = np.empty(self.batch_size, dtype=object)
		else:
			batch = np.zeros((self._NUM_TRAIN_IMGS-start, width, height, 3), dtype=np.float32)
			label = np.zeros(self._NUM_TRAIN_IMGS-start, dtype=np.int)
			bb = np.empty(self._NUM_TRAIN_IMGS-start, dtype=(np.float32, 4))
			pl = np.empty(self._NUM_TRAIN_IMGS-start, dtype=object)
			end = self._NUM_TRAIN_IMGS
			is_last = True

		# load images from the self._train_file_list[] and assign to batch
		i = start
		while i < end:
			# images
			image = Image.open(self._train_file_list[i]).convert('RGB')

			if self.isotropical is True:
				image_resized, bb_resized, pl_resized = isotropical_resize(
					image, min(self.resize), upscale=True, bounding_box=self._train_bb_list[i],
					part_locs=self._train_pl_list[i])
			else:
				image_resized, bb_resized, pl_resized = resize_image(
					image, self.resize, bounding_box=self._train_bb_list[i],
					part_locs=self._train_pl_list[i])

			# random crop image and adjust bounding box accordingly
			image_cropped, bb[i-start], pl[i-start] = random_crop(
				image_resized, width, height, bounding_box=bb_resized, part_locs=pl_resized)
			batch[i-start,:,:,:] = np.array(image_cropped, dtype=np.float32)

			# labels
			# subtract one to force label to start from 0
			label[i-start] = self._train_label_list[i] - 1

			# if head location is out of cropped region, try cropping again
			if check_part_locs_boundary(pl[i-start], width, height, 'head') is True:
				i -= 1

			i += 1

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
			self._train_bb_list = self._train_bb_list[shuffle_idx]
			self._train_pl_list = self._train_pl_list[shuffle_idx]

		# augment image if requested
		if augment is True:
			batch, bb, pl = augment_image_batch(
				batch,
				flr=0.5,
				bounding_box=bb,
				part_locs=pl)

		# # do some initial plots
		# for i in range(5, 20):
		# 	np_image = np.array(batch[i], dtype=np.float32) / 256.0

		# 	fig = plt.figure(i+1)
		# 	ax = fig.add_subplot(111, aspect='equal')
		# 	plt.imshow(np_image)

		# 	x, y, w, h = bb[i]
		# 	ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor='r'))

		# 	for key, value in pl[i].items():
		# 		x, y = value
		# 		if key is 'head':
		# 			#ax.add_patch(patches.Circle(value, fill=False, edgecolor='r'))
		# 			pass
		# 		else:
		# 			ax.add_patch(patches.Circle(value, fill=False, edgecolor='b'))
		# 			plt.text(x, y, '{}'.format(key), color='w')

		# 	plt.show()

		# rescale bounding box coordinates from pixels to values between 0 & 1
		bb = bb_pixels_2_relative(bb, width, height)
		pl = pl_pixels_2_relative(pl, width, height)

		# extract only specific parts from pl which contains all part locations
		head_pl = extract_part_from_part_locs(pl, 'head')

		# convert label to one-hot label format
		label_onehot = np.zeros((label.shape[0], self.get_num_classes()))
		label_onehot[np.arange(label.shape[0]), label] = 1

		return batch, label_onehot, bb, head_pl, is_last

	def get_next_val_batch(self):
		# variables
		width, height = self.crop_shape
		start = self._val_batch_idx
		end = self._val_batch_idx + self.batch_size
		is_last = False

		# dynamically allocate batch size depending
		# on the location of current batch index
		if end < self._NUM_VAL_IMGS:
			batch = np.zeros((self.batch_size, width, height, 3), dtype=np.float32)
			label = np.zeros(self.batch_size, dtype=np.int)
			bb = np.empty(self.batch_size, dtype=(np.float32, 4))
			pl = np.empty(self.batch_size, dtype=object)
		else:
			batch = np.zeros((self._NUM_VAL_IMGS-start, width, height, 3), dtype=np.float32)
			label = np.zeros(self._NUM_VAL_IMGS-start, dtype=np.int)
			bb = np.empty(self._NUM_VAL_IMGS-start, dtype=(np.float32, 4))
			pl = np.empty(self._NUM_VAL_IMGS-start, dtype=object)
			end = self._NUM_VAL_IMGS
			is_last = True

		# load images from the self._val_file_list[] and assign to batch
		i = start
		while i < end:
			# images
			image = Image.open(self._val_file_list[i]).convert('RGB')

			if self.isotropical is True:
				image_resized, bb_resized, pl_resized = isotropical_resize(
					image, min(self.resize), upscale=True, bounding_box=self._val_bb_list[i],
					part_locs=self._val_pl_list[i])
			else:
				image_resized, bb_resized, pl_resized = resize_image(
					image, self.resize, bounding_box=self._val_bb_list[i],
					part_locs=self._val_pl_list[i])

			# random crop image and adjust bounding box accordingly
			image_cropped, bb[i-start], pl[i-start] = random_crop(
				image_resized, width, height, bounding_box=bb_resized, part_locs=pl_resized)
			batch[i-start,:,:,:] = np.array(image_cropped, dtype=np.float32)

			# labels
			# subtract one to force label to start from 0
			label[i-start] = self._val_label_list[i] - 1

			# if head location is out of cropped region, try cropping again
			if check_part_locs_boundary(pl[i-start], width, height, 'head') is True:
				i -= 1

			i += 1

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
			self._val_bb_list = self._val_bb_list[shuffle_idx]
			self._val_pl_list = self._val_pl_list[shuffle_idx]

		# rescale bounding box coordinates from pixels to values between 0 & 1
		bb = bb_pixels_2_relative(bb, width, height)
		pl = pl_pixels_2_relative(pl, width, height)

		# extract only specific parts from pl which contains all part locations
		head_pl = extract_part_from_part_locs(pl, 'head')

		# convert label to one-hot label format
		label_onehot = np.zeros((label.shape[0], self.get_num_classes()))
		label_onehot[np.arange(label.shape[0]), label] = 1

		return batch, label_onehot, bb, head_pl, is_last

	def get_next_test_batch(self):
		# variables
		width, height = self.crop_shape
		start = self._test_batch_idx
		end = self._test_batch_idx + self.batch_size
		is_last = False

		# dynamically allocate batch size depending
		# on the location of current batch index
		if end < self._NUM_TEST_IMGS:
			batch = np.zeros((self.batch_size, width, height, 3), dtype=np.float32)
			label = np.zeros(self.batch_size, dtype=np.int)
			bb = np.empty(self.batch_size, dtype=(np.float32, 4))
			pl = np.empty(self.batch_size, dtype=object)
		else:
			batch = np.zeros((self._NUM_TEST_IMGS-start, width, height, 3), dtype=np.float32)
			label = np.zeros(self._NUM_TEST_IMGS-start, dtype=np.int)
			bb = np.empty(self._NUM_TEST_IMGS-start, dtype=(np.float32, 4))
			pl = np.empty(self._NUM_TEST_IMGS-start, dtype=object)
			end = self._NUM_TEST_IMGS
			is_last = True

		# load images from the self._test_file_list[] and assign to batch
		i = start
		while i < end:
			# images
			image = Image.open(self._test_file_list[i]).convert('RGB')

			if self.isotropical is True:
				image_resized, bb_resized, pl_resized = isotropical_resize(
					image, min(self.resize), upscale=True, bounding_box=self._test_bb_list[i],
					part_locs=self._test_pl_list[i])
			else:
				image_resized, bb_resized, pl_resized = resize_image(
					image, self.resize, bounding_box=self._test_bb_list[i],
					part_locs=self._test_pl_list[i])

			# random crop image and adjust bounding box accordingly
			image_cropped, bb[i-start], pl[i-start] = central_crop(
				image_resized, width, height, bounding_box=bb_resized, part_locs=pl_resized)
			batch[i-start,:,:,:] = np.array(image_cropped, dtype=np.float32)

			# labels
			# subtract one to force label to start from 0
			label[i-start] = self._test_label_list[i] - 1

			# if head location is out of cropped region, try cropping again
			# if check_part_locs_boundary(pl[i-start], width, height, 'head') is True:
			# 	i -= 1

			i += 1

		# update the global static variable for batch index
		if is_last is True:
			self._test_batch_idx = 0
		else:
			self._test_batch_idx += self.batch_size

		# rescale bounding box coordinates from pixels to values between 0 & 1
		bb = bb_pixels_2_relative(bb, width, height)
		pl = pl_pixels_2_relative(pl, width, height)

		# extract only specific parts from pl which contains all part locations
		head_pl = extract_part_from_part_locs(pl, 'head')

		# convert label to one-hot label format
		label_onehot = np.zeros((label.shape[0], self.get_num_classes()))
		label_onehot[np.arange(label.shape[0]), label] = 1

		return batch, label_onehot, bb, head_pl, is_last
