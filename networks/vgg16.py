import os
import numpy as np
import tensorflow as tf
import logging as log
from image_utils import *
from networks.network_common import NetworkCommon

class VGG16:
	"""
	Class for the trainable VGG-16 network.
	"""

	_VGG_RGB_MEAN = [123.68, 116.779, 103.939]

	def __init__(self, num_classes=None, vgg16_npy_path=None,
			init_layers=None, trainable=True, dropout_rate=0.5):
		"""
		VGG-16 class constructor.

		Inputs:
			- num_classes: number of classes to set the final FC layer output size.
				note that this could be left to None if using only partial VGG net
			- vgg16_npy_path: path of the vgg16 pre-trained weights in .npy format
			- init_layers: force the specified layers in this python list to
				truncated_normal instead of loading from the data_dict
			- trainable: boolean. True if tf.Variable False if tf.constant
			- dropout_rate: dropout rate
		"""
		if vgg16_npy_path is not None:
			data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
		else:
			data_dict = None

		self.num_classes = num_classes
		self.dropout_rate = dropout_rate
		self.nc = NetworkCommon(init_layers, data_dict, trainable)

	def build(self, rgb, train_mode=None):
		"""
		Build the full VGG-16 network.

		Inputs:
			- rgb: input image in rgb format in range [0, 255]
			- train_mode: boolean. True if training mode False if testing mode

		Returns:
			- fc8: logits output
			- prob: probability output
		"""

		log.info('Full VGG 16 model build started')

		# call build_partial() to build the full network until 'fc8' layer
		fc8 = self.build_partial(rgb, 'fc8', train_mode=train_mode)

		# softmax loss
		prob = tf.nn.softmax(fc8, name="prob")

		log.info('Full VGG 16 model build finished')

		return fc8, prob

	def build_partial(self, rgb, build_until, train_mode=None):
		"""
		Build partial VGG-16 network.
		For example, one can call this to build a VGG-16 without FC layers.

		Inputs:
			- rgb: input image in rgb format in range [0, 255]
			- build_until: string denoting the layer name until which to build
			- train_mode: boolean. True if training mode False if testing mode

		Returns:
			- layer specified by the input 'build_until'
		"""

		log.info('Partial VGG 16 model build started')

		tf.summary.image('input', rgb, 3)

		# Convert RGB to BGR
		centered_bgr = rgb_2_centered_bgr(rgb, self._VGG_RGB_MEAN)

		# conv layers
		conv1_1 = self.nc.conv_layer(centered_bgr, 3, 3, 64, 'conv1_1')
		if build_until is 'conv1_1': return conv1_1
		conv1_2 = self.nc.conv_layer(conv1_1, 3, 64, 64, 'conv1_2')
		if build_until is 'conv1_2': return conv1_2
		pool1 = self.nc.max_pool(conv1_2, 'pool1')
		if build_until is 'pool1': return pool1

		conv2_1 = self.nc.conv_layer(pool1, 3, 64, 128, 'conv2_1')
		if build_until is 'conv2_1': return conv2_1
		conv2_2 = self.nc.conv_layer(conv2_1, 3, 128, 128, 'conv2_2')
		if build_until is 'conv2_2': return conv2_2
		pool2 = self.nc.max_pool(conv2_2, 'pool2')
		if build_until is 'pool2': return pool2

		conv3_1 = self.nc.conv_layer(pool2, 3, 128, 256, 'conv3_1')
		if build_until is 'conv3_1': return conv3_1
		conv3_2 = self.nc.conv_layer(conv3_1, 3, 256, 256, 'conv3_2')
		if build_until is 'conv3_2': return conv3_2
		conv3_3 = self.nc.conv_layer(conv3_2, 3, 256, 256, 'conv3_3')
		if build_until is 'conv3_3': return conv3_3
		pool3 = self.nc.max_pool(conv3_3, 'pool3')
		if build_until is 'pool3': return pool3

		conv4_1 = self.nc.conv_layer(pool3, 3, 256, 512, 'conv4_1')
		if build_until is 'conv4_1': return conv4_1
		conv4_2 = self.nc.conv_layer(conv4_1, 3, 512, 512, 'conv4_2')
		if build_until is 'conv4_2': return conv4_2
		conv4_3 = self.nc.conv_layer(conv4_2, 3, 512, 512, 'conv4_3')
		if build_until is 'conv4_3': return conv4_3
		pool4 = self.nc.max_pool(conv4_3, 'pool4')
		if build_until is 'pool4': return pool4

		conv5_1 = self.nc.conv_layer(pool4, 3, 512, 512, 'conv5_1')
		if build_until is 'conv5_1': return conv5_1
		conv5_2 = self.nc.conv_layer(conv5_1, 3, 512, 512, 'conv5_2')
		if build_until is 'conv5_2': return conv5_2
		conv5_3 = self.nc.conv_layer(conv5_2, 3, 512, 512, 'conv5_3')
		if build_until is 'conv5_3': return conv5_3
		pool5 = self.nc.max_pool(conv5_3, 'pool5')
		if build_until is 'pool5': return pool5

		# fc layers
		fc6 = self.nc.fc_layer(pool5, 4096, False, 'fc6')
		if build_until is 'fc6': return fc6
		dropout6 = self.nc.dropout_layer(fc6, self.dropout_rate, train_mode, 'dropout6')
		if build_until is 'dropout6': return dropout6

		fc7 = self.nc.fc_layer(dropout6, 4096, False, 'fc7')
		if build_until is 'fc7': return fc7
		dropout7 = self.nc.dropout_layer(fc7, self.dropout_rate, train_mode, 'dropout7')
		if build_until is 'dropout7': return dropout7

		fc8 = self.nc.fc_layer(dropout7, self.num_classes, True, 'fc8')
		if build_until is 'fc8': return fc8

		log.warning('Matching keyword not found for partial VGG 16 model build!')
