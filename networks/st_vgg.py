import os
import numpy as np
import tensorflow as tf
import logging as log
from image_utils import *
from networks.vgg16 import VGG16
from networks.vgg19 import VGG19
from networks.network_common import NetworkCommon
from networks.stn.spatial_transformer import *

class ST_VGG:
	"""
	Class for the trainable Spatial Transformer VGG network.
	"""

	def __init__(self, num_classes=None, npy_path=None,
			init_layers=None, trainable=True, dropout_rate=0.5):
		"""
		ST_VGG class constructor.

		Inputs:
			- num_classes: number of classes to set the final FC layer output size
				note that this could be left to None if using only partial VGG net
			- npy_path: path of the vgg pre-trained weights in .npy format
				to be used for localization & classification in STN
				(python dictionary format: {'localization': path, 'classification': path})
			- init_layers: force the specified layers in this python list to
				truncated_normal instead of loading from the loc_data_dict
			- trainable: boolean. True if tf.Variable False if tf.constant
			- dropout_rate: dropout rate
		"""
		self.dropout_rate = dropout_rate

		self.nc_loc = NetworkCommon(
			init_layers=init_layers['localization'],
			npy_path=npy_path['localization'],
			trainable=trainable)
		self.vgg_loc = VGG16(
			npy_path=npy_path['localization'],
			trainable=trainable)
		self.vgg_cls = VGG16(
			num_classes=num_classes,
			npy_path=npy_path['classification'],
			init_layers=init_layers['classification'],
			trainable=trainable,
			dropout_rate=self.dropout_rate)

	def build(self, rgb, train_mode=None, **kwargs):

		log.info('STN VGG model build started')

		# STN
		with tf.variable_scope('localization'):
			loc_base = self.vgg_loc.build_partial(rgb, 'pool5', train_mode=train_mode)
			loc_conv6 = self.nc_loc.conv_layer(loc_base, 1, 512, 128, 'conv6')

			loc_fc7 = self.nc_loc.fc_layer(loc_conv6, 128, 128, False, 'fc7')
			loc_dropout7 = self.nc_loc.dropout_layer(loc_fc7, self.dropout_rate, train_mode, 'dropout7')

			loc_fc8_bias_init = np.array([0.5, 0, 0.5, 0], dtype=np.float32)
			loc_fc8 = self.nc_loc.fc_layer(loc_dropout7, 128, 4, True, 'fc8',
				init_bias=loc_fc8_bias_init)

			loc_fc8 = clip_transcale(loc_fc8)

			self.affine = tf.map_fn(transcale_2_affine, loc_fc8, dtype=tf.float32)

			out_size = (224, 224)
			stn_out = transformer(rgb, self.affine, out_size)

		# classification layer
		with tf.variable_scope('classification'):
			logits, prob = self.vgg_cls.build(stn_out, train_mode=train_mode)

		# convert translation / scale info into bounding box format
		pred_bounding_box = transcale_2_bb(loc_fc8)

		# draw bounding boxes
		for key, value in kwargs.items():
			if key is 'gt_bounding_box':
				rgb_bb = draw_bounding_boxes(rgb, [value, pred_bounding_box])
				tf.summary.image('input_with_gt_bb', rgb_bb, 3)
			else:
				raise RuntimeError('No matching argument')

		log.info('STN VGG model build finished')

		return logits, prob, pred_bounding_box
