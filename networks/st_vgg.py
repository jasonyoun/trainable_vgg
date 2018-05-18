import os
import numpy as np
import tensorflow as tf
import logging as log
from image_utils import *
from networks.vgg16 import VGG16
from networks.vgg19 import VGG19
from networks.network_common import NetworkCommon
from networks.stn.spatial_transformer import transformer

class ST_VGG:
	"""
	Class for the trainable Spatial Transformer VGG network.
	"""

	def __init__(self, num_classes=None, loc_vgg_npy_path=None, cls_vgg_npy_path=None,
			init_layers=None, trainable=True, dropout_rate=0.5):
		"""
		ST_VGG class constructor.

		Inputs:
			- num_classes: number of classes to set the final FC layer output size
				note that this could be left to None if using only partial VGG net
			- loc_vgg_npy_path: path of the vgg16 pre-trained weights in .npy format
				to be used for localization in STN
			- cls_vgg_npy_path: path of the vgg16 pre-trained weights in .npy format
				to be used for classification of the attended region
			- init_layers: force the specified layers in this python list to
				truncated_normal instead of loading from the loc_data_dict
			- trainable: boolean. True if tf.Variable False if tf.constant
			- dropout_rate: dropout rate
		"""
		self.dropout_rate = dropout_rate

		self.nc_loc = NetworkCommon(
			init_layers=[],
			npy_path=loc_vgg_npy_path,
			trainable=trainable)
		self.vgg_loc = VGG16(
			vgg16_npy_path=loc_vgg_npy_path,
			trainable=trainable)
		self.vgg_cls = VGG16(
			num_classes=num_classes,
			vgg16_npy_path=cls_vgg_npy_path,
			init_layers=[],
			trainable=trainable,
			dropout_rate=self.dropout_rate)

	def build(self, rgb, train_mode=None, **kwargs):

		log.info('STN VGG model build started')

		# STN
		with tf.variable_scope('localization'):
			loc_base = self.vgg_loc.build_partial(rgb, 'pool5', train_mode=train_mode)
			loc_conv6 = self.nc_loc.conv_layer(loc_base, 1, 512, 128, 'conv6')

			loc_fc7 = self.nc_loc.fc_layer(loc_conv6, 128, False, 'fc7')
			loc_dropout7 = self.nc_loc.dropout_layer(loc_fc7, self.dropout_rate, train_mode, 'dropout7')

			loc_fc8_bias_init = np.array([0.5, 0, 0.5, 0], dtype=np.float32)
			loc_fc8 = self.nc_loc.fc_layer(loc_dropout7, 4, True, 'fc8',
				init_bias=loc_fc8_bias_init)

			def batch_concat_scale_translate(x):
				w, tx, h, ty = tf.split(x, 4)
				return tf.concat([w, [0], tx, [0], h, ty], axis=0)

			def batch_concat_translate(x):
				tx, ty = tf.split(x, 2)
				return tf.concat([[0.5], [0.], tx, [0.], [0.5], ty], axis=0)

			self.affine = tf.map_fn(batch_concat_scale_translate, loc_fc8, dtype=tf.float32)

			out_size = (224, 224)
			stn_out = transformer(rgb, self.affine, out_size)

		# classification layer
		with tf.variable_scope('classification'):
			logits, prob = self.vgg_cls.build(stn_out, train_mode=train_mode)

		# convert translation / scale info into bounding box format
		pred_bounding_box = self._affine_ts_2_bb(loc_fc8)

		# draw bounding boxes
		for key, value in kwargs.items():
			if key is 'gt_bounding_box':
				rgb_bb = draw_bounding_boxes(rgb, [value, pred_bounding_box])
				tf.summary.image('input_with_gt_bb', rgb_bb, 3)
			else:
				raise RuntimeError('No matching argument')

		log.info('STN VGG model build finished')

		return logits, prob, pred_bounding_box

	def _affine_ts_2_bb(self, ts_matrix):
		w, tx, h, ty = tf.split(ts_matrix, 4, axis=1)

		x = (1 - w + tx) / 2
		y = (1 - h + ty) / 2

		bounding_boxes = tf.concat([x, y, w, h], axis=1)

		return bounding_boxes

	def _bb_2_affine_ts(self, bounding_boxes):
		x, y, w, h = tf.split(bounding_boxes, 4, axis=1)

		tx = 2*x + w - 1
		ty = 2*y + h - 1

		affine_ts = tf.concat([w, tx, h, ty], axis=1)

		return affine_ts

	def _clip_bb(self, bounding_boxes):
		x, y, w, h = tf.split(bounding_boxes, 4, axis=1)

		x_new = tf.where(x < 0, tf.zeros_like(x), x)
		y_new = tf.where(y < 0, tf.zeros_like(y), y)

		w_new = tf.where(x < 0, (w + x), w)
		h_new = tf.where(y < 0, (h + y), h)

		w_new = tf.where(x_new+w_new>1, (1 - x_new), w_new)
		h_new = tf.where(y_new+h_new>1, (1 - y_new), h_new)

		clipped_bb = tf.concat([x_new, y_new, w_new, h_new])

		return clipped_bb
