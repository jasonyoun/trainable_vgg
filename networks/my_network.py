import os
import numpy as np
import tensorflow as tf
import logging as log
from image_utils import *
from networks.network_common import NetworkCommon
from networks.stn.spatial_transformer import *

class MyNetwork:
	"""
	Class for the trainable Spatial Transformer VGG network.
	"""
	_OUT_SIZE = (7, 7)
	_RGB_MEAN = [123.68, 116.779, 103.939]

	def __init__(self, num_classes=None, npy_path=None,
			init_layers=None, trainable=True, dropout_rate=0.5):
		"""
		MyNetwork class constructor.

		Inputs:
			- num_classes: number of classes to set the final FC layer output size
				note that this could be left to None if using only partial VGG net
			- npy_path: path of the pre-trained weights in .npy format
				to be used for localization & classification in STN
				(python dictionary format: {'localization': path, 'classification': path})
			- init_layers: force the specified layers in this python list to
				truncated_normal instead of loading from the loc_data_dict
			- trainable: boolean. True if tf.Variable False if tf.constant
			- dropout_rate: dropout rate
		"""
		self.num_classes = num_classes
		self.dropout_rate = dropout_rate

		self.vgg = NetworkCommon(
			init_layers=init_layers['vgg'],
			npy_path=npy_path['vgg'],
			trainable=trainable)

		self.stn_1 = NetworkCommon(
			init_layers=init_layers['stn_1'],
			npy_path=npy_path['stn_1'],
			trainable=trainable)

		self.stn_2 = NetworkCommon(
			init_layers=init_layers['stn_2'],
			npy_path=npy_path['stn_2'],
			trainable=trainable)

		# self.stn_3 = NetworkCommon(
		# 	init_layers=init_layers['stn_3'],
		# 	npy_path=npy_path['stn_3'],
		# 	trainable=trainable)

		self.cls = NetworkCommon(
			init_layers=init_layers['classification'],
			npy_path=npy_path['classification'],
			trainable=trainable,
			stddev=0.01)

	def build(self, rgb, train_mode=None, **kwargs):

		log.info('STN VGG model build started')

		tf.summary.image('input', rgb, 3)

		# Convert RGB to BGR
		centered_bgr = rgb_2_centered_bgr(rgb, self._RGB_MEAN)

		# vgg base network
		with tf.variable_scope('vgg'):
			# conv layers
			vgg_conv1_1 = self.vgg.conv_layer(centered_bgr, 3, 3, 64, 'conv1_1')
			vgg_conv1_2 = self.vgg.conv_layer(vgg_conv1_1, 3, 64, 64, 'conv1_2')
			vgg_pool1 = self.vgg.max_pool(vgg_conv1_2, 'pool1')

			vgg_conv2_1 = self.vgg.conv_layer(vgg_pool1, 3, 64, 128, 'conv2_1')
			vgg_conv2_2 = self.vgg.conv_layer(vgg_conv2_1, 3, 128, 128, 'conv2_2')
			vgg_pool2 = self.vgg.max_pool(vgg_conv2_2, 'pool2')

			vgg_conv3_1 = self.vgg.conv_layer(vgg_pool2, 3, 128, 256, 'conv3_1')
			vgg_conv3_2 = self.vgg.conv_layer(vgg_conv3_1, 3, 256, 256, 'conv3_2')
			vgg_conv3_3 = self.vgg.conv_layer(vgg_conv3_2, 3, 256, 256, 'conv3_3')
			vgg_conv3_4 = self.vgg.conv_layer(vgg_conv3_3, 3, 256, 256, 'conv3_4')
			vgg_pool3 = self.vgg.max_pool(vgg_conv3_4, 'pool3')

			vgg_conv4_1 = self.vgg.conv_layer(vgg_pool3, 3, 256, 512, 'conv4_1')
			vgg_conv4_2 = self.vgg.conv_layer(vgg_conv4_1, 3, 512, 512, 'conv4_2')
			vgg_conv4_3 = self.vgg.conv_layer(vgg_conv4_2, 3, 512, 512, 'conv4_3')
			vgg_conv4_4 = self.vgg.conv_layer(vgg_conv4_3, 3, 512, 512, 'conv4_4')
			vgg_pool4 = self.vgg.max_pool(vgg_conv4_4, 'pool4')

			vgg_conv5_1 = self.vgg.conv_layer(vgg_pool4, 3, 512, 512, 'conv5_1')
			vgg_conv5_2 = self.vgg.conv_layer(vgg_conv5_1, 3, 512, 512, 'conv5_2')
			vgg_conv5_3 = self.vgg.conv_layer(vgg_conv5_2, 3, 512, 512, 'conv5_3')
			vgg_conv5_4 = self.vgg.conv_layer(vgg_conv5_3, 3, 512, 512, 'conv5_4')
			vgg_pool5 = self.vgg.max_pool(vgg_conv5_4, 'pool5')

		# stn_1
		with tf.variable_scope('stn_1'):
			stn_1_conv1 = self.stn_1.conv_layer(vgg_pool5, 1, 512, 128, 'conv1')

			stn_1_fc2 = self.stn_1.fc_layer(stn_1_conv1, 6272, 128, False, 'fc2')
			stn_1_dropout2 = self.stn_1.dropout_layer(stn_1_fc2, self.dropout_rate, train_mode, 'dropout2')

			stn_1_fc3 = self.stn_1.fc_layer(stn_1_dropout2, 128, 4, True, 'fc3',
				init_bias=np.array([1, 0, 1, 0], dtype=np.float32))

			stn_1_fc3 = clip_transcale(stn_1_fc3)

			stn_1_affine = tf.map_fn(transcale_2_affine, stn_1_fc3, dtype=tf.float32)

			stn_1_out = transformer(vgg_conv5_4, stn_1_affine, self._OUT_SIZE)

		# stn_2 a.k.a. head locator
		with tf.variable_scope('stn_2'):
			stn_2_conv1 = self.stn_2.conv_layer(stn_1_out, 1, 512, 128, 'conv1')

			stn_2_fc2 = self.stn_2.fc_layer(stn_2_conv1, 6272, 128, False, 'fc2')
			stn_2_dropout2 = self.stn_2.dropout_layer(stn_2_fc2, self.dropout_rate, train_mode, 'dropout2')

			stn_2_fc3 = self.stn_2.fc_layer(stn_2_dropout2, 128, 2, True, 'fc3',
				init_bias=np.array([0, 0], dtype=np.float32))

			stn_2_fc3 = rel_trans_2_abs_pl(transcale_2_bb(stn_1_fc3), stn_2_fc3)

			stn_2_affine = tf.map_fn(trans_2_affine, (stn_1_fc3, stn_2_fc3), dtype=tf.float32)

			stn_2_out = transformer(vgg_conv5_4, stn_2_affine, self._OUT_SIZE)

		# # stn_3 a.k.a. body locator
		# with tf.variable_scope('stn_3'):
		# 	stn_3_conv1 = self.stn_3.conv_layer(stn_1_out, 1, 512, 128, 'conv1')

		# 	stn_3_fc2 = self.stn_3.fc_layer(stn_3_conv1, 6272, 128, False, 'fc2')
		# 	stn_3_dropout2 = self.stn_3.dropout_layer(stn_3_fc2, self.dropout_rate, train_mode, 'dropout2')

		# 	stn_3_fc3 = self.stn_3.fc_layer(stn_3_dropout2, 128, 2, True, 'fc3',
		# 		init_bias=np.array([0, 0], dtype=np.float32))

		# 	stn_3_fc3 = rel_trans_2_abs_pl(transcale_2_bb(stn_1_fc3), stn_3_fc3)

		# 	stn_3_affine = tf.map_fn(trans_2_affine, (stn_1_fc3, stn_3_fc3), dtype=tf.float32)

		# 	stn_3_out = transformer(vgg_conv5_4, stn_3_affine, self._OUT_SIZE)

		# classification fc layer
		with tf.variable_scope('classification'):
			pre_cls_concat = tf.concat([stn_1_out, stn_2_out], axis=3)
			#cls_conv1 = self.cls.conv_layer(pre_cls_concat, 1, 1024, 512, 'conv1')

			cls_fc2 = self.cls.fc_layer(pre_cls_concat, 50176, 4096, False, 'fc2')
			cls_dropout2 = self.cls.dropout_layer(cls_fc2, self.dropout_rate, train_mode, 'dropout2')

			cls_fc3 = self.cls.fc_layer(cls_dropout2, 4096, 4096, False, 'fc3')
			cls_dropout3 = self.cls.dropout_layer(cls_fc3, self.dropout_rate, train_mode, 'dropout3')

			cls_fc4 = self.cls.fc_layer(cls_dropout3, 4096, self.num_classes, True, 'fc4')

			# softmax loss
			prob = tf.nn.softmax(cls_fc4, name="prob")

		# convert translation / scale info into bounding box format
		pred_bounding_box = transcale_2_bb(stn_1_fc3)
		pred_head_pl = stn_2_fc3
		# pred_body_pl = stn_3_fc3

		# draw bounding boxes
		bounding_boxes_list = []

		if 'gt_bounding_box' in kwargs:
			bounding_boxes_list.append(kwargs['gt_bounding_box'])
			bounding_boxes_list.append(pred_bounding_box)

		if 'gt_part_loc_head' in kwargs and 'gt_bounding_box' in kwargs:
			bounding_boxes_list.append(abs_pl_2_bb(kwargs['gt_bounding_box'], kwargs['gt_part_loc_head']))
			bounding_boxes_list.append(abs_pl_2_bb(pred_bounding_box, pred_head_pl))

		# if 'gt_part_loc_body' in kwargs and 'gt_bounding_box' in kwargs:
		# 	bounding_boxes_list.append(abs_pl_2_bb(kwargs['gt_bounding_box'], kwargs['gt_part_loc_body']))
		# 	bounding_boxes_list.append(abs_pl_2_bb(pred_bounding_box, pred_body_pl))

		if len(bounding_boxes_list) > 0:
			rgb_bb = draw_bounding_boxes(rgb, bounding_boxes_list)
			tf.summary.image('input_with_gt_bb', rgb_bb, 3)


		log.info('STN VGG model build finished')

		return cls_fc4, prob, pred_bounding_box, pred_head_pl #, pred_body_pl
