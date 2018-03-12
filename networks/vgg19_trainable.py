import os
import numpy as np
import tensorflow as tf
import logging as log
from networks.vgg_common import VggCommon

class VGG19:
	"""
	Class for the trainable VGG-19 network.
	"""

	def __init__(self, num_classes, mean_rgb, vgg19_npy_path=None, init_layers=None, trainable=True, dropout_rate=0.5):
		"""
		VGG-19 class constructor.

		Inputs:
			- num_classes: number of classes to set the final FC layer output size
			- mean_rgb: python list containing the mean of the dataset in RGB order
			- vgg19_npy_path: path of the vgg19 pre-trained weights in .npy format
			- init_layers: force the specified layers in this python list to
				truncated_normal instead of loading from the data_dict
			- trainable: boolean. True if tf.Variable False if tf.constant
			- dropout_rate: dropout rate
		"""
		if vgg19_npy_path is not None:
			data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
		else:
			data_dict = None

		self.num_classes = num_classes
		self.dropout_rate = dropout_rate
		self.vc = VggCommon(init_layers, data_dict, mean_rgb, trainable)

	def build(self, rgb, train_mode=None):
		"""
		Build the VGG-19 network.

		Inputs:
			- rgb: input image in rgb format in range [0, 255]
			- train_mode: boolean. True if training mode False if testing mode

		Returns:
			- self.fc8: logits output
			- self.prob: probability output
		"""

		log.info('VGG 19 model build started')

		tf.summary.image('input', rgb, 5)

		# Convert RGB to BGR
		bgr = self.vc.rgb_2_bgr(rgb)

		# conv layers
		self.conv1_1 = self.vc.conv_layer(bgr, 3, 64, 'conv1_1')
		self.conv1_2 = self.vc.conv_layer(self.conv1_1, 64, 64, 'conv1_2')
		self.pool1 = self.vc.max_pool(self.conv1_2, 'pool1')

		self.conv2_1 = self.vc.conv_layer(self.pool1, 64, 128, 'conv2_1')
		self.conv2_2 = self.vc.conv_layer(self.conv2_1, 128, 128, 'conv2_2')
		self.pool2 = self.vc.max_pool(self.conv2_2, 'pool2')

		self.conv3_1 = self.vc.conv_layer(self.pool2, 128, 256, 'conv3_1')
		self.conv3_2 = self.vc.conv_layer(self.conv3_1, 256, 256, 'conv3_2')
		self.conv3_3 = self.vc.conv_layer(self.conv3_2, 256, 256, 'conv3_3')
		self.conv3_4 = self.vc.conv_layer(self.conv3_3, 256, 256, 'conv3_4')
		self.pool3 = self.vc.max_pool(self.conv3_4, 'pool3')

		self.conv4_1 = self.vc.conv_layer(self.pool3, 256, 512, 'conv4_1')
		self.conv4_2 = self.vc.conv_layer(self.conv4_1, 512, 512, 'conv4_2')
		self.conv4_3 = self.vc.conv_layer(self.conv4_2, 512, 512, 'conv4_3')
		self.conv4_4 = self.vc.conv_layer(self.conv4_3, 512, 512, 'conv4_4')
		self.pool4 = self.vc.max_pool(self.conv4_4, 'pool4')

		self.conv5_1 = self.vc.conv_layer(self.pool4, 512, 512, 'conv5_1')
		self.conv5_2 = self.vc.conv_layer(self.conv5_1, 512, 512, 'conv5_2')
		self.conv5_3 = self.vc.conv_layer(self.conv5_2, 512, 512, 'conv5_3')
		self.conv5_4 = self.vc.conv_layer(self.conv5_3, 512, 512, 'conv5_4')
		self.pool5 = self.vc.max_pool(self.conv5_4, 'pool5')

		# fc layers
		self.fc6 = self.vc.fc_layer(self.pool5, 25088, 4096, False, 'fc6') # 25088 = ((224 // (2 ** 5)) ** 2) * 512
		self.dropout6 = self.vc.dropout_layer(self.fc6, self.dropout_rate, train_mode, 'dropout6')

		self.fc7 = self.vc.fc_layer(self.dropout6, 4096, 4096, False, 'fc7')
		self.dropout7 = self.vc.dropout_layer(self.fc7, self.dropout_rate, train_mode, 'dropout7')

		self.fc8 = self.vc.fc_layer(self.dropout7, 4096, self.num_classes, True, 'fc8')

		# softmax loss
		self.prob = tf.nn.softmax(self.fc8, name="prob")

		log.info('VGG 19 model build finished')

		return self.fc8, self.prob

	def save_npy(self, sess, npy_path):
		"""
		Save all variables to .npy file.

		Inputs:
			- sess: TensorFlow session
			- npy_path: path for the .npy file to be saved to
		"""
		self.vc.save_npy(sess, npy_path)

	def get_var_count(self):
		"""
		Count the number of variables in the network.
		VGG-19: 143667240 (when number of classes is 1000)
		VGG-19: 138357544 (when number of classes is 1000)

		Returns:
			- count: variable count
		"""
		return self.vc.get_var_count()
