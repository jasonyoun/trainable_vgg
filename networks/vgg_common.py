import tensorflow as tf
import numpy as np
import logging as log
from functools import reduce

class VggCommon:
	"""
	Common functions for the VGG network.
	"""

	def __init__(self, init_layers, data_dict, mean_rgb, trainable):
		"""
		Constructor for the VggCommon class.

		Inputs:
			- init_layers: force the specified layers in this python list to
				truncated_normal instead of loading from the data_dict
			- data_dict: dictionary containing the weights loaded from npy
			- mean_rgb: mean of the dataset in RGB format
			- trainable: boolean. True is trainable
		"""
		self.init_layers = init_layers
		self.data_dict = data_dict
		self.mean_rgb = mean_rgb
		self.trainable = trainable
		self.var_dict = {}

	def rgb_2_bgr(self, rgb):
		"""
		Convert RGB image to BGR image.

		Inputs:
			- rgb: input image in RGB format in range [0, 255]

		Returns:
			- bgr: converted bgr image
		"""
		red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)

		assert red.get_shape().as_list()[1:] == [224, 224, 1]
		assert green.get_shape().as_list()[1:] == [224, 224, 1]
		assert blue.get_shape().as_list()[1:] == [224, 224, 1]

		bgr = tf.concat(
			axis=3,
			values=[blue - self.mean_rgb[2],
					green - self.mean_rgb[1],
					red - self.mean_rgb[0],])

		assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

		return bgr

	def avg_pool(self, value, name):
		"""
		Generate a average pooling layer.

		Inputs:
			- value: input tensor
			- name: name of the layer

		Returns:
			- average pooled tensor
		"""
		return tf.nn.avg_pool(
			value,
			ksize=[1, 2, 2, 1],
			strides=[1, 2, 2, 1],
			padding='SAME',
			name=name)

	def max_pool(self, value, name):
		"""
		Generate a max pooling layer.

		Inputs:
			- value: input tensor
			- name: name of the layer

		Returns:
			- max pooled tensor
		"""
		return tf.nn.max_pool(
			value,
			ksize=[1, 2, 2, 1],
			strides=[1, 2, 2, 1],
			padding='SAME',
			name=name)

	def conv_layer(self, value, in_channels, out_channels, name):
		"""
		Generate a convolutional layer.
		ReLu is also performed by default.

		Inputs:
			- value: input tensor
			- in_channels: number of input channel
			- out_channels: number of output channel
			- name: name of the layer

		Returns:
			- relu: generated conv layer with ReLu
		"""
		with tf.variable_scope(name):
			filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)
			conv = tf.nn.conv2d(value, filt, [1, 1, 1, 1], padding='SAME')
			bias = tf.nn.bias_add(conv, conv_biases)
			relu = tf.nn.relu(bias)

			return relu

	def fc_layer(self, value, in_size, out_size, is_last, name):
		"""
		Generate a FC layer.
		ReLu is performed depending if the layer is last or not.

		Inputs:
			- value: input tensor
			- in_size: number of input size
			- out_size: number of output size
			- is_last: boolean. True if this is the last layer
			- name: name of the layer

		Returns:
			- post_activation: generated FC layer with/without ReLu
		"""
		with tf.variable_scope(name):
			weights, biases = self.get_fc_var(in_size, out_size, name)
			x = tf.reshape(value, [-1, in_size])
			pre_activation = tf.nn.bias_add(tf.matmul(x, weights), biases)

			if is_last is True:
				post_activation = tf.identity(pre_activation)
			else:
				post_activation = tf.nn.relu(pre_activation)

			return post_activation

	def dropout_layer(self, inputs, dropout_rate, train_mode, name):
		"""
		Generate a dropout layer.

		Inputs:
			- inputs: input tensor
			- dropout_rate: dropout rate
			- train_mode: boolean. True if dropout need to be turned on
			- name: name of the layer
		"""
		return tf.layers.dropout(
			inputs,
			rate=dropout_rate,
			training=train_mode,
			name=name)

	def get_conv_var(self, filter_size, in_channels, out_channels, name):
		"""
		Get convolutional layer variables.

		Inputs:
			- filter_size: size of the convolutional filter (filter_size x filter_size)
			- in_channels: number of input channels
			- out_channels: number of output channels
			- name: name of the layer

		Returns:
			- filters: filters of the convolutional filter
			- biases: biases of the convolutional filter
		"""
		initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
		filters = self.get_var(initial_value, name, 0, name + "_filters")

		initial_value = tf.truncated_normal([out_channels], .0, .001)
		biases = self.get_var(initial_value, name, 1, name + "_biases")

		return filters, biases

	def get_fc_var(self, in_size, out_size, name):
		"""
		Get FC layer variables.

		Inputs:
			- in_size: input size
			- out_size: output size
			- name: name of the layer

		Returns:
			- weights: weights of the FC layer
			- biases: biases of the FC layer
		"""
		initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
		weights = self.get_var(initial_value, name, 0, name + "_weights")

		initial_value = tf.truncated_normal([out_size], .0, .001)
		biases = self.get_var(initial_value, name, 1, name + "_biases")

		return weights, biases

	def get_var(self, initial_value, name, idx, var_name):
		"""
		Init / fetch variables for the weights and biases.

		Inputs:
			- initial_value: tensor to init to if name is in self.init_layers
			- name: name of the layer
			- idx: index of the weight
			- var_name: variable name to match with the weight

		Returns:
			- var: initialized / fetched variable
		"""
		if self.init_layers is not None and name in self.init_layers:
			log.info('Initializing \'{}\''.format(var_name))
			value = initial_value
		elif self.data_dict is not None and name in self.data_dict:
			log.info('Loading \'{}\''.format(var_name))
			value = self.data_dict[name][idx]
		else:
			log.error('No matching variable')
			raise RuntimeError('No matching variable')

		if self.trainable:
			var = tf.Variable(value, name=var_name)
		else:
			var = tf.constant(value, dtype=tf.float32, name=var_name)

		self.var_dict[(name, idx)] = var

		assert var.get_shape() == initial_value.get_shape()

		return var

	def save_npy(self, sess, npy_path):
		"""
		Save all variables to .npy file.

		Inputs:
			- sess: TensorFlow session
			- npy_path: path for the .npy file to be saved to
		"""
		assert isinstance(sess, tf.Session)

		data_dict = {}

		for (name, idx), var in list(self.var_dict.items()):
			var_out = sess.run(var)
			if name not in data_dict:
				data_dict[name] = {}
			data_dict[name][idx] = var_out

		np.save(npy_path, data_dict)
		log.info("npy saved to {}".format(npy_path))

	def get_var_count(self):
		"""
		Count the number of variables in the network.
		VGG-19: 143667240 (when number of classes is 1000)
		VGG-16: 138357544 (when number of classes is 1000)

		Returns:
			- count: variable count
		"""
		count = 0

		for v in list(self.var_dict.values()):
			count += reduce(lambda x, y: x * y, v.get_shape().as_list())

		return count
