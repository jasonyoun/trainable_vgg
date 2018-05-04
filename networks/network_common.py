import tensorflow as tf
import numpy as np
import logging as log
from functools import reduce

class NetworkCommon:
	"""
	Common functions for the convolutional network.
	"""

	def __init__(self, init_layers, data_dict, trainable):
		"""
		Constructor for the NetworkCommon class.

		Inputs:
			- init_layers: force the specified layers in this python list to
				truncated_normal instead of loading from the data_dict
			- data_dict: dictionary containing the weights loaded from npy
			- trainable: boolean. True is trainable
		"""
		self.init_layers = init_layers
		self.data_dict = data_dict
		self.trainable = trainable
		self.var_dict = {}

	def avg_pool(self, inputs, name):
		"""
		Generate a average pooling layer.

		Inputs:
			- inputs: input tensor
			- name: name of the layer

		Returns:
			- average pooled tensor
		"""
		return tf.nn.avg_pool(
			inputs,
			ksize=[1, 2, 2, 1],
			strides=[1, 2, 2, 1],
			padding='SAME',
			name=name)

	def max_pool(self, inputs, name, stride=2):
		"""
		Generate a max pooling layer.

		Inputs:
			- inputs: input tensor
			- name: name of the layer

		Returns:
			- max pooled tensor
		"""
		return tf.nn.max_pool(
			inputs,
			ksize=[1, stride, stride, 1],
			strides=[1, stride, stride, 1],
			padding='SAME',
			name=name)

	def conv_layer(self, inputs, filter_size, in_channels, out_channels, name, do_batch_norm=False):
		"""
		Generate a convolutional layer.
		ReLu is also performed by default.

		Inputs:
			- inputs: input tensor
			- filter_size: size of the convolutional filter (filter_size x filter_size)
			- in_channels: number of input channel
			- out_channels: number of output channel
			- name: name of the layer

		Returns:
			- relu: generated conv layer with ReLu
		"""
		with tf.variable_scope(name):
			filt, conv_biases = self._get_conv_var(filter_size, in_channels, out_channels, name)
			conv = tf.nn.conv2d(inputs, filt, [1, 1, 1, 1], padding='SAME')
			bias = tf.nn.bias_add(conv, conv_biases)

			if do_batch_norm is True:
				mean, var = tf.nn.moments(bias, [0, 1, 2])
				bias = tf.nn.batch_normalization(bias, mean, var, None, None, 1e-5)

			relu = tf.nn.relu(bias)

			return relu

	def fc_layer(self, inputs, out_size, is_last, name, do_batch_norm=False, activation_fn=None, **kwargs):
		"""
		Generate a FC layer.
		ReLu is performed depending if the layer is last or not.

		Inputs:
			- inputs: input tensor
			- out_size: number of output size
			- is_last: boolean. True if this is the last layer
			- name: name of the layer
			- kwargs
				- init_weight: numpy array or tf tensor of weight to initialize to.
				- init_bias: numpy array or tf tensor of bias to initialize to

		Returns:
			- post_activation: generated FC layer with/without ReLu
		"""
		init_weight = None
		init_bias = None

		for key, value in kwargs.items():
			if key is 'init_weight':
				init_weight = value
			elif key is 'init_bias':
				init_bias = value
			else:
				raise RuntimeError('No matching argument')

		with tf.variable_scope(name):
			in_size = np.prod(inputs.get_shape().as_list()[1:])
			weights, biases = self._get_fc_var(in_size, out_size, init_weight, init_bias, name)
			x = tf.reshape(inputs, [-1, in_size])
			pre_activation = tf.nn.bias_add(tf.matmul(x, weights), biases)

			if do_batch_norm is True:
				mean, var = tf.nn.moments(pre_activation, [0])
				pre_activation = tf.nn.batch_normalization(pre_activation, mean, var, None, None, 1e-5)

			if is_last is True:
				post_activation = tf.identity(pre_activation)
			else:
				if activation_fn is not None:
					post_activation = activation_fn(pre_activation)
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

	def entry_stop_gradients(self, target, mask):
		mask_h = tf.logical_not(mask)
		mask = tf.cast(mask, dtype=target.dtype)
		mask_h = tf.cast(mask_h, dtype=target.dtype)

		return tf.stop_gradient(mask_h * target) + mask * target

	def _get_conv_var(self, filter_size, in_channels, out_channels, name):
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
		initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.0001)
		filters = self._get_var(initial_value, name, 0, name + "_filters")

		initial_value = tf.zeros([out_channels])
		biases = self._get_var(initial_value, name, 1, name + "_biases")

		return filters, biases

	def _get_fc_var(self, in_size, out_size, init_weight, init_bias, name):
		"""
		Get FC layer variables.

		Inputs:
			- in_size: input size
			- out_size: output size
			- init_weight: numpy array or tf tensor of weight to initialize to.
				set to 'None' if initializing to default normal distribution
			- init_bias: numpy array or tf tensor of bias to initialize to
				set to 'None' if initializing to default normal distribution
			- name: name of the layer

		Returns:
			- weights: weights of the FC layer
			- biases: biases of the FC layer
		"""
		if init_weight is None:
			log.debug('Initializing weight of layer {} to truncated_normal'.format(name))
			initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.0001)
		else:
			log.debug('Initializing weight of layer {} to user defined value'.format(name))
			initial_value = init_weight
		weights = self._get_var(initial_value, name, 0, name + "_filters")

		if init_bias is None:
			log.debug('Initializing bias of layer {} to zeros'.format(name))
			initial_value = tf.zeros([out_size])
		else:
			log.debug('Initializing bias of layer {} to user defined value'.format(name))
			initial_value = init_bias
		biases = self._get_var(initial_value, name, 1, name + "_biases")

		return weights, biases

	def _get_var(self, initial_value, name, idx, var_name):
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
			log.debug('Initializing \'{}\''.format(var_name))
			value = initial_value
		elif self.data_dict is not None and name in self.data_dict:
			log.debug('Loading \'{}\''.format(var_name))
			value = self.data_dict[name][idx]
		else:
			raise RuntimeError('No matching variable')

		if self.trainable:
			var = tf.Variable(value, name=var_name)
		else:
			var = tf.constant(value, dtype=tf.float32, name=var_name)

		self.var_dict[(name, idx)] = var

		return var
