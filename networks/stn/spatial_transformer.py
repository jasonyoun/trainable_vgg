# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from six.moves import xrange
import tensorflow as tf


def transformer(U, theta, out_size, name='SpatialTransformer', **kwargs):
	"""Spatial Transformer Layer
	Implements a spatial transformer layer as described in [1]_.
	Based on [2]_ and edited by David Dao for Tensorflow.
	Parameters
	----------
	U : float
		The output of a convolutional net should have the
		shape [num_batch, height, width, num_channels].
	theta: float
		The output of the
		localisation network should be [num_batch, 6].
	out_size: tuple of two ints
		The size of the output of the network (height, width)
	References
	----------
	.. [1]  Spatial Transformer Networks
			Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
			Submitted on 5 Jun 2015
	.. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
	Notes
	-----
	To initialize the network to the identity transform init
	``theta`` to :
		identity = np.array([[1., 0., 0.],
							 [0., 1., 0.]])
		identity = identity.flatten()
		theta = tf.Variable(initial_value=identity)
	"""

	def _repeat(x, n_repeats):
		with tf.variable_scope('_repeat'):
			rep = tf.transpose(
				tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
			rep = tf.cast(rep, 'int32')
			x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
			return tf.reshape(x, [-1])

	def _interpolate(im, x, y, out_size):
		with tf.variable_scope('_interpolate'):
			# constants
			num_batch = tf.shape(im)[0]
			height = tf.shape(im)[1]
			width = tf.shape(im)[2]
			channels = tf.shape(im)[3]

			x = tf.cast(x, 'float32')
			y = tf.cast(y, 'float32')
			height_f = tf.cast(height, 'float32')
			width_f = tf.cast(width, 'float32')
			out_height = out_size[0]
			out_width = out_size[1]
			zero = tf.zeros([], dtype='int32')
			max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
			max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

			# scale indices from [-1, 1] to [0, width/height]
			x = (x + 1.0)*(width_f) / 2.0
			y = (y + 1.0)*(height_f) / 2.0

			# do sampling
			x0 = tf.cast(tf.floor(x), 'int32')
			x1 = x0 + 1
			y0 = tf.cast(tf.floor(y), 'int32')
			y1 = y0 + 1

			x0 = tf.clip_by_value(x0, zero, max_x)
			x1 = tf.clip_by_value(x1, zero, max_x)
			y0 = tf.clip_by_value(y0, zero, max_y)
			y1 = tf.clip_by_value(y1, zero, max_y)
			dim2 = width
			dim1 = width*height
			base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
			base_y0 = base + y0*dim2
			base_y1 = base + y1*dim2
			idx_a = base_y0 + x0
			idx_b = base_y1 + x0
			idx_c = base_y0 + x1
			idx_d = base_y1 + x1

			# use indices to lookup pixels in the flat image and restore
			# channels dim
			im_flat = tf.reshape(im, tf.stack([-1, channels]))
			im_flat = tf.cast(im_flat, 'float32')
			Ia = tf.gather(im_flat, idx_a)
			Ib = tf.gather(im_flat, idx_b)
			Ic = tf.gather(im_flat, idx_c)
			Id = tf.gather(im_flat, idx_d)

			# and finally calculate interpolated values
			x0_f = tf.cast(x0, 'float32')
			x1_f = tf.cast(x1, 'float32')
			y0_f = tf.cast(y0, 'float32')
			y1_f = tf.cast(y1, 'float32')
			wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
			wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
			wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
			wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
			output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
			return output

	def _meshgrid(height, width):
		with tf.variable_scope('_meshgrid'):
			# This should be equivalent to:
			#  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
			#                         np.linspace(-1, 1, height))
			#  ones = np.ones(np.prod(x_t.shape))
			#  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
			x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
							tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
			y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
							tf.ones(shape=tf.stack([1, width])))

			x_t_flat = tf.reshape(x_t, (1, -1))
			y_t_flat = tf.reshape(y_t, (1, -1))

			ones = tf.ones_like(x_t_flat)
			grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
			return grid

	def _transform(theta, input_dim, out_size):
		with tf.variable_scope('_transform'):
			num_batch = tf.shape(input_dim)[0]
			height = tf.shape(input_dim)[1]
			width = tf.shape(input_dim)[2]
			num_channels = tf.shape(input_dim)[3]
			theta = tf.reshape(theta, (-1, 2, 3))
			theta = tf.cast(theta, 'float32')

			# grid of (x_t, y_t, 1), eq (1) in ref [1]
			height_f = tf.cast(height, 'float32')
			width_f = tf.cast(width, 'float32')
			out_height = out_size[0]
			out_width = out_size[1]
			grid = _meshgrid(out_height, out_width)
			grid = tf.expand_dims(grid, 0)
			grid = tf.reshape(grid, [-1])
			grid = tf.tile(grid, tf.stack([num_batch]))
			grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

			# Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
			T_g = tf.matmul(theta, grid)
			x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
			y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
			x_s_flat = tf.reshape(x_s, [-1])
			y_s_flat = tf.reshape(y_s, [-1])

			input_transformed = _interpolate(
				input_dim, x_s_flat, y_s_flat,
				out_size)

			output = tf.reshape(
				input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))

			return output

	with tf.variable_scope(name):
		output = _transform(theta, U, out_size)
		return output


def batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer'):
	"""Batch Spatial Transformer Layer
	Parameters
	----------
	U : float
		tensor of inputs [num_batch,height,width,num_channels]
	thetas : float
		a set of transformations for each input [num_batch,num_transforms,6]
	out_size : int
		the size of the output [out_height,out_width]
	Returns: float
		Tensor of size [num_batch*num_transforms,out_height,out_width,num_channels]
	"""
	with tf.variable_scope(name):
		num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
		indices = [[i]*num_transforms for i in xrange(num_batch)]
		input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
		return transformer(input_repeated, thetas, out_size)


def transcale_2_bb(ts_matrix):
	"""
	Convert translation / scale information to bounding box format.
	Opposite of bb_2_transcale().

	Inputs:
		- ts_matrix: tensorflow array of shape (batch, 4).
			each row is composed of translation / scale info
			(sx, tx, sy, ty) where s* denotes scale info between
			0 and 1, and t* denotes translation info between -1 and 1

	Returns:
		- bounding_boxes: tensorflow placeholder for bounding boxes
			information corresponding to the images
			(ex. (x, y, w, h) relative to width and height of image)
	"""
	w, tx, h, ty = tf.split(ts_matrix, 4, axis=1)

	x = (1 - w + tx) / 2
	y = (1 - h + ty) / 2

	bounding_boxes = tf.concat([x, y, w, h], axis=1)

	return bounding_boxes

def bb_2_transcale(bounding_boxes):
	"""
	Convert bounding box format to translation / scale information.
	Opposite of transcale_2_bb().

	Inputs:
		- bounding_boxes: tensorflow placeholder for bounding boxes
			information corresponding to the images
			(ex. (x, y, w, h) relative to width and height of image)

	Returns:
		- ts_matrix: tensorflow array of shape (batch, 4).
			each row is composed of translation / scale info
			(sx, tx, sy, ty) where s* denotes scale info between
			0 and 1, and t* denotes translation info between -1 and 1
	"""
	x, y, w, h = tf.split(bounding_boxes, 4, axis=1)

	tx = 2*x + w - 1
	ty = 2*y + h - 1

	ts_matrix = tf.concat([w, tx, h, ty], axis=1)

	return ts_matrix

def clip_transcale(ts_matrix):
	"""
	If the translation / scale information is outside the range,
	clip it so that we are not sampling from outside of the input
	feature map.

	Inputs:
		- ts_matrix: tensorflow array of shape (batch, 4).
			each row is composed of translation / scale info
			(sx, tx, sy, ty) where s* denotes scale info between
			0 and 1, and t* denotes translation info between -1 and 1

	Returns:
		- clipped_ts: same format as input but clipped
	"""
	bounding_boxes = transcale_2_bb(ts_matrix)

	x, y, w, h = tf.split(bounding_boxes, 4, axis=1)

	x_new = tf.where(x < 0, tf.zeros_like(x), x)
	y_new = tf.where(y < 0, tf.zeros_like(y), y)

	w_new = tf.where(x < 0, (w + x), w)
	h_new = tf.where(y < 0, (h + y), h)

	w_new = tf.where(x_new+w_new>1, (1 - x_new - 0.015), w_new)
	h_new = tf.where(y_new+h_new>1, (1 - y_new - 0.015), h_new)

	clipped_bb = tf.concat([x_new, y_new, w_new, h_new], axis=1)

	clipped_ts = bb_2_transcale(clipped_bb)

	return clipped_ts

def transcale_2_affine(ts_matrix):
	"""
	Convert the translation / scale information to affine matrix format.

	Inputs:
		- ts_matrix: tensorflow array of shape (batch, 4).
			each row is composed of translation / scale info
			(sx, tx, sy, ty) where s* denotes scale info between
			0 and 1, and t* denotes translation info between -1 and 1

	Returns:
		- affine matrix of shape (batch, 6) constructed using the input
	"""
	w, tx, h, ty = tf.split(ts_matrix, 4)

	return tf.concat([w, [0], tx, [0], h, ty], axis=0)

def trans_2_affine(elems):
	"""
	Convert the translation information to affine matrix format.
	* For now, fix scale to 0.33.

	Inputs:
		- elems[0]: tensorflow array of shape (batch, 4).
			each row is composed of translation / scale info
			(sx, tx, sy, ty) where s* denotes scale info between
			0 and 1, and t* denotes translation info between -1 and 1
		- elems[1]: tensorflow array of shape (batch, 2).
			each row is composed of translation info
			(tx, ty) which denotes translation info between -1 and 1

	Returns:
		- affine matrix of shape (batch, 6) constructed using the input
	"""
	w_bb, _, h_bb, _ = tf.split(elems[0], 4)
	tx, ty = tf.split(elems[1], 2)

	scalar = 0.33 * tf.maximum(w_bb, h_bb)

	tmp_ts_matrix = tf.concat([scalar, tx, scalar, ty], axis=0)
	tmp_ts_matrix = clip_transcale(tf.expand_dims(tmp_ts_matrix, 0))

	w, tx, h, ty = tf.split(tf.squeeze(tmp_ts_matrix, 0), 4)

	return tf.concat([w, [0], tx, [0], h, ty], axis=0)

def rel_trans_2_abs_pl(bounding_boxes, t_matrix):
	"""
	Convert translation info relative to the bounding box
	to an absolute part location which is relative to
	the original image.

	Inputs:
		- bounding_boxes: tensorflow placeholder for bounding boxes
			information corresponding to the images
			(ex. (x, y, w, h) relative to width and height of image)
		- t_matrix: tensorflow array of shape (batch, 2).
			each row is composed of translation info
			(tx, ty) which denotes translation info between -1 and 1

	Returns:
		- tf array of size (batch, 2) containing absolute part location
	"""
	x, y, w, h = tf.split(bounding_boxes, 4, axis=1)
	tx, ty = tf.split(t_matrix, 2, axis=1)

	x_center = x + (tx+1)*w/2
	y_center = y + (ty+1)*h/2

	return tf.concat([x_center, y_center], axis=1)

def abs_pl_2_bb(bounding_boxes, abs_pl):
	"""
	Convert absolute part location to bounding box format using
	pre-defined width and height.

	Inputs:
		- bounding_boxes: tensorflow placeholder for bounding boxes
			information corresponding to the images
			(ex. (x, y, w, h) relative to width and height of image)
		- abs_pl: tf array of size (batch, 2) containing
			absolute part location

	Returns:
		- bounding box locations converted from absolute part location
	"""
	x, y, w, h = tf.split(bounding_boxes, 4, axis=1)
	x_center, y_center = tf.split(abs_pl, 2, axis=1)

	# abs_w = w * 0.33
	# abs_h = h * 0.33
	abs_w = tf.ones_like(w) * 0.33 * tf.maximum(w, h)
	abs_h = tf.ones_like(h) * 0.33 * tf.maximum(w, h)

	x_ul = x_center - (abs_w / 2)
	y_ul = y_center - (abs_h / 2)

	return tf.concat([x_ul, y_ul, abs_w, abs_h], axis=1)
