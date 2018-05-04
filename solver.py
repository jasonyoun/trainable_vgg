import os
import re
import numpy as np
import tensorflow as tf
import logging as log
from parsedata.parse_imagenet import ParseImageNet
from parsedata.parse_cub import ParseCub
from networks.vgg16 import VGG16
from networks.vgg19 import VGG19
from networks.st_vgg import ST_VGG
from functools import reduce
import utils

class Solver():
	_TRAIN_LOG_FOLDER = 'train'
	_VAL_LOG_FOLDER = 'val'
	_TEST_LOG_FOLDER = 'test'
	_BATCH_SIZE = 32

	def __init__(
			self,
			sess,
			dataset_type,
			dataset_dir,
			resize,
			crop_shape,
			network_type,
			log_dir,
			weights_path=None,
			init_layers=None):

		self.sess = sess

		if dataset_type is 'cub':
			self.dataset = ParseCub(
				dataset_dir=dataset_dir,
				resize=resize,
				crop_shape=crop_shape,
				batch_size=self._BATCH_SIZE,
				isotropical=True,
				initial_load=False)
		elif dataset_type is 'imagenet':
			self.dataset = ParseImageNet(batch_size=self._BATCH_SIZE)

		if network_type is 'vgg16':
			self.network = VGG16(
				num_classes=self.dataset.get_num_classes(),
				vgg16_npy_path=weights_path,
				init_layers=init_layers)
		elif network_type is 'vgg19':
			self.network = VGG19(
				num_classes=self.dataset.get_num_classes(),
				vgg19_npy_path=weights_path,
				init_layers=init_layers)
		elif network_type is 'st_vgg':
			self.network = ST_VGG(
				num_classes=self.dataset.get_num_classes(),
				loc_vgg_npy_path=weights_path[0],
				cls_vgg_npy_path=weights_path[1])

		self.train_log_path = os.path.join(log_dir, self._TRAIN_LOG_FOLDER)
		self.val_log_path = os.path.join(log_dir, self._VAL_LOG_FOLDER)
		self.test_log_path = os.path.join(log_dir, self._TEST_LOG_FOLDER)

		# tf placeholders
		self.images = tf.placeholder(tf.float32, (None, crop_shape[0], crop_shape[1], 3))
		self.true_out = tf.placeholder(tf.float32, (None, self.dataset.get_num_classes()))
		self.train_mode = tf.placeholder(tf.bool)
		self.learning_rate = tf.placeholder(tf.float32)
		self.learning_rate_fast = tf.placeholder(tf.float32)

		# build the network
		self.logits, self.prob = self.network.build(self.images, self.train_mode)

		log.debug('Network variables count: {}'.format(self._get_var_count()))

	def trainer(
			self,
			learning_rate=0.001,
			epochs=100,
			learning_rate_fast=None,
			lr_fast_vars=None,
			l2_regularization_decay=0,
			save_path=None,
			save_scope=[],
			save_epoch=[]):
		# variables
		iteration = 0
		is_last = False

		# trainable variables
		trainable_vars = [v for v in tf.trainable_variables()]

		log.debug('trainable variables...')
		for v in trainable_vars:
			log.debug('  {} {}'.format(v.name, v.get_shape().as_list()))

		# L2 loss
		if l2_regularization_decay > 0:
			regularization_vars = [v for v in trainable_vars if 'bias' not in v.name]
			loss_L2 = tf.add_n([tf.nn.l2_loss(v) for v in regularization_vars]) * l2_regularization_decay
			log.debug('applying L2 regularization to...')
			for v in regularization_vars:
				log.debug('  {} {}'.format(v.name, v.get_shape().as_list()))

		# define loss
		total_loss = tf.losses.softmax_cross_entropy(self.true_out, self.logits)
		if l2_regularization_decay > 0:
			mean_loss = tf.reduce_mean(total_loss) + loss_L2
		else:
			mean_loss = tf.reduce_mean(total_loss)
		tf.summary.scalar('mean_loss', mean_loss)

		# accuracy
		correct_prediction = tf.equal(tf.argmax(self.prob, 1), tf.argmax(self.true_out, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)

		# define optimizer
		if learning_rate_fast is not None and lr_fast_vars is not None: # layer-wise optimizer
			fast_lr_vars = [v for v in trainable_vars if v.name.split(':')[0] in lr_fast_vars]
			normal_lr_vars = [v for v in trainable_vars if v not in fast_lr_vars]

			log.debug('variables to train with fast learning rate')
			for v in fast_lr_vars:
				log.debug('  {} {}'.format(v.name, v.get_shape().as_list()))

			log.debug('variables to train with normal learning rate')
			for v in normal_lr_vars:
				log.debug('  {} {}'.format(v.name, v.get_shape().as_list()))

			# optimizer
			opt_fast = tf.train.GradientDescentOptimizer(self.learning_rate_fast)
			opt_normal = tf.train.GradientDescentOptimizer(self.learning_rate)

			# gradients
			grads = tf.gradients(mean_loss, fast_lr_vars + normal_lr_vars)
			grads_fast = grads[:len(fast_lr_vars)]
			grads_normal = grads[len(fast_lr_vars):]

			# group optimizers with different lr
			train_op_fast = opt_fast.apply_gradients(zip(grads_fast, fast_lr_vars))
			train_op_normal = opt_normal.apply_gradients(zip(grads_normal, normal_lr_vars))
			train_op = tf.group(train_op_fast, train_op_normal)

			tf.summary.scalar('learning_rate_fast', self.learning_rate_fast)
			tf.summary.scalar('learning_rate', self.learning_rate)
		else: # normal optimizer
			opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			train_op = opt.minimize(mean_loss)

			tf.summary.scalar('learning_rate', self.learning_rate)

		# summaries
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(self.train_log_path, self.sess.graph)
		val_writer = tf.summary.FileWriter(self.val_log_path, self.sess.graph)

		# init variables
		self.sess.run(tf.global_variables_initializer())

		for e in range(epochs):
			log.info('Current epoch: {}'.format(e))

			# weight decay if necessary
			# if e is 60:
			#  	learning_rate_fast /= 2
			#  	learning_rate /= 2

			# if e is 70:
			#  	learning_rate_fast /= 2
			#  	learning_rate /= 2

			while is_last is False:
				# get next train batch
				train_batch, train_label, is_last = self.dataset.get_next_train_batch(augment=True)

				# create feed dictionary
				feed_dict = {self.images: train_batch,
							 self.true_out: train_label,
							 self.train_mode: True,
							 self.learning_rate: learning_rate,
							 self.learning_rate_fast: learning_rate_fast}

				# run train op
				self.sess.run(train_op, feed_dict=feed_dict)

				if iteration%10 == 0:# and iteration != 0:
					# training summaries
					feed_dict = {self.images: train_batch,
								 self.true_out: train_label,
								 self.train_mode: False,
								 self.learning_rate: learning_rate,
								 self.learning_rate_fast: learning_rate_fast}

					train_summary, train_acc, theta = self.sess.run(
						[merged, accuracy, self.network.affine], feed_dict=feed_dict)

					train_writer.add_summary(train_summary, iteration)
					log.debug(theta[0])
					log.debug(theta[1])
					log.debug(theta[2])


					# validation summaries
					val_batch, val_label, _ = self.dataset.get_next_val_batch()

					# create validation feed dictionary
					feed_dict = {self.images: val_batch,
								 self.true_out: val_label,
								 self.train_mode: False,
								 self.learning_rate: learning_rate,
								 self.learning_rate_fast: learning_rate_fast}

					val_summary, val_acc = self.sess.run(
						[merged, accuracy], feed_dict=feed_dict)

					val_writer.add_summary(val_summary, iteration)

					log.info('iteration {}. train accuracy: {:.3f} / val accuracy: {:.3f}'
						.format(iteration, train_acc, val_acc))

				iteration +=1

			is_last = False

			# save weights to file
			if e in save_epoch:
				assert save_path is not None
				name, ext = os.path.splitext(save_path)

				if save_scope:
					for ss in save_scope:
						self.save_npy('{}_{}_epoch{}{}'.format(name, ss, e, ext), ss)
				else:
					self.save_npy('{}_epoch{}{}'.format(name, e, ext), None)

		train_writer.close()
		val_writer.close()

	def tester(self):
		# variables
		iteration = 0
		is_last = False
		sum_accuracy = []

		# accuracy
		correct_prediction = tf.equal(tf.argmax(self.prob, 1), tf.argmax(self.true_out, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)

		# summaries
		merged = tf.summary.merge_all()
		test_writer = tf.summary.FileWriter(self.test_log_path, self.sess.graph)

		# init variables
		self.sess.run(tf.global_variables_initializer())

		while is_last is False:
			# get next batch
			batch, label, is_last = self.dataset.get_next_test_batch()

			# create feed dictionary
			feed_dict = {self.images: batch,
						 self.true_out: label,
						 self.train_mode: False}

			temp_accuracy, summary, theta = self.sess.run([accuracy, merged, self.network.affine], feed_dict=feed_dict)
			sum_accuracy.append(temp_accuracy)

			test_writer.add_summary(summary, iteration)
			log.info('iteration {} accuracy: {}'.format(iteration, temp_accuracy))
			log.debug(theta[0])
			log.debug(theta[1])
			log.debug(theta[2])

			iteration +=1

		test_writer.close()

		# final accuracy
		log.info('final accuracy over the test set: {}'.format(np.mean(sum_accuracy)))

	def predictor(self, image):
		feed_dict={self.images: image, self.train_mode: False}
		prob = self.sess.run(self.prob, feed_dict=feed_dict)
		utils.print_prob(prob[0], './synset.txt')

	def save_npy(self, save_path, scope=None):
		"""
		Save all network weights and biases of the current session to .npy format.

		Inputs:
			- save_path: path for the .npy file to be saved to
		"""
		# some variables
		data_dict = {}

		log.info("start saving npy to {}".format(save_path))

		for tensor in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope):
			log.debug('tensor name: {}'.format(tensor.name))

			split_name = re.split('/|:', tensor.name)

			layer_name = split_name[-3]
			var_name = split_name[-2]

			if 'filters' in var_name:
				idx = 0
			elif 'biases' in var_name:
				idx = 1
			else:
				raise RuntimeError('No matching argument')

			log.debug('processing {}'.format(var_name))

			var_out = self.sess.run(self.sess.graph.get_tensor_by_name(tensor.name))

			if layer_name not in data_dict:
				data_dict[layer_name] = {}

			data_dict[layer_name][idx] = var_out

		np.save(save_path, data_dict)

		log.info("finished saving npy to {}".format(save_path))

	def _get_var_count(self):
		"""
		Count the number of variables in the network.
		VGG-19: 143667240 (when number of classes is 1000)
		VGG-16: 138357544 (when number of classes is 1000)

		Returns:
			- count: variable count
		"""
		# some variables
		count = 0

		for tensor in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
			v = self.sess.graph.get_tensor_by_name(tensor.name)
			count += reduce(lambda x, y: x * y, v.get_shape().as_list())

		return count
