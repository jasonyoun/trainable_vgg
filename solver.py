import os
import numpy as np
import tensorflow as tf
import logging as log
from parsedata.parse_imagenet import ParseImageNet
from parsedata.parse_cub import ParseCub
from parsedata.parse_pedestrian import ParsePedestrian
from networks.vgg16_trainable import VGG16
from networks.vgg19_trainable import VGG19
import utils

class Solver():
	_LEARNING_RATE = 0.001
	_TRAIN_LOG_FOLDER = 'train'
	_TEST_LOG_FOLDER = 'test'
	_BATCH_SIZE = 70

	def __init__(
			self,
			sess,
			dataset_type,
			dataset_dir,
			resize,
			crop_shape,
			network_type,
			weights_path,
			init_layers,
			log_dir):

		self.sess = sess

		if dataset_type is 'cub':
			self.dataset = ParseCub(
				dataset_dir=dataset_dir,
				resize=resize,
				crop_shape=crop_shape,
				batch_size=self._BATCH_SIZE,
				initial_load=False)
		elif dataset_type is 'pedestrian':
			self.dataset = ParsePedestrian(
				dataset_dir=dataset_dir,
				resize=resize,
				crop_shape=crop_shape,
				batch_size=self._BATCH_SIZE,
				initial_load=False)
		elif dataset_type is 'imagenet':
			self.dataset = ParseImageNet(batch_size=self._BATCH_SIZE)

		if network_type is 'vgg16':
			self.network = VGG16(
				num_classes=self.dataset.get_num_classes(),
				mean_rgb=self.dataset.get_mean(),
				vgg16_npy_path=weights_path,
				init_layers=init_layers)
		elif network_type is 'vgg19':
			self.network = VGG19(
				num_classes=self.dataset.get_num_classes(),
				mean_rgb=self.dataset.get_mean(),
				vgg19_npy_path=weights_path,
				init_layers=init_layers)

		self.train_log_path = os.path.join(log_dir, self._TRAIN_LOG_FOLDER)
		self.test_log_path = os.path.join(log_dir, self._TEST_LOG_FOLDER)

		# tf placeholders
		self.images = tf.placeholder(tf.float32, (None, crop_shape[0], crop_shape[1], 3))
		self.true_out = tf.placeholder(tf.float32, (None, self.dataset.get_num_classes()))
		self.train_mode = tf.placeholder(tf.bool)
		self.learning_rate = tf.placeholder(tf.float32)

		# build the network
		self.logits, self.prob = self.network.build(self.images, self.train_mode)

		log.debug('Network variables count: {}'.format(self.network.get_var_count()))

		# init variables
		self.sess.run(tf.global_variables_initializer())

	def trainer(self, epochs=30, updated_weights_path=None):
		# variables
		iteration = 0
		is_last = False

		# define loss & optimizer
		total_loss = tf.losses.softmax_cross_entropy(self.true_out, self.logits)
		mean_loss = tf.reduce_mean(total_loss)
		tf.summary.scalar('mean_loss', mean_loss)

		optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
		tf.summary.scalar('learning_rate', self.learning_rate)

		# accuracy
		correct_prediction = tf.equal(tf.argmax(self.prob, 1), tf.argmax(self.true_out, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)

		# trainable variables
		vars_to_optimize = [v for v in tf.trainable_variables()]

		log.debug('trainable variables...')
		for v in vars_to_optimize:
			log.debug('{} {}'.format(v.name, v.get_shape().as_list()))

		# variables to optimize
		# vars_to_optimize = [v for v in tf.trainable_variables() if v.name.split('/')[0] in self.train_layer]

		# log.debug('variables to optimize...')
		# for v in vars_to_optimize:
		# 	log.debug('{} {}'.format(v.name, v.get_shape().as_list()))

		# train op
		train_op = optimizer.minimize(mean_loss)

		# summaries
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(self.train_log_path, self.sess.graph)

		for e in range(epochs):
			log.info('Current epoch: {}'.format(e))

			if e%15 is 0 and e is not 0:
				self._LEARNING_RATE = self._LEARNING_RATE / 10
				log.info('Learning rate updated to {}'.format(self._LEARNING_RATE))

			while is_last is False:
				# get next batch
				batch, label, is_last = self.dataset.get_next_train_batch(
					augment=True, shuffle_after_each_epoch=True)

				# create feed dictionary
				feed_dict = {self.images: batch,
							 self.true_out: label,
							 self.train_mode: True,
							 self.learning_rate: self._LEARNING_RATE}

				_, summary = self.sess.run([train_op, merged], feed_dict=feed_dict)

				if iteration%10 == 0:
					train_writer.add_summary(summary, iteration)
					log.info('iteration {}'.format(iteration))

				iteration +=1

			is_last = False

		train_writer.close()

		# save the weights
		if updated_weights_path is not None:
			self.network.save_npy(self.sess, updated_weights_path)

	def tester(self):
		# variables
		iteration = 0
		is_last = False
		sum_accuracy = np.zeros(1, dtype=np.float64)

		# accuracy
		correct_prediction = tf.equal(tf.argmax(self.prob, 1), tf.argmax(self.true_out, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)

		# summaries
		merged = tf.summary.merge_all()
		test_writer = tf.summary.FileWriter(self.test_log_path, self.sess.graph)

		while is_last is False:
			# get next batch
			batch, label, is_last = self.dataset.get_next_test_batch()

			# create feed dictionary
			feed_dict = {self.images: batch,
						 self.true_out: label,
						 self.train_mode: False}

			temp_accuracy, summary = self.sess.run([accuracy, merged], feed_dict=feed_dict)
			sum_accuracy += temp_accuracy.astype(sum_accuracy.dtype)

			test_writer.add_summary(summary, iteration)
			log.info('iteration {} accuracy: {}'.format(iteration, temp_accuracy))

			iteration +=1

		test_writer.close()

		# final accuracy
		log.info('final accuracy over the test set: {}'.format(sum_accuracy / iteration))

	def predictor(self, image):
		feed_dict={self.images: image, self.train_mode: False}
		prob = self.sess.run(self.network.prob, feed_dict=feed_dict)
		utils.print_prob(prob[0], './synset.txt')
