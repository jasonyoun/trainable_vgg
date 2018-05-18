import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging as log
import tensorflow as tf
from PIL import Image
from solver import Solver

# suppress warning
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

####################
# global variables #
####################
# data types
DATA_TYPE_CUB = 'cub'
DATA_TYPE_COMPCARS = 'compcars'
DATA_TYPE_IMAGENET = 'imagenet'
# folders
WEIGHTS_FOLDER = 'weights'
LOG_FOLDER = 'log'
NETWORKS_FOLDER = 'networks'
# weight files
WEIGHT_VGG16_IMAGENET = 'vgg16.npy'
WEIGHT_VGG19_IMAGENET = 'vgg19.npy'
# directories
DEFAULT_DATASET_DIR = 'D:\\Jason_Folder\\Yonsei\\Research\\dataset'
DEFAULT_CODE_DIR = 'D:\\Jason_Folder\\Yonsei\\Research\\fine_grained\\code'
LOG_DIR = os.path.join(DEFAULT_CODE_DIR, LOG_FOLDER)
NETWORKS_DIR = os.path.join(DEFAULT_CODE_DIR, NETWORKS_FOLDER)
WEIGHTS_DIR = os.path.join(NETWORKS_DIR, WEIGHTS_FOLDER)

#############
# functions #
#############
def set_logging():
	"""
	Configure logging information.
	"""
	# tf.logging.set_verbosity(tf.logging.INFO)
	log.basicConfig(format='%(levelname)s: %(message)s', level=log.DEBUG)
	log.info('Logging set')

def st_vgg_func(mode='train'):
	with tf.Session() as sess:
		if mode is 'train':
			solver = Solver(
				sess=sess,
				dataset_type='cub',
				dataset_dir=DEFAULT_DATASET_DIR,
				resize=(256, 256),
				crop_shape=(224, 224),
				network_type='st_vgg',
				log_dir=LOG_DIR,
				weights_path=[os.path.join(WEIGHTS_DIR, 'vgg16.npy'),
							  os.path.join(WEIGHTS_DIR, 'vgg16.npy')],
				init_layers=[])

			lr_fast_vars=[
				'localization/conv6/conv6_filters', 'localization/conv6/conv6_biases',
				'localization/fc7/fc7_filters', 'localization/fc7/fc7_biases',
				'localization/fc8/fc8_filters', 'localization/fc8/fc8_biases',
				'classification/fc8/fc8_filters', 'classification/fc8/fc8_biases']

			solver.trainer(
				learning_rate=0.0001,
				epochs=200,
				learning_rate_fast=0.001,
				lr_fast_vars=lr_fast_vars,
				l2_regularization_decay=5e-4,
				save_path=os.path.join(WEIGHTS_DIR, 'st_vgg16_cub_ft_supervised.npy'),
				save_scope=['localization', 'classification'],
				save_epoch=[50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 199])

		elif mode is 'test':
			solver = Solver(
				sess=sess,
				dataset_type='cub',
				dataset_dir=DEFAULT_DATASET_DIR,
				resize=(256, 256),
				crop_shape=(224, 224),
				network_type='st_vgg',
				log_dir=LOG_DIR,
				weights_path=[os.path.join(WEIGHTS_DIR, 'st_vgg16_cub_ft_supervised_localization_epoch190.npy'),
							  os.path.join(WEIGHTS_DIR, 'st_vgg16_cub_ft_supervised_classification_epoch190.npy')],
				init_layers=[])

			solver.tester()

		else:
			log.error('Invalid mode.')

def vgg_cub_ft_func(mode='train'):
	with tf.Session() as sess:
		if mode is 'train':
			solver = Solver(
				sess=sess,
				dataset_type='cub',
				dataset_dir=DEFAULT_DATASET_DIR,
				resize=(256, 256),
				crop_shape=(224, 224),
				network_type='vgg16',
				log_dir=LOG_DIR,
				weights_path=os.path.join(WEIGHTS_DIR, 'vgg16.npy'),
				init_layers=['fc8'])

			solver.trainer(
				learning_rate=0.0001,
				epochs=100,
				learning_rate_fast=0.001,
				lr_fast_vars=['fc8/fc8_filters', 'fc8/fc8_biases'],
				l2_regularization_decay=5e-4,
				save_path=os.path.join(WEIGHTS_DIR, 'vgg16_cub_ft.npy'),
				save_epoch=[60, 70, 80, 90, 99])

		elif mode is 'test':
			solver = Solver(
				sess=sess,
				dataset_type='cub',
				dataset_dir=DEFAULT_DATASET_DIR,
				resize=(256, 256),
				crop_shape=(224, 224),
				network_type='vgg16',
				log_dir=LOG_DIR,
				weights_path=os.path.join(WEIGHTS_DIR, 'vgg16_cub_ft_epoch99.npy'),
				init_layers=[])

			solver.tester()

		else:
			log.error('Invalid mode.')


# main
if __name__ == '__main__':
	# set logging
	set_logging()

	# call appropricate functions
	st_vgg_func(mode='test')
