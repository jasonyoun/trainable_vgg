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

FLAGS = None # parser namespace

#############
# functions #
#############
def parse_argument():
	"""
	Parse input arguments.

	Returns:
		- namespace containing the attributes
	"""
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--data_type',
		type=str,
		default=DATA_TYPE_IMAGENET,
		help='Dataset type (cub / compcars / imagenet)')

	parser.add_argument(
		'--weight_file',
		type=str,
		default=WEIGHT_VGG16_IMAGENET,
		help='Weight file name')

	return parser.parse_args()

def set_logging():
	"""
	Configure logging information.
	"""
	# tf.logging.set_verbosity(tf.logging.INFO)
	log.basicConfig(format='%(levelname)s: %(message)s', level=log.INFO)
	log.info('Logging set')

# main
if __name__ == '__main__':
	# some variables

	# parse arguments & set logging
	FLAGS = parse_argument()
	set_logging()


	# test_img = Image.open('laska.png').convert('RGB').resize((224,224))
	# test_img = np.array(test_img, dtype=np.float32).reshape((1, 224, 224, 3))

	# with tf.Session() as sess:
	# 	solver = Solver(
	# 		sess=sess,
	# 		dataset_type='pedestrian',
	# 		dataset_dir=DEFAULT_DATASET_DIR,
	# 		resize=(256, 256),
	# 		crop_shape=[224, 224],
	# 		network_type='vgg16',
	# 		weights_path=os.path.join(WEIGHTS_DIR, WEIGHT_VGG16_IMAGENET),
	# 		init_layers=['fc8'],
	# 		log_dir=LOG_DIR)

	# 	solver.trainer(updated_weights_path=os.path.join(WEIGHTS_DIR, 'vgg16_pedestrian_ft.npy'))

	with tf.Session() as sess:
		solver = Solver(
			sess=sess,
			dataset_type='pedestrian',
			dataset_dir=DEFAULT_DATASET_DIR,
			resize=(224, 224),
			crop_shape=[224, 224],
			network_type='vgg16',
			weights_path=os.path.join(WEIGHTS_DIR, 'vgg16_pedestrian_ft.npy'),
			init_layers=[],
			log_dir=LOG_DIR)

		solver.tester()
