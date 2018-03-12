import os
import logging as log
import numpy as np
import tensorflow as tf
from PIL import Image
from imgaug import augmenters as iaa
from utils import *

def get_image_mean(list_file, resize=None, isotropical=False):
	"""
	Given a .txt file which consists of list of image files
	in the format (number image_path label), find their mean for each channel.

	Inputs:
		- list_file: .txt file with lists of image files
		- resize: size of the smallest side of for resizing in Python tuple format

	Returns:
		- mean: numpy array of shape (3,) containing the mean
	"""
	# variables
	file_list = []
	mean = np.zeros(3, dtype=np.float64)

	log.info('Calculating mean of images in the list file ' +
		'\'{}\'.'.format(list_file))

	# check if list file exists
	if not file_exists(list_file):
		log.error('File \'{}\' does not exist!'.format(list_file))

	# get lists from file
	with open(list_file, 'r') as text_file:
		for _, line in enumerate(text_file):
			file_list.append(line.split()[1])

	log.debug('Using {} images to get the mean.'.format(len(file_list)))

	# open each image
	for i in range(len(file_list)):
		assert file_exists(file_list[i])
		image = Image.open(file_list[i]).convert('RGB')
		# resize image
		if resize is not None:
			if isotropical is True:
				image_resized = isotropical_resize(image, min(resize), upscale=False)
			else:
				image_resized = image.resize(resize, resample=Image.ANTIALIAS)
		mean += np.mean(np.array(image_resized, dtype=np.float32), axis=(0,1))
		if i%1000 is 0:
			log.debug('Processing {}th image.'.format(i))

	mean /= len(file_list)

	log.info('Calculated mean of {} images: {}'.format(len(file_list), mean))

	return mean

def isotropical_resize(input_img, base, upscale):
	"""
	Resize the input_img isotropically by setting the smaller side
	of the image to the number of pixels given by base.
	If upscale is True and smaller side of the input_img is
	smaller than base, perform upscaling.

	Inputs:
		- input_img: input image of PIL Image format
		- base: base to isotropically resize the input_img
		- upscale: bool. True if wish to perform upscaling


	Retruns:
		- output: isotropically resized output image of PIL Image format
	"""
	# variables
	width, height = input_img.size
	output = None

	# if any dimension of the input image is smaller than the
	# base size and upscale is disabled, just return the input image
	if (upscale is False) and ((height < base) or (width < base)):
		return input_img

	# resize image by either downscaling or upscaling
	if height >= width:
		percent = base / float(width)
		height_new = int((float(height) * float(percent)))
		output = input_img.resize((base, height_new), Image.ANTIALIAS)
	else:
		percent = base / float(height)
		width_new = int((float(width) * float(percent)))
		output = input_img.resize((width_new, base), Image.ANTIALIAS)

	return output

def random_crop(input_img, crop_height, crop_width):
	"""
	Randomly crop the input_img to specified size.

	Inputs:
		- input_img: input image of PIL Image format
		- crop_height: height of the desired crop size
		- crop_width: width of the desired crop size

	Returns:
		- cropped output image of PIL Image format
	"""
	input_width, input_height = input_img.size

	# width start and height start
	ws = int(np.random.choice(input_width - crop_width + 1, 1))
	hs = int(np.random.choice(input_height - crop_height + 1 , 1))

	return input_img.crop((ws, hs, ws+crop_width, hs+crop_height))

def augment_image_batch(
		input_batch,
		flr=None,
		add=None,
		agn=None,
		coarse_dropout=None,
		coarse_sp=None):
	"""
	Augment image batch using imgaug.
	(https://github.com/aleju/imgaug)
	(http://imgaug.readthedocs.io/en/latest/source/augmenters.html#noop)

	Inputs:
		- flr: rate of flipping left and right (ex. 0.5)
		- add: add value to all pixels (ex. (-10, 10))
		- agn: add gaussian noise (ex. 0.05)
		- coarse_dropout: add coarse dropout by certain fraction of pixels to zero (ex. 0.2)
		- coarse_sp: add coarse salt & pepper noise (ex. 0.2)

	Returns:
		- augmented image in numpy data type
	"""

	augment_list = []

	if flr is not None:
		augment_list.append(iaa.Fliplr(flr))

	if add is not None:
		augment_list.append(iaa.Add(add))

	if agn is not None:
		augment_list.append(iaa.AdditiveGaussianNoise(scale=(0, agn*255)))

	if coarse_dropout is not None:
		augment_list.append(iaa.CoarseDropout(coarse_dropout, size_percent=0.5))

	if coarse_sp is not None:
		augment_list.append(iaa.CoarseSaltAndPepper(coarse_sp, size_percent=0.5))

	if not augment_list:
		aug = iaa.Noop()
	else:
		aug = iaa.SomeOf((0,None), augment_list, random_order=True)

	return aug.augment_images(input_batch)
