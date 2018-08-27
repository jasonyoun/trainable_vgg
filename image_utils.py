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

def isotropical_resize(input_img, base, upscale, bounding_box=None, part_locs=None):
	"""
	Resize the input_img isotropically by setting the smaller side
	of the image to the number of pixels given by base.
	If upscale is True and smaller side of the input_img is
	smaller than base, perform upscaling.

	Inputs:
		- input_img: input image of PIL Image format
		- base: base to isotropically resize the input_img
		- upscale: bool. True if wish to perform upscaling
		- bounding_box: (optional) bounding box info in tuple format(x, y, w, h)
		- part_locs: (optional) part locations info in dictionary format

	Retruns:
		- output: isotropically resized output image of PIL Image format
		- modified_bb: (optional) modified bounding box in tuple format (x, y, w, h)
		- modified_pl: (optional) modified part locations in dictionary format
	"""
	# variables
	width, height = input_img.size

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

	# modify bouding box if needed
	if bounding_box is not None:
		modified_bb = tuple([percent*x for x in bounding_box])

	# modify part locations if needed
	if part_locs is not None:
		modified_pl = {}
		for key, value in part_locs.items():
			x, y = value
			modified_pl[key] = (percent*x, percent*y)

	# return accordingly
	if bounding_box is not None and part_locs is not None:
		return output, modified_bb, modified_pl
	elif bounding_box is not None:
		return output, modified_bb
	elif part_locs is not None:
		return output, modified_pl
	else:
		return output

def resize_image(input_img, shape, bounding_box=None, part_locs=None):
	"""
	Resize the input_image to the given shape.
	For isotropical resizing, please refer to isotropical_resize().

	Inputs:
		- input_img: input image of PIL Image format
		- shape: python tuple format denoting the resize shape (w, h)
		- bounding_box: (optional) bounding box info in tuple format(x, y, w, h)
		- part_locs: (optional) part locations info in dictionary format

	Returns:
		- output: resized output image of PIL Image format
		- modified_bb: (optional) modified bounding box in tuple format (x, y, w, h)
		- modified_pl: (optional) modified part locations in dictionary format
	"""
	output = input_img.resize(shape, resample=Image.ANTIALIAS)

	width, height = input_img.size
	x_percent = shape[0] / float(width)
	y_percent = shape[1] / float(height)

	# modify bouding box if needed
	if bounding_box is not None:
		x, y, w, h = bounding_box

		x *= x_percent
		w *= x_percent
		y *= y_percent
		h *= y_percent

		modified_bb = (x, y, w, h)

	# modify part locations if needed
	if part_locs is not None:
		modified_pl = {}
		for key, value in part_locs.items():
			x, y = value
			modified_pl[key] = (x_percent*x, y_percent*y)

	# return accordingly
	if bounding_box is not None and part_locs is not None:
		return output, modified_bb, modified_pl
	elif bounding_box is not None:
		return output, modified_bb
	elif part_locs is not None:
		return output, modified_pl
	else:
		return output

def _modify_crop_bb(crop_width, crop_height, ws, hs, bounding_box):
	"""
	(private) Modify the bounding box according to the cropping info.

	Inputs:
		- crop_width: width of the cropped image
		- crop_height: height of the cropped image
		- ws: starting point of the width in the cropped image (pixels)
		- hs: starting point of the height in the cropped image (pixels)
		- bounding_box: bounding box info in tuple format(x, y, w, h)

	Returns:
		- python tuple of the modified bounding box
	"""
	# modify bounding box
	x, y, w, h = bounding_box
	x_mod = x - ws
	y_mod = y - hs

	if (x_mod+w) > crop_width:
		w = ws + crop_width - x

	if (y_mod+h) > crop_height:
		h = hs + crop_height - y

	# clip the bounding box if outside image
	if x < ws:
		x_diff = ws - x
		x_mod = 0
		w = w - x_diff

	if y < hs:
		y_diff = hs - y
		y_mod = 0
		h = h - y_diff

	return (x_mod, y_mod, w, h)

def _modify_crop_pl(crop_width, crop_height, ws, hs, part_locs):
	"""
	(private) Modify the part locations according to the cropping info.

	Inputs:
		- crop_width: width of the cropped image
		- crop_height: height of the cropped image
		- ws: starting point of the width in the cropped image (pixels)
		- hs: starting point of the height in the cropped image (pixels)
		- part_locs: part locations info in dictionary format

	Returns:
		- dictionary of the modified part locations
	"""
	modified_pl = {}

	for key, value in part_locs.items():
		x, y = value
		modified_pl[key] = (x-ws, y-hs)

	return modified_pl

def random_crop(input_img, crop_width, crop_height, bounding_box=None, part_locs=None):
	"""
	Randomly crop the input_img to specified size.
	Adjust the bounding box according to the cropped image if specified.

	Inputs:
		- input_img: input image of PIL Image format
		- crop_width: width of the desired crop size
		- crop_height: height of the desired crop size
		- bounding_box: (optional) bounding box info in tuple format(x, y, w, h)
		- part_locs: (optional) part locations info in dictionary format

	Returns:
		- cropped_img: cropped output image of PIL Image format
		- modified_bb: (optional) modified bounding box in tuple format (x, y, w, h)
		- modified_pl: (optional) modified part locations in dictionary format
	"""
	input_width, input_height = input_img.size

	# width start and height start
	ws = int(np.random.choice(input_width - crop_width + 1, 1))
	hs = int(np.random.choice(input_height - crop_height + 1 , 1))

	# crop image
	cropped_img = input_img.crop((ws, hs, ws+crop_width, hs+crop_height))

	# modify bouding box if needed
	if bounding_box is not None:
		modified_bb = _modify_crop_bb(crop_width, crop_height, ws, hs, bounding_box)

	# modify part locations if needed
	if part_locs is not None:
		modified_pl = _modify_crop_pl(crop_width, crop_height, ws, hs, part_locs)

	# return accordingly
	if bounding_box is not None and part_locs is not None:
		return cropped_img, modified_bb, modified_pl
	elif bounding_box is not None:
		return cropped_img, modified_bb
	elif part_locs is not None:
		return cropped_img, modified_pl
	else:
		return cropped_img

def central_crop(input_img, crop_width, crop_height, bounding_box=None, part_locs=None):
	"""
	Central crop the input_img to specified size.
	Adjust the bounding box according to the cropped image if specified.

	Inputs:
		- input_img: input image of PIL Image format
		- crop_width: width of the desired crop size
		- crop_height: height of the desired crop size
		- bounding_box: (optional) bounding box info in tuple format(x, y, w, h)
		- part_locs: (optional) part locations info in dictionary format

	Returns:
		- cropped_img: cropped output image of PIL Image format
		- modified_bb: (optional) modified bounding box in tuple format (x, y, w, h)
		- modified_pl: (optional) modified part locations in dictionary format
	"""
	input_width, input_height = input_img.size

	# width start and height start
	ws = int((input_width - crop_width)/2)
	hs = int((input_height - crop_height)/2)

	# crop image
	cropped_img = input_img.crop((ws, hs, ws+crop_width, hs+crop_height))

	# modify bouding box if needed
	if bounding_box is not None:
		modified_bb = _modify_crop_bb(crop_width, crop_height, ws, hs, bounding_box)

	# modify part locations if needed
	if part_locs is not None:
		modified_pl = _modify_crop_pl(crop_width, crop_height, ws, hs, part_locs)

	# return accordingly
	if bounding_box is not None and part_locs is not None:
		return cropped_img, modified_bb, modified_pl
	elif bounding_box is not None:
		return cropped_img, modified_bb
	elif part_locs is not None:
		return cropped_img, modified_pl
	else:
		return cropped_img

def augment_image_batch(
		input_batch,
		flr=None,
		add=None,
		agn=None,
		coarse_dropout=None,
		coarse_sp=None,
		bounding_box=None,
		part_locs=None):
	"""
	Augment image batch using imgaug.
	(https://github.com/aleju/imgaug)
	(http://imgaug.readthedocs.io/en/latest/source/augmenters.html#noop)

	Inputs:
		- input_batch: batch of numpy format images of shape (batch, width, height, channel)
		- flr: rate of flipping left and right (ex. 0.5)
		- add: add value to all pixels (ex. (-10, 10))
		- agn: add gaussian noise (ex. 0.05)
		- coarse_dropout: add coarse dropout by certain fraction of pixels to zero (ex. 0.2)
		- coarse_sp: add coarse salt & pepper noise (ex. 0.2)
		- bounding_box: (optional) bounding box info in tuple format(x, y, w, h)
		- part_locs: (optional) part locations info in numpy array of dictionaries

	Returns:
		- output: augmented image in numpy data type
		- bounding_box: (optional) modified bounding box
		- part_locs: (optional) modified part locations info in numpy array of dictionaries
	"""
	augment_list = []

	# we flip left and right without using the 'imgaug' library
	# because we need to modify the bounding box info
	if flr is not None:
		batch_size = input_batch.shape[0]
		flr_idx = np.random.choice([0, 1], batch_size, p=[1-flr, flr])

		for i, flip in enumerate(flr_idx):
			if flip == 1:
				input_batch[i,:] = np.fliplr(input_batch[i,:])

				if bounding_box is not None:
					x, y, w, h = bounding_box[i]
					width = input_batch[i].shape[0]
					x = width - x - w

					bounding_box[i] = (x, y, w, h)

				if part_locs is not None:
					for key, value in part_locs[i].items():
						x, y = value
						width = input_batch[i].shape[0]
						x = width - x

						part_locs[i][key] = (x, y)

	# do rest of the augmentation
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

	output = aug.augment_images(input_batch)

	if bounding_box is not None and part_locs is not None:
		return output, bounding_box, part_locs
	elif bounding_box is not None:
		return output, bounding_box
	elif part_locs is not None:
		return output, part_locs
	else:
		return output

def rgb_2_centered_bgr(rgb, mean_rgb):
	"""
	Convert RGB image to mean subtracted BGR image.

	Inputs:
		- rgb: input image in RGB format in range [0, 255]
		- mean_rgb: mean of the dataset in RGB format

	Returns:
		- centered_bgr: centered bgr image
	"""
	red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)

	centered_bgr = tf.concat(
		axis=3,
		values=[blue - mean_rgb[2],
				green - mean_rgb[1],
				red - mean_rgb[0],])

	return centered_bgr

def centered_bgr_2_rgb(centered_bgr, mean_rgb):
	"""
	Convert centered BGR image to original RGB image.

	Inputs:
		- centered_bgr: input image in centered BGR format
		- mean_rgb: mean of the dataset in RGB format

	Returns:
		- rgb: rgb image
	"""
	blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=centered_bgr)

	rgb = tf.concat(
		axis=3,
		values=[red + mean_rgb[0],
				green + mean_rgb[1],
				blue + mean_rgb[2],])

	return rgb

def draw_bounding_boxes(images, bounding_boxes_list):
	"""
	Draw bounding boxes and return.

	Inputs:
		- images: tensorflow placeholder containing batch of images
		- bounding_boxes_list: tensorflow placeholder containing list of
			batch bounding boxes information corresponding to the images
			(ex. (x, y, w, h) relative to width and height of image)

	Returns:
		- tensor containing the batch of images with bounding box
	"""
	batch_size = images.get_shape().as_list()[0]
	temp_boxes_list = []

	# make sure bounding_boxes_list is a list
	assert isinstance(bounding_boxes_list, list)

	# iterate through the bounding box list and convert to
	# appropriate shape
	for bb in bounding_boxes_list:
		assert batch_size is bb.get_shape().as_list()[0]

		x, y, w, h = tf.split(bb, 4, 1)
		x_min = tf.clip_by_value(x, 0.0, 1.0)
		y_min = tf.clip_by_value(y, 0.0, 1.0)
		x_max = tf.clip_by_value(tf.add(x, w), 0.0, 1.0)
		y_max = tf.clip_by_value(tf.add(y, h), 0.0, 1.0)

		temp_boxes = tf.concat([y_min, x_min, y_max, x_max], axis=1)
		temp_boxes = tf.expand_dims(temp_boxes, axis=1)

		temp_boxes_list.append(temp_boxes)

	# we want the shape to be [batch, num_bounding_boxes, 4]
	boxes = tf.concat(temp_boxes_list, axis=1)

	return tf.image.draw_bounding_boxes(images, boxes)

def bb_pixels_2_relative(bounding_boxes, width, height):
	"""
	Convert bounding box information in pixels to values
	relative to width and height of the image.

	Inputs:
		- bounding_boxes: numpy array containing batches of
			tuples containing bounding box info in pixels
		- width: width of the image
		- height: height of the image

	Returns:
		- relative_bb: numpy array same as bounding_boxes format
			containing the bounding box info in relative coordinates
	"""
	batch_size = bounding_boxes.shape[0]
	relative_bb = np.empty_like(bounding_boxes)

	for i in range(batch_size):
		x, y, w, h = bounding_boxes[i]
		x /= float(width)
		y /= float(height)
		w /= float(width)
		h /= float(height)
		relative_bb[i] = (x, y, w, h)

	return relative_bb

def pl_pixels_2_relative(part_locs, width, height):
	"""
	Convert part locations information in pixels to values
	relative to width and height of the image.

	Inputs:
		- part_locs: numpy array containing batches of
			dictionaries containing part locations info in pixels
		- width: width of the image
		- height: height of the image

	Returns:
		- relative_pl: numpy array same as part_locs format
			containing the bounding box info in relative coordinates
	"""
	batch_size = part_locs.shape[0]
	relative_pl = np.empty_like(part_locs)

	for i in range(batch_size):
		dictionary = {}
		for key, value in part_locs[i].items():
			x, y = value
			x /= float(width)
			y /= float(height)
			dictionary[key] = (x, y)
		relative_pl[i] = dictionary

	return relative_pl

def check_part_locs_boundary(part_locs, crop_width, crop_height, part_key):
	"""
	Check if specified part_key locations in the part_locs numpy array
	is outside the boundary of the image size. For example, if specified
	part_key is a negative value, or larger than crop_width or crop_height,
	function returns True denoting part is out of boundary.

	Inputs:
		- part_locs: numpy array containing batches of
			dictionaries containing part locations info
		- crop_width: image width upper bound
		- crop_height: image height upper bound
		- part_key: string denoting specific part to check for (ex. 'head')

	Returns:
		- out_of_bounds: True if out of bound, False if not
	"""
	out_of_bounds = False

	for key, value in part_locs.items():
		if key is part_key:
			x, y = value
			if x < 0 or y < 0:
				out_of_bounds = True

			if x > crop_width or y > crop_height:
				out_of_bounds = True

	return out_of_bounds

def extract_part_from_part_locs(part_locs, part_key):
	"""
	Extract specified 'part_key' from dictionary containing
	multiple part locatioins.

	Inputs:
		- part_locs: numpy array containing batches of
			dictionaries containing part locations info
		- part_key: string denoting specific part location (ex. 'head')

	Returns:
		- part_loc: numpy array containing extracted single
			part location, where each row is a tuple (x, y)
	"""
	batch_size = part_locs.shape[0]
	part_loc = np.empty(batch_size, dtype=(np.float32, 2))

	for i in range(batch_size):
		for key, value in part_locs[i].items():
			if key is part_key:
				x, y = value
				break

		part_loc[i] = (x, y)

	return part_loc
