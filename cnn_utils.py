import os
import json
import numpy as np
from PIL import Image

#By default, assigned to the data augmentation directories
TRAIN_IMAGE_DIR = "Data/images_train_data_augmentation/"
VALIDATION_IMAGE_DIR = "Data/images_validation_data_augmentation/"
TEST_IMAGE_DIR = "Data/images_test/"
DESCR_DIR = "Data/descriptions/"

IMAGE_SIZE = 224



#Change the directories if data augmentation is not used
def set_directories(data_augmentation = False):

	if(not(data_augmentation)):
		global TRAIN_IMAGE_DIR
		TRAIN_IMAGE_DIR = "Data/images_train/"
		global VALIDATION_IMAGE_DIR
		VALIDATION_IMAGE_DIR = "Data/images_validation/"



#Selects and returns the appropriate directory
def get_directory(image_or_description = "image", is_training = False, is_validation = False):

	if(image_or_description == "image"):	

		if(is_training):
			return TRAIN_IMAGE_DIR
		elif(is_validation):
			return VALIDATION_IMAGE_DIR
		else:
			return TEST_IMAGE_DIR
	
	else:

		if(is_training):
			file_name = "descriptions_train.json"
		elif(is_validation):
			file_name = "descriptions_validation.json"
		else:
			file_name = "descriptions_test.json"
		
		return DESCR_DIR, file_name



#Returns the selected image's pixels in the shape (batch, width, height, channels)
def get_image_pixels(image_number, is_training = False, is_validation = False):
	
	#Gets the directory and image_name
	directory = get_directory(is_training = is_training, is_validation = is_validation)
	image_name = str(image_number) + ".jpg"
	
	#Opens the image and resizes it
	image = Image.open(directory + image_name)
	image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)

	#Converts the image to numpy array with shape (batch, width, height, channels)
	#Makes sure all the values are between 0 and 1
	image_pixels = np.array(image, dtype = "float32")
	image_pixels /= 255.
	image_pixels = np.expand_dims(image_pixels, 0)

	return image_pixels



#Returns the selected image's description with 0 = "benign" and 1 = "malignant"
def get_image_description(image_number, is_training = False, is_validation = False):

	#Gets the directory and image_name
	directory, file_name = get_directory("description", is_training, is_validation)

	#Retrieves the image's description
	description = json.load(open(directory + file_name))
	
	#Returns 0 for "benign" and 1 for "malignant"
	return int(description[str(image_number)])



#Returns the full list of all the image names in the directory
def get_image_names(is_training = False, is_validation = False):

	directory = get_directory(is_training = is_training, is_validation = is_validation)
	
	image_names = []

	for image_name in os.listdir(directory):
		image_name = "%07d" % int(image_name[:-4])
		image_names.append(image_name)

	return image_names

