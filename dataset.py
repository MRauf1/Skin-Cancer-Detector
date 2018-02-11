import json
import os
import re



#Creating JSON file for the descriptions of the training images from ISIC dataset(the benign images have been downsampled)
def format_training_descriptions():

	descriptions_directory = "Data/descriptions_backup/"
	images_directory = "Data/images_train/"

	descriptions = {}

	#For each image, get the file number and the description of the image and add them to the main dictionary
	for image_name in os.listdir(images_directory):
		file_name = "ISIC_" + image_name[:-4]
		description = json.load(open(descriptions_directory + file_name))
		description = description["meta"]["clinical"]["benign_malignant"]
		description = 0 if description == "benign" else 1
		file_number = image_name[:-4]
		descriptions[file_number] = description
	
	#Write the dictionary into the JSON file
	with open(descriptions_directory + "descriptions_train.json", "w") as file:
		json.dump(descriptions, file)



#Rename the training images to 7 digit number + .jpg
def format_training_images():

	directory = "Data/images_train/"

	for image_name in os.listdir(directory):
		os.rename(directory + image_name, directory + image_name[5:])



def format_validation_images():

	#Prepare the benign images
	rename_and_move_files("Data/complete_mednode_dataset/naevus/", "Data/images_validation/")
	
	#Prepare the malignant images
	rename_and_move_files("Data/complete_mednode_dataset/melanoma/", "Data/images_validation/")



#format_validation_images helper function
#Renames the existing images in the directory and moves them to the assigned directory
def rename_and_move_files(old_directory, new_directory):

	for image_name in os.listdir(old_directory):
		image_number = int(re.search(r'\d+', image_name).group())
		new_image_name = "%07d.jpg" % image_number
		os.rename(old_directory + image_name, new_directory + new_image_name)



#Writes a JSON file with the validation image descriptions
def format_validation_descriptions():

	descriptions = {}

	#Append the data about the benign skin cancer into the dictionary
	for image_name in os.listdir("Data/complete_mednode_dataset/naevus/"):
		image_number = "%07d" % int(re.search(r'\d+', image_name).group())
		descriptions[image_number] = 0

	#Append the data about the malignant skin cancer into the dictionary
	for image_name in os.listdir("Data/complete_mednode_dataset/melanoma/"):
		image_number = "%07d" % int(re.search(r'\d+', image_name).group())
		descriptions[image_number] = 1

	#Write the data into the JSON file
	with open("Data/descriptions/descriptions_validation.json", "w") as file:
		json.dump(descriptions, file)



#Formats the test images and places them in the images_test folder
def format_testing_images():

	#Contains the needed images' numbers and descriptions
	descriptions = json.load(open("Data/descriptions/descriptions_test.json"))

	#Rename and move each test image to images_test folder
	for image_number in descriptions:
		image_name = "IMD%03d.bmp" % int(image_number)
		new_image_name = "%07d.jpg" % int(image_number)
		os.rename("Data/PH2Dataset/PH2 Dataset images/" + image_name[:-4] + "/" + image_name[:-4] + "_Dermoscopic_Image/" + image_name,
			"Data/images_test/" + new_image_name)



#Writes a JSON file with the test image descriptions
def format_testing_descriptions():

	descriptions = {}

	file = open("Data/PH2Dataset/PH2_dataset.txt", "r")

	i = -1

	#Extract all the necessary data from each line
	for line in file:

		i += 1

		#Skip the first 40 benign images to create a balanced 40:40 image testing set
		if(i < 40):
			continue

		file_number = "%07d" % int(line[6:9])
		description = line[56]

		#In PH2, 1 is for atypical nevus, which is not needed, while 2 stands for malignant skin cancer
		if(description == "1"):
			continue
		elif(description == "2"):
			description = 1

		#Store all the data in the dictionary
		descriptions[file_number] = description

	file.close()

	#Write the data into the JSON file
	with open("Data/descriptions/descriptions_test.json", "w") as file:
		json.dump(descriptions, file)



#format_training_images()
#format_training_descriptions()
#format_validation_images()
#format_validation_descriptions()
#format_testing_images()
#format_testing_descriptions()