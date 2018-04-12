import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import os
import sys
from PIL import Image, ImageDraw, ImageFont
from random import shuffle
from cnn_utils import set_directories, get_directory, get_image_pixels, get_image_description, get_image_names
import numpy as np
import json



MODEL_NAME = "model"

TRAIN_SIZE = 2172
VALIDATION_SIZE = 140
TEST_SIZE = 80

BATCH_SIZE = 25

TRAIN_STEPS = TRAIN_SIZE // BATCH_SIZE
VALIDATION_STEPS = VALIDATION_SIZE // BATCH_SIZE
EPOCHS = 50

TEST_STEPS = TEST_SIZE // BATCH_SIZE

IMAGE_SIZE = 224



#Generator that yields the input and true output data
def data_generator(is_training = False, is_validation = False, is_evaluate = False):

	#Create the placeholders for input and output
	X = np.random.rand(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3) 
	Y = np.random.rand(BATCH_SIZE, 1)

	image_names = get_image_names(is_training, is_validation)
	index = 0

	if(is_training or is_validation or is_evaluate):
		shuffle(image_names)

	#Generates the data indefinitely
	while True:

		#For each image in the batch, returns its pixels and description
		for i in range(BATCH_SIZE):

			image_number = image_names[index]
			index += 1

			X[i] = get_image_pixels(image_number, is_training = is_training, is_validation = is_validation)

			if(is_training or is_validation or is_evaluate):
				Y[i] = get_image_description(image_number, is_training = is_training, is_validation = is_validation)
			
			if(index == len(image_names) - 1):
				index = 0
				shuffle(image_names)

		#Once the whole batch is ready, yield the inputs and true outputs
		if(is_training or is_validation or is_evaluate):
			yield X, Y
		else:
			yield X


#Create the architecture of the model as well as compile it
def create_model():

	model = Sequential()

	#Conv Layer 1
	model.add(Conv2D(16, (3, 3), input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3), padding = "same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(strides = (2, 2), padding = "same"))

	#Conv Layer 2
	model.add(Conv2D(8, (3, 3), padding = "same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(strides = (2, 2), padding = "same"))

	#Conv Layer 3
	model.add(Conv2D(4, (3, 3), padding = "same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(strides = (2, 2), padding = "same"))

	#Conv Layer 4
	model.add(Conv2D(2, (3, 3), padding = "same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(strides = (2, 2), padding = "same"))

	#Conv Layer 5
	model.add(Conv2D(1, (3, 3), padding = "same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(strides = (2, 2), padding = "same"))

	model.add(Flatten())

	#Fully Connected Layer 1
	model.add(Dense(32, activation = "relu"))
	model.add(Dropout(0.5))

	#Fully Connected Layer 2
	model.add(Dense(16, activation = "relu"))
	model.add(Dropout(0.5))

	#Final Activation Layer
	model.add(Dense(1, activation = "sigmoid"))

	model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
	
	print(model.summary())
	
	return model


#Train the model with training and validation images using data augmentation or without it
def train(data_augmentation = True):

	model = create_model()

	#Add some checkpoints
	tensorboard = TensorBoard(log_dir = './Graph', histogram_freq = 0, write_graph = True, write_images = True)
	checkpoint_train = ModelCheckpoint("model_train.h5", monitor = "loss", save_best_only = True)
	checkpoint_validation = ModelCheckpoint("model_validation.h5", monitor = "val_loss", save_best_only = True)
	
	if(data_augmentation):

		#Generators with data augmentation for the training and validation images
		train_data_generator = ImageDataGenerator(rotation_range=10,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    rescale=1/255.,
                    fill_mode='nearest',
                    channel_shift_range=0.2*255)
	
		validation_data_generator = ImageDataGenerator(rescale=1/255.)
		
		train_generator = train_data_generator.flow_from_directory(
	                get_directory(is_training = True),
	                target_size=(IMAGE_SIZE, IMAGE_SIZE),
	                batch_size=BATCH_SIZE,
	                shuffle = True,
	                class_mode='binary')
		
		validation_generator = validation_data_generator.flow_from_directory(
                    get_directory(is_validation = True),
                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                    batch_size=BATCH_SIZE,
                    shuffle = True,
                    class_mode='binary')

		#Train the model on the images
		model.fit_generator(train_generator, steps_per_epoch = TRAIN_STEPS, validation_data = validation_generator, 
			validation_steps = VALIDATION_STEPS, epochs = EPOCHS, callbacks = [tensorboard, checkpoint_train, checkpoint_validation])

	else:

		#Train the model on the images without data augmentation
		model.fit_generator(data_generator(True), steps_per_epoch = TRAIN_STEPS, validation_data = data_generator(is_validation = True), 
			validation_steps = VALIDATION_STEPS, epochs = EPOCHS, callbacks = [tensorboard, checkpoint_train, checkpoint_validation])

	

#Test the model either by predicting or evaluating test images
def test(predict_or_evaluate = "predict"):

	#Different results due to loading the model - model is compiled exactly, meaning Dropout still remains
	model = load_model("model_validation.h5")	#Up to 87% correct on test
	
	#Predict the inputted images' output
	if(predict_or_evaluate == "predict"):

		predictions = model.predict_generator(data_generator(), steps = TEST_STEPS, verbose = 1)

		image_names = get_image_names()

		#Iterate through each test image
		for i in range(len(predictions)):

			image_name = image_names[i]

			image = Image.open("Data/images_test/" + image_name + ".jpg").convert("RGBA")

			#Create empty image for text
			text = Image.new('RGBA', image.size, (255, 255, 255, 0))

			font = ImageFont.truetype('arial.ttf', 40)

			#Drawing context
			draw = ImageDraw.Draw(text)

			#Prediction is from 0-1. <0.5 for benign and 0.5-1 for malignant
			if(predictions[i] < 0.5):
				prediction = "benign"
			else:
				prediction = "malignant"

			#Draw the text
			draw.text((10, 10), prediction, font = font, fill = (255, 255, 255, 255))

			#Combine the original image with text image
			output = Image.alpha_composite(image, text)

			output.save("Data/output/" + image_name + ".png", "PNG")
	
	else:

		#Evaluate the model on test images
		evaluations = model.evaluate_generator(data_generator(is_evaluate = True), steps = TEST_STEPS)
		print(evaluations)


#Convert the model into a photobuf (.pb) file
def export_model(saver, model, input_node_names, output_node_name):

    tf.train.write_graph(K.get_session().graph_def, 'out', \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")



#Code for running the program from the terminal
terminal_length =  len(sys.argv)

if(terminal_length >= 2):
    
    #Help command
    if((sys.argv[1] == "-h" or sys.argv[1] == "--help") and terminal_length == 2):
        
        print("-h or --help for the list of all possible commands")
        print("--train for training the model with data augmentation by default")
        print("    --no_data_augmentation subcommand for training if you don't want to use data augmentation")
        print("--predict for predicting an output for a set of input images")
        print("--evaluate for evaluating the model on a set of input images and their corresponding outputs")
    
    #Train command
    elif(sys.argv[1] == "--train" and terminal_length == 2):
        
        print("Training with data_augmentation...")
        train()
    
    #Train with no data augmentation command
    elif(sys.argv[1] == "--train" and terminal_length == 3 and sys.argv[2] == "--no_data_augmentation"):
            
        print("Training with no data augmentation...")
        set_directories()
        train(False)
    
    #Predict command
    elif(sys.argv[1] == "--predict" and terminal_length == 2):
        
        print("Predicting images...")
        test()
        
    #Evaluate command
    elif(sys.argv[1] == "--evaluate" and terminal_length == 2):
        
        print("Evaluating images...")
        test("evaluate")
        
    elif(sys.argv[1] == "--export" and terminal_length == 2):

    	print("Exporting the model into .pb file")
    	model = load_model("model_validation.h5")
    	export_model(tf.train.Saver(), model, ["conv2d_1_input"], "dense_3/Sigmoid")

    #Invalid command
    else:
        
        print("Invalid command.")
        print("Use -h or --help for the list of all possible commands")

else:
    
    print("No arguments given. Please be sure to include one.")
    print("Use -h or --help for the list of all possible commands")
