Skin Cancer Detector
By Rauf Makharov
For the 2017-2018 Science and Programming Fairs

Uses a Convolutional Neural Network to classify skin cancer either as benign or malignant. The model has 5 Conv,
5 ReLU activation, 5 Max Pooling, Flatten, 3 Dense (last one with Sigmoid), and 2 Dropout layers. 

The model was trained on ~4,000 images from ISIC, PH2, and Complete MedNode online databases.

The program was built using Tensorflow with Keras framework and achieved an accuracy of 87%.

WARNING: The program is to be used as a supplementary tool. You should not be solely relying on the predictions
of the program. If you're concerned about skin cancer, do not hesitate to contact a professional dermatologist.
