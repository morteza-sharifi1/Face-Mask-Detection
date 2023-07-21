# Face-Mask-Detection

his repository contains the code for a simple face mask detection system. The system is designed to detect faces in images and videos and determine whether the face is wearing a mask or not. This is achieved by training a Convolutional Neural Network (CNN) model on a dataset of images containing faces with and without masks.

## Dataset
The dataset used for training the model consists of 1376 images, 690 face images with masks and 686 without masks. The dataset was prepared by Prajna Bhandary and is available on Github [https://github.com/prajnasb/observations/tree/master/experiements/data]

## Data Preprocessing
The images are first read and converted to grayscale using OpenCV. They are then resized to a common size of 100x100 pixels. The pixel values are normalized to fall between 0 and 1 by dividing by 255. The processed images are reshaped to a 4D array format to be compatible with the Keras model.

The labels are also processed and converted to a categorical format.

The data preprocessing code can be found in the data_preprocessing Jupyter notebook.

## Model Training
The CNN model is defined and compiled in the training_cnn Jupyter notebook. The model consists of two convolutional layers, each followed by a max pooling layer, a flatten layer, a dropout layer, a dense layer, and finally, a softmax layer for output.

The model is trained using the Adam optimizer and the categorical cross entropy loss function. The best model is saved during training using the ModelCheckpoint callback.

## Mask Detection
The trained model is used to detect faces in a video and determine whether each face is wearing a mask or not. This is accomplished by first detecting faces in each frame using the **Haar Cascade classifier**. Each detected face is then passed through the trained model to determine whether it's wearing a mask or not.

The mask detection code can be found in the detection_mask Jupyter notebook.

## Requirements
The code was written in Python and requires the following libraries:

OpenCV
Keras
NumPy
scikit-learn
Please ensure you have these libraries installed before running the code.

## Usage
To use this code, first clone the repository. Then, run the data_preprocessing notebook to preprocess the data. Once that is done, run the training_cnn notebook to train and save the model. Finally, run the detection_mask notebook to use the trained model for mask detection in a video.












