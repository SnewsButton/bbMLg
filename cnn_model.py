import numpy as np
import cv2 as cv
from preprocessing import process_all_images
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, concatenate
from keras.callbacks import ModelCheckpoint

def get_cnn_model_1():
	# https://keras.io/getting-started/functional-api-guide/
	# IMAGE
	blob_image_input = Input(shape=(60, 50, 3), name='blob_image_input')

	conv2d_1 = Conv2D(64, (3, 3), padding='same', activation='linear')(blob_image_input)
	maxpooling2d_1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(conv2d_1)
	conv2d_2 = Conv2D(64, (3, 3), padding='same', activation='linear')(maxpooling2d_1)
	maxpooling2d_2 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(conv2d_2)
	image_dense_5 = Dense(64, activation='linear')(maxpooling2d_2)
	image_flatten = Flatten()(image_dense_5) # concatenate requires that all layers yield the same dimension

	# LINE
	line_input = Input(shape=(8,))

	line_output_1 = Dense(64, activation='linear')(line_input)
	line_output_2 = Dense(64, activation='linear')(line_output_1)
	line_output_3 = Dense(64, activation='linear')(line_output_2)
	line_output_4 = Dense(64, activation='linear')(line_output_3)

	# CENTER
	center_input = Input(shape=(2,))
	center_dense = Dense(64, activation='linear')(center_input)

	# CONCATENATE
	concatenated = concatenate([image_flatten, line_output_4, center_dense])

	dense_6 = Dense(64, activation='linear')(concatenated)
	dense_7 = Dense(64, activation='linear')(dense_6)
	output = Dense(21, activation='softmax')(dense_7)


	model = Model(inputs=[blob_image_input, line_input, center_input], outputs=[output])

	model.compile(loss='binary_crossentropy',optimizer="adam", metrics=['accuracy'])
	
	return model

def get_cnn_model():
	# https://keras.io/getting-started/functional-api-guide/
	# IMAGE
	blob_image_input = Input(shape=(60, 50, 3), name='blob_image_input')

	conv2d_1 = Conv2D(64, (3, 3), padding='same', activation='linear')(blob_image_input)
	conv2d_1_2 = Conv2D(64, (3, 3), padding='same', activation='linear')(conv2d_1)

	maxpooling2d_1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(conv2d_1_2)
	conv2d_2 = Conv2D(64, (3, 3), padding='same', activation='linear')(maxpooling2d_1)
	conv2d_2_2 = Conv2D(64, (3, 3), padding='same', activation='linear')(conv2d_2)

	maxpooling2d_2 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(conv2d_2_2)
	image_dense_5 = Dense(64, activation='linear')(maxpooling2d_2)
	image_flatten = Flatten()(image_dense_5) # concatenate requires that all layers yield the same dimension

	# LINE
	line_input = Input(shape=(8,))

	line_output_1 = Dense(64, activation='linear')(line_input)
	line_output_2 = Dense(64, activation='linear')(line_output_1)
	line_output_3 = Dense(64, activation='linear')(line_output_2)
	line_output_4 = Dense(64, activation='linear')(line_output_3)

	# CENTER
	center_input = Input(shape=(2,))
	center_dense = Dense(64, activation='linear')(center_input)

	# CONCATENATE
	concatenated = concatenate([image_flatten, line_output_4, center_dense])

	dense_6 = Dense(64, activation='linear')(concatenated)
	dense_7 = Dense(64, activation='linear')(dense_6)
	output = Dense(21, activation='softmax', name='output')(dense_7)


	model = Model(inputs=[blob_image_input, line_input, center_input], outputs=[output])

	model.compile(loss='sparse_categorical_crossentropy',optimizer="adam", metrics=['accuracy'])
	
	return model
