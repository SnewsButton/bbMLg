import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import cv2 as cv
from preprocessing import process_all_images
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, LSTM, Embedding, Concatenate, Add, Reshape
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
	line_input = Input(shape=(20,))

	line_output_1 = Dense(64, activation='linear')(line_input)
	line_output_2 = Dense(64, activation='linear')(line_output_1)
	line_output_3 = Dense(64, activation='linear')(line_output_2)
	line_output_4 = Dense(64, activation='linear')(line_output_3)

	# CENTER
	center_input = Input(shape=(3,))
	center_dense = Dense(64, activation='linear')(center_input)

	# CONCATENATE
	concatenated = concatenate([image_flatten, line_output_4, center_dense])

	dense_6 = Dense(64, activation='linear')(concatenated)
	dense_7 = Dense(64, activation='linear')(dense_6)
	output = Dense(21, activation='softmax', name='output')(dense_7)


	model = Model(inputs=[blob_image_input, line_input, center_input], outputs=[output])

	model.compile(loss='sparse_categorical_crossentropy',optimizer="adam", metrics=['accuracy'])
	
	return model

def get_rnn_model():
	# https://keras.io/getting-started/functional-api-guide/
	# IMAGE
	blob_image_input = Input(shape=(60, 50, 3), name='blob_image_input')

	conv2d_1 = Conv2D(64, (3, 3), padding='same', activation='linear')(blob_image_input)
	conv2d_1_2 = Conv2D(64, (3, 3), padding='same', activation='linear')(conv2d_1)
	maxpooling2d_1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(conv2d_1_2)
	
	conv2d_2 = Conv2D(64, (3, 3), padding='same', activation='linear')(maxpooling2d_1)
	conv2d_2_2 = Conv2D(64, (3, 3), padding='same', activation='linear')(conv2d_2)
	maxpooling2d_2 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(conv2d_2_2)
	
	image_flatten = Flatten()(maxpooling2d_2) # concatenate requires that all layers yield the same dimension
	image_output = Dense(128, activation='linear')(image_flatten)

	# LINE
	line_input = Input(shape=(20,))
	line_output = Dense(64, activation='linear')(line_input)

	# CENTER
	center_input = Input(shape=(3,))

	# CONCATENATE
	concatenated = Concatenate()([image_output, line_output, center_input])
	dense_1 = Dense(256, activation='linear')(image_output)
	lstm_reshape = Reshape((256,1))(dense_1)
	lstm_1 = LSTM(64)(lstm_reshape)
	dense_2 = Dense(64, activation='linear')(lstm_1)
	output = Dense(21, activation='softmax', name='output')(dense_2)


	model = Model(inputs=[blob_image_input, line_input, center_input], outputs=[output])

	model.compile(loss='sparse_categorical_crossentropy',optimizer="adam", metrics=['accuracy'])
	
	return model

def get_rnn_model_simp():

	line_input = Input(shape=(20,))
	center_input = Input(shape=(3,))

	blob_image_input = Input(shape=(60, 50, 3), name='blob_image_input')

	conv2d_1 = Conv2D(64, (3, 3), padding='same', activation='linear')(blob_image_input)
	conv2d_1_2 = Conv2D(64, (3, 3), padding='same', activation='linear')(conv2d_1)
	maxpooling2d_1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(conv2d_1_2)
	
	conv2d_2 = Conv2D(64, (3, 3), padding='same', activation='linear')(maxpooling2d_1)
	conv2d_2_2 = Conv2D(64, (3, 3), padding='same', activation='linear')(conv2d_2)
	maxpooling2d_2 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(conv2d_2_2)
	
	image_flatten = Flatten()(maxpooling2d_2) # concatenate requires that all layers yield the same dimension
	dense_image = Dense(64,activation='linear')(image_flatten)
	image_reshape = Reshape((64,1))(dense_image)	
	
	
	lstm_image = LSTM(64)(image_reshape)
	image_output = Dense(64, activation='linear')(lstm_image)
	
	
        
	dense_1 = Dense(64, activation='linear')(image_output)
	dense_2 = Dense(64, activation='linear')(dense_1)
	output = Dense(21, activation='softmax', name='output')(dense_2)


	model = Model(inputs=[blob_image_input, line_input, center_input], outputs=[output])

	model.compile(loss='sparse_categorical_crossentropy',optimizer="adam", metrics=['accuracy'])
	
	return model
