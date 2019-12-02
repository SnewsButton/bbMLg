import numpy as np
import cv2 as cv
from preprocessing import process_all_images
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, concatenate


def create_train_and_test_data(N):
	all_info = process_all_images(N)
	alldata = all_info['alldata']
	allblobs = all_info['allblobs']
	alllines = all_info['alllines']
	size = alldata.size

	#labels
	training_size = int(size * .8)
	data_split = np.split(alldata, [training_size])
	data_train, data_test = data_split[0], data_split[1]

	#features
	blobs_data_split = np.split(allblobs, [training_size])
	blobs_data_train, blobs_data_test = blobs_data_split[0], blobs_data_split[1]

	# cv.cvtColor reduces the dimension making it unusable for Conv2D which requires 4D (3D, it adds another dimension)
	# blobs_img_train = np.array([cv.cvtColor(blobs_data['Image'],cv.COLOR_BGR2GRAY) for blobs_data in blobs_data_train])
	# blobs_img_test = np.array([cv.cvtColor(blobs_data['Image'],cv.COLOR_BGR2GRAY) for blobs_data in blobs_data_test])
	blobs_img_train = np.array([blobs_data['Image'] for blobs_data in blobs_data_train])
	blobs_img_test = np.array([blobs_data['Image'] for blobs_data in blobs_data_test])

	lines_split = np.split(alllines, [training_size])
	lines_train, lines_test = [], []

	for lines in lines_split[0]:
		lines = lines.flatten()
		while lines.size < 8:
			lines = np.append(lines, [0])
		lines_train.append(lines)
	
	for lines in lines_split[1]:
		lines = lines.flatten()
		while lines.size < 8:
			lines = np.append(lines, [0])
		lines_test.append(lines)

	lines_train, lines_test = np.array(lines_train), np.array(lines_test)


	blobs_center_train = np.array([blobs_data['Center'] for blobs_data in blobs_data_train])
	blobs_center_test = np.array([blobs_data['Center'] for blobs_data in blobs_data_test])

	return {
		'data_train': data_train,
		'data_test': data_test,
		'blobs_train': [blobs_img_train, lines_train, blobs_center_train],
		'blobs_test': [blobs_img_test, lines_test, blobs_center_test]
	}


def get_cnn_model():
	# https://keras.io/getting-started/functional-api-guide/
	# IMAGE
	blob_image_input = Input(shape=(60, 50, 3), name='blob_image_input')

	conv2d_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(blob_image_input)
	maxpooling2d_1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(conv2d_1)
	conv2d_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(maxpooling2d_1)
	maxpooling2d_2 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(conv2d_2)
	image_dense_5 = Dense(64, activation='relu')(maxpooling2d_2)
	image_flatten = Flatten()(image_dense_5) # concatenate requires that all layers yield the same dimension

	# LINE
	line_input = Input(shape=(8,))

	line_output_1 = Dense(64, activation='relu')(line_input)
	line_output_2 = Dense(64, activation='relu')(line_output_1)
	line_output_3 = Dense(64, activation='relu')(line_output_2)
	line_output_4 = Dense(64, activation='relu')(line_output_3)

	# CENTER
	center_input = Input(shape=(2,))
	center_dense = Dense(64, activation='relu')(center_input)

	# CONCATENATE
	concatenated = concatenate([image_flatten, line_output_4, center_dense])

	dense_6 = Dense(64, activation='relu')(concatenated)
	dense_7 = Dense(64, activation='relu')(dense_6)
	output = Dense(1, activation='sigmoid')(dense_7)


	model = Model(inputs=[blob_image_input, line_input, center_input], outputs=[output])

	model.compile(loss='binary_crossentropy',optimizer="adam", metrics=['accuracy'])
	
	return model

def train_cnn(model, train_and_test_data):
	print('Start training')
	blobs_train, blobs_test = train_and_test_data['blobs_train'], train_and_test_data['blobs_test']
	data_train, data_test = train_and_test_data['data_train'], train_and_test_data['data_test']

	model.fit(blobs_train, data_train,
		epochs=50,
		verbose=0,
		validation_data=(blobs_test, data_test))

	score = model.evaluate(blobs_test, data_test, verbose=0)
	print(score)

	model.save_weights('50_epochs.h5')
	print('End training')


# data = create_train_and_test_data(100)
# model = get_cnn_model()
# train_cnn(model, data)
