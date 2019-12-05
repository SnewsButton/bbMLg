import numpy as np
import cv2 as cv
from preprocessing import process_all_images, process_all_selected_image
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, concatenate
from keras.callbacks import ModelCheckpoint
from cnn_model import *

def create_train_and_test_data(N):
	all_info = process_all_images(N)
	alldata = all_info['alldata']
	allblobs = all_info['allblobs']
	alllines = all_info['alllines']
	size = alldata.size
	print('size:', size)

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

	lines_train = np.array(lines_split[0])
	lines_test = np.array(lines_split[1])

	blobs_center_train = np.array([blobs_data['Center'] for blobs_data in blobs_data_train])
	blobs_center_test = np.array([blobs_data['Center'] for blobs_data in blobs_data_test])

	return {
		'data_train': data_train,
		'data_test': data_test,
		'blobs_train': [blobs_img_train, lines_train, blobs_center_train],
		'blobs_test': [blobs_img_test, lines_test, blobs_center_test]
	}


def train_cnn(model, train_and_test_data, N):
	print('Start training')
	blobs_train, blobs_test = train_and_test_data['blobs_train'], train_and_test_data['blobs_test']
	data_train, data_test = train_and_test_data['data_train'], train_and_test_data['data_test']

	record_weights = ModelCheckpoint(filepath='./model_weights/weights_' + str(N) + '_{epoch:02d}-{val_loss:.2f}.hdf5',
		monitor='val_loss',
		verbose=0,
		save_best_only=False,
		save_weights_only=False,
		mode='auto',
		period=1)

	model.fit(blobs_train, data_train,
		epochs=50,
		verbose=2,
		validation_data=(blobs_test, data_test),
		callbacks=[record_weights])

	score = model.evaluate(blobs_test, data_test, verbose=0)
	print(score)

	model.save_weights('./model_weights/' + str(N) + '_epochs.h5')
	print('End training')

class DicePredicter():

        def __init__(self,filename):
                self.model = self.load_model_from_file(filename)
        
        def load_model_from_file(self,filename):
                print('Loading model from', filename)
                model = load_model(filename)
                print('Finished loading model')
                return model

        def predict_on_files(self, file_tuples):
                predict_for_tuple = {}
                print(file_tuples)
                for file in file_tuples:
                        all_info = process_all_selected_image(file)
                        allblobs, alllines = all_info['allblobs'], all_info['alllines']

                        blobs_img = np.array([blobs_data['Image'] for blobs_data in allblobs])
                        lines_data = np.stack((alllines,)*blobs_img.shape[0])
                        blobs_center = np.array([blobs_data['Center'] for blobs_data in allblobs])

                        input_data = [blobs_img, lines_data, blobs_center]
                        prediction = self.model.predict(input_data)
                        #print(prediction)
                        print(prediction.shape[0])

                        face_values = self.extract_value_from_prediction(prediction)
                        print(face_values)

                        predict_for_tuple[file] = face_values
                
                return predict_for_tuple

        def extract_value_from_prediction(self,prediction):
                values = []
                shape = prediction.shape
                for i_1 in range(shape[0]):
                        max_val, max_ind = 0, 0
                        for i_2 in range(shape[1]):
                                if prediction[i_1][i_2] > max_val:
                                        max_val = prediction[i_1][i_2]
                                        max_ind = i_2

                        values.append(max_ind)

                return values

# load model from file
# predict_on_files('model_weights/weights_4000_17-12.08.hdf5', ('data1120b/3d3884.png','data1120b/1d0.png'))

# Training model
# N = 4000
# print('N:', N)
# data = create_train_and_test_data(N)
# model = get_cnn_model()
# train_cnn(model, data, N)
