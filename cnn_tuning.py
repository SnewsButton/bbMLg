from cnn_dice import *

def get_tuned_model(hyperparameter_combo):
	if len(hyperparameter_combo) is not 6:
		return

	blob_image_input = Input(shape=(60, 50, 3), name='blob_image_input')

	conv2d_1 = Conv2D(hyperparameter_combo[0], (3, 3), padding='same', activation='relu')(blob_image_input)
	maxpooling2d_1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(conv2d_1)
	conv2d_2 = Conv2D(hyperparameter_combo[1], (3, 3), padding='same', activation='relu')(maxpooling2d_1)
	maxpooling2d_2 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(conv2d_2)
	image_dense_5 = Dense(hyperparameter_combo[2], activation='relu')(maxpooling2d_2)
	image_flatten = Flatten()(image_dense_5) # concatenate requires that all layers yield the same dimension

	# LINE
	line_input = Input(shape=(8,))

	line_output_1 = Dense(hyperparameter_combo[3], activation='relu')(line_input)
	line_output_2 = Dense(hyperparameter_combo[4], activation='relu')(line_output_1)
	line_output_3 = Dense(hyperparameter_combo[5], activation='relu')(line_output_2)
	line_output_4 = Dense(hyperparameter_combo[2], activation='relu')(line_output_3)

	# CENTER
	center_input = Input(shape=(2,))
	center_dense = Dense(hyperparameter_combo[2], activation='relu')(center_input)

	# CONCATENATE
	concatenated = concatenate([image_flatten, line_output_4, center_dense])

	dense_6 = Dense(64, activation='relu')(concatenated)
	dense_7 = Dense(64, activation='relu')(dense_6)
	output = Dense(1, activation='sigmoid')(dense_7)


	model = Model(inputs=[blob_image_input, line_input, center_input], outputs=[output])

	model.compile(loss='binary_crossentropy',optimizer="adam", metrics=['accuracy'])
	
	return model

def get_hyperparameter_combos(hyperparameter_values, n):
	combos=[[]]

	for _ in range(n):
		new_combos = []
		for combo in combos:
			for value in hyperparameter_values:
				combo_copy = combo.copy()
				combo_copy.append(value)
				new_combos.append(combo_copy)
		combos = new_combos
	
	return combos

def hyperparameter_tuning(hyperparameter_values, n):
	data = create_train_and_test_data(100)

	hyperparameter_combos = get_hyperparameter_combos(hyperparameter_values, n)

	for combo in hyperparameter_combos:
		tuned_model = get_tuned_model(combo)
		print(combo)
		train_cnn(tuned_model, data)

hyperparameter_values = [32, 64]
hyperparameter_tuning(hyperparameter_values, 6)
