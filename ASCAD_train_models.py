import os
import os.path
import sys
import h5py
import numpy as np
import tensorflow as tf
from tf_keras.models import Model, Sequential
from tf_keras.layers import Flatten, Dense, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D, BatchNormalization, DepthwiseConv2D, Activation, Add, add, Dropout
from tf_keras import backend as K
from tf_keras.applications.imagenet_utils import decode_predictions
from tf_keras.applications.imagenet_utils import preprocess_input
from tf_keras.optimizers import RMSprop, Adam
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping
from tf_keras.utils import to_categorical
from tf_keras.models import load_model

# Profiling 

# Author : Edoardo Diana

# Resources used : https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1
# Python version : 3.9

# Checks if the given file path exists, if not then terminate
def check_file_exists(file_path):
	file_path = os.path.normpath(file_path)
	if os.path.exists(file_path) == False:
		print("Error: provided file path '%s' does not exist!" % file_path)
		sys.exit(-1)
	return

# The default function for generating the optimal mlp model (from the original git code) is mlp_best.
# It uses six layers of 200 neurons each to build a Multi-Layer Perceptron (MLP) model.
# It uses a SoftMax activation function on the final layer for classification and ReLU as an activation function 
# for the hidden layers.RMSprop is used for optimization, with a learning rate of 0.00001.calculates the accuracy 
# using a categorical-crossentropy loss function.

def mlp_best(node=200, layer_nb=6, input_dim=1400):
	model = Sequential()
	model.add(Dense(node, input_dim=input_dim, activation='relu'))
	for i in range(layer_nb-2):
		model.add(Dense(node, activation='relu'))

	model.add(Dense(256, activation='softmax'))
	optimizer = RMSprop(learning_rate=0.00001)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model


def cnn_best(classes=256, input_dim=700):
	input_shape = (input_dim, 1)
	img_input = Input(shape=input_shape)
	# Block 1
	x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(img_input)
	x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
	# Block 2
	x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
	x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
	# Block 3
	x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
	x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
	# Block 4
	x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
	x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
	# Block 5
	x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
	x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
	# Classification block
	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dense(classes, activation='softmax', name='predictions')(x)

	inputs = img_input
	# Create model.
	model = Model(inputs, x, name='cnn_best')
	optimizer = RMSprop(learning_rate=0.00001)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model



def load_sca_model(model_file):
	check_file_exists(model_file)
	try:
			model = load_model(model_file)
	except:
		print("Error: can't load Keras model file '%s'" % model_file)
		sys.exit(-1)
	return model


#### ASCAD helper to load profiling and attack data (traces and labels)
# Loads the profiling and attack datasets from the ASCAD
# database
def load_ascad(ascad_database_file, load_metadata=False):
	check_file_exists(ascad_database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file	 = h5py.File(ascad_database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
		sys.exit(-1)
	# Load profiling traces
	X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
	# Load profiling labels
	Y_profiling = np.array(in_file['Profiling_traces/labels'])
	# Load attacking traces
	X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
	# Load attacking labels
	Y_attack = np.array(in_file['Attack_traces/labels'])
	if load_metadata == False:
		return (X_profiling, Y_profiling), (X_attack, Y_attack)
	else:
		return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])


def multilabel_to_categorical(Y):
	y = {}
	y['alpha_output'] = to_categorical(Y['alpha_mask'], num_classes=256)
	y['beta_output'] = to_categorical(Y['beta_mask'], num_classes=256)
	for i in range(16):
		y['sbox_'+str(i)+'_output'] = to_categorical(Y['sbox_masked'][:,i], num_classes=256)
	for i in range(16):
		y['permind_'+str(i)+'_output'] = to_categorical(Y['perm_index'][:,i], num_classes=16)
	return y


def multilabel_without_permind_to_categorical(Y):
	y = {}
	y['alpha_output'] = to_categorical(Y['alpha_mask'], num_classes=256)
	y['beta_output'] = to_categorical(Y['beta_mask'], num_classes=256)
	for i in range(16):
		y['sbox_'+str(i)+'_output'] = to_categorical(Y['sbox_masked_with_perm'][:,i], num_classes=256)
	return y


#### Training high level function
def train_model(X_profiling, Y_profiling, model, save_file_name, epochs=150, batch_size=100, multilabel=0, validation_split=0, early_stopping=0):
	check_file_exists(os.path.dirname(save_file_name))
	# Save model calllback
	save_model = ModelCheckpoint(save_file_name)
	callbacks = [save_model]
	# Early stopping callback
	if (early_stopping != 0):
		if validation_split == 0:
			validation_split = 0.1
		callbacks.append(EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True))
	# Get the input layer shape
	if isinstance(model.get_layer(index=0).input_shape, list):
		input_layer_shape = model.get_layer(index=0).input_shape[0]
	else:
		input_layer_shape = model.get_layer(index=0).input_shape
	# Sanity check
	if input_layer_shape[1] != len(X_profiling[0]):
		print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_profiling[0])))
		sys.exit(-1)
	# Adapt the data shape according our model input
	if len(input_layer_shape) == 2:
		# This is a MLP
		Reshaped_X_profiling = X_profiling
	elif len(input_layer_shape) == 3:
		# This is a CNN: expand the dimensions
		Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
	else:
		print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
		sys.exit(-1)
	if (multilabel==1):
		y = multilabel_to_categorical(Y_profiling)
	elif (multilabel==2):
		y = multilabel_without_permind_to_categorical(Y_profiling)
	else:
		y = to_categorical(Y_profiling, num_classes=256)
	history = model.fit(x=Reshaped_X_profiling, y=y, batch_size=batch_size, verbose = 1, validation_split=validation_split, epochs=epochs, callbacks=callbacks)
	return history


def read_parameters_from_file(param_filename):
	#read parameters for the train_model and load_ascad functions
	#TODO: sanity checks on parameters
	param_file = open(param_filename, "r")

	#TODO: replace eval() by ast.linear_eval()
	my_parameters = eval(param_file.read())

	ascad_database = my_parameters["ascad_database"]
	training_model = my_parameters["training_model"]
	network_type = my_parameters["network_type"]
	epochs = my_parameters["epochs"]
	batch_size = my_parameters["batch_size"]
	train_len = 0
	if ("train_len" in my_parameters):
		train_len = my_parameters["train_len"]
	validation_split = 0
	if ("validation_split" in my_parameters):
		validation_split = my_parameters["validation_split"]
	multilabel = 0
	if ("multilabel" in my_parameters):
		multilabel = my_parameters["multilabel"]
	early_stopping = 0
	if ("early_stopping" in my_parameters):
		early_stopping = my_parameters["early_stopping"]
	return ascad_database, training_model, network_type, epochs, batch_size, train_len, validation_split, multilabel, early_stopping


# Here i decided to focus on training a MLP model becouse of the short training time
# Using a range of training epochs lower than 200 will probably result in underfitting, more than 800 in overfitting.
# So I decided to stay between 200 and 600.

# In the folder ./ASCAD_data/ASCAD_trained_models/ you can find all trained models

if __name__ == "__main__":

	if len(sys.argv) != 2:

		ascad_database = "./ASCAD_data/ASCAD_databases/ASCAD.h5"

		#MLP training
		network_type = "mlp"
		training_model = "./ASCAD_data/ASCAD_trained_models/MLP_500epochs_50batch.h5"

		#CNN training
		#network_type = "cnn"
		#training_model = "./ASCAD_data/ASCAD_trained_models/NEW_CNN_val_split_best_desync0_epochs75_batchsize200.h5"

		validation_split = 0.1
		multilabel = 0
		train_len = 0
		epochs = 500
		batch_size = 50
		bugfix = 0
		early_stopping = 0

	else:
		#get parameters from user input
		ascad_database, training_model, network_type, epochs, batch_size, train_len, validation_split, multilabel, early_stopping = read_parameters_from_file(sys.argv[1])

	#load traces
	(X_profiling, Y_profiling), (X_attack, Y_attack) = load_ascad(ascad_database)

	# Here we call the model generation using the MLP or the CNN.
	if(network_type=="mlp"):
		best_model = mlp_best(input_dim=len(X_profiling[0]))
	elif(network_type=="cnn"):
		best_model = cnn_best(input_dim=len(X_profiling[0]))
	#display an error and abort
	else: 
		print("Error: no topology found for network '%s' ..." % network_type)
		sys.exit(-1)
	#  print best_model.summary()

	### training
	if (train_len == 0):
		early_stopping = 0
		train_model(X_profiling, Y_profiling, best_model, training_model, epochs, batch_size, multilabel, validation_split, early_stopping)
	else:
		early_stopping = 0
		train_model(X_profiling[:train_len], Y_profiling[:train_len], best_model, training_model, epochs, batch_size, multilabel, validation_split, early_stopping)
