#Tested on Python3 running on Linux Mint Cinnamon & MacOSX Catalina 
#Requires PySimpleGUI, TensorFlow
#Tested on wine quality dataset from UCI database (https://archive.ics.uci.edu/ml/datasets/wine+quality) 
#Requires feature set to be in dtype=float


import pandas as pd 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt 
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout

import os


import PySimpleGUI as sg
sg.theme('DarkAmber')


layout = [[sg.Text('Your parameters')],
          [sg.Text('Input file', size=(15, 1)), sg.Input(), sg.FileBrowse()],
          [sg.Frame(layout=[
          [sg.Text('# neurons first layer: ', size=(15, 1)), sg.InputText()],
          [sg.Text('# neurons second layer: ', size=(15, 1)), sg.InputText()],
          [sg.Text('Batch size: ', size=(15, 1)), sg.InputText()],
          [sg.Text('# of epochs ', size=(15, 1)), sg.InputText()]], [sg.Text('Label column ID: ', size=(15, 1)), sg.InputText()]],
          [sg.Text('Regularization ', size=(15, 1)), sg.InputText()],
          [sg.Text('Dropout ', size=(15, 1)), sg.InputText()], title='Options',title_color='red', relief=sg.RELIEF_SUNKEN, tooltip='Enter your parameters')],
          [sg.Submit(), sg.Cancel()]]

window = sg.Window('Run Analysis', layout)
event, values = window.read()
window.close()

def run_nn(nneurons_0=values[1], nneurons_1=values[2], batchsize=values[3], nepochs=values[4], target_label=values[5):

	path = os.getcwd()
	os.chdir(path)
	print('your working directory is: ' + path)

	df = pd.read_csv(values[0])

	print('reading in data...complete')

	y = df[target_label].values
	X = df.select_dtypes('float64').values

	min_max_scaler = preprocessing.MinMaxScaler()
	X_scale = min_max_scaler.fit_transform(X)

	X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size = 0.2, random_state = 4)

	print('splitting into train and test split...complete')

	y_train_sequential = tf.keras.utils.to_categorical(y_train)

	if values[6] and values[7] == 'False':

		model = Sequential([
		  Dense(64, activation='relu', input_shape(11,)),
		  Dense(64, activation='relu'),
		  Dense(y_train_categorical.shape[1], activation='softmax')
		])

	if values[6] == 'True':

		model = Sequential([
  			Dense(64, activation='relu', input_shape(11,), kernel_regularizer=regularizers.l2(0.001)),
  			Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
  			Dense(y_train_categorical.shape[1], activation='softmax')
		])

	else:

		model = Sequential([
  			Dense(64, activation='relu', input_shape(11,)),
  			Dropout(0.5),
  			Dense(64, activation='relu'),
  			Dropout(0.5),
 		 	Dense(y_train_categorical.shape[1], activation='softmax')
		])

	print('building neural network...complete')

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	history = model.fit(X_train, y_train_categorical, batch_size=512, epochs=1000, validation_split=0.2)

	def myprint(w):
		with open ('neural_network_summary.txt', 'a') as f:
			print(w, file=f)

	model.summary(print_fn = myprint)

	print('saving model summary...done')

	plt.style.use('fivethirtyeight')
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend(['Train', 'Val'], loc='upper right')
	plt.show()

	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend(['Train', 'Val'], loc='upper right')
	plt.show()

	results = model.evaluate(X_test, y_test_categorical) 

run_nn(nneurons_0=values[1], nneurons_1=values[2], batchsize=values[3], nepochs=values[4])
