# Importing necessary packages
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
# for pasing command line arguments for number 
import sys

arg = int(sys.argv[1])

# load training and test datasets
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scaling pixels for normalizing
def scale_pixel(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model(layerr):
	model = Sequential()
	model.add(Conv2D(filters=32, kernel_size=3, activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	filterr=64
	for layerr in range(layerr):
		model.add(Conv2D(filters=filterr, kernel_size=3, activation='relu', kernel_initializer='he_uniform'))
		filterr=filterr*2
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# run the preprocessing and modelling 
def run_test_model(layerr=arg):
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = scale_pixel(trainX, testX)
	# define model
	model = define_model(layerr)
	# fit model
	model.fit(trainX, trainY, epochs=1)
	#accuracy
	pred=model.evaluate(testX,testY)
	# printing accuracy
	print("Model Accuracy is :",int(pred[1]*100))

	try:
		f= open("accuracy.txt","w")
		f.write(str(int(pred[1]*100)))
	except:
		print("not written")
	finally:
		f.close()
	# save model
	model.save('model.h5')

# entry point, run the test the model
if __name__ == "__main__":
	run_test_model()
