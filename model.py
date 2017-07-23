import csv
import cv2
import numpy as np
import random
import gc

PATH = './data5'
NUMBER_IMAGES_LR = 1 # 1 for center image, 3 to use also left and right
FLIP = 2 # 2 to flip images, 1 normal

# Converts the image to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])[:, :, np.newaxis]
    
lines = []

with open(PATH + '/driving_log.csv')as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
X_train = np.array([])
cnt = 0

# To reduce the memory, I resize the array initially to prevent data duplication
# when copying the data, and I store the images directly in X_train
# I prefer this method because the generators are very slow, and the results are worse
X_train = np.zeros((len(lines)*FLIP*NUMBER_IMAGES_LR, 160,320,1), dtype=int)

for line in lines:
	for i in range(NUMBER_IMAGES_LR):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = PATH + '/IMG/' + filename
		image = cv2.imread(current_path)
		if image is not None:
			
			image = rgb2gray(image)

			X_train[cnt] = np.array(image)
			measurement = float(line[3])
			
			measurement = float(line[3])
			
			# correction for the left and right images
			correction = 0.20
			if i == 1:
				measurement = measurement - correction
			if i == 2:
				measurement = measurement + correction
			
			measurements.append(measurement)
			
			if FLIP == 2:
				## flipped image to consider the reverse move
				image2 = cv2.flip(image, 1)[:, :, np.newaxis]
				X_train[cnt+1] = np.array(image2)
				measurements.append(measurement*-1.0)

			cnt += FLIP

			if cnt % 1000 == 0:
				gc.collect()
				print (cnt)


#if False:
#	augmented_images, augmented_measurements = [], []
#	m = 8000
#	for image, measurement in zip(images, measurements):
#		augmented_images.append(image)
#		augmented_measurements.append(measurement)
#	for image, measurement in random.sample(list(zip(images, measurements)), m):
#		augmented_images.append(cv2.flip(image, 1))
#		augmented_measurements.append(measurement*-1.0)

#X_train = np.array(X_train.tolist())
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D, Activation, merge, Input
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam

def LeNetModel(model):
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((70,25),(0,0))))
	model.add(Convolution2D(6,5,5, activation="relu"))
	model.add(MaxPooling2D())
	model.add(Dropout(0.25))
	model.add(Convolution2D(6,5,5, activation="relu"))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')

def NVIDIAModel(model):
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,1)))
	model.add(Cropping2D(cropping=((70,25),(0,0))))
	model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
	model.add(Dropout(0.25))
	model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
	model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
	model.add(Dropout(0.25))
	model.add(Convolution2D(64,3,3, activation="relu"))
	model.add(Convolution2D(64,3,3, activation="relu"))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dropout(0.5))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer=Adam(lr=0.00001))

def DuelingNetworkModel():
	input = Input(shape=(160, 320, 1))
		
	#state_value = Sequential()
	x = (Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(160, 320, 1)))(input)
	x = (Activation('relu'))(x)
	x = (Convolution2D(64, 4, 4, subsample=(2, 2)))(x)
	x = (Activation('relu'))(x)
	x = (Convolution2D(64, 3, 3))(x)
	x = (Activation('relu'))(x)
	x = (Flatten())(x)
	state_value = (Dense(512))(x)
	state_value = (Activation('relu'))(state_value)
	state_value = (Dense(1))(state_value)
	#model.compile(loss='mse', optimizer=Adam(lr=0.00001))

	#target_model = Sequential()
	target_model = (Dense(512))(x)
	target_model = (Activation('relu'))(target_model)
	target_model = (Dense(1))(target_model)
	#target_model.compile(loss='mse', optimizer=Adam(lr=0.00001))
	#target_model.set_weights(model.get_weights())
	
	state_action_value = merge([state_value, target_model], mode='sum')
	
	model = Model(input=input, output=state_action_value)
	model.compile(loss='mse', optimizer='adam')


model = Sequential()

NVIDIAModel(model)

model.fit(X_train, y_train, validation_split=0.20, shuffle=True, nb_epoch=120)

model.save('model.h5')

