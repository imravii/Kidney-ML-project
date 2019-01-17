

# import the necessary packages

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import train_test_split

from imutils import paths

import numpy as np

import argparse

import imutils

from sklearn import svm

import cv2

import matplotlib.pyplot as plt

import os
from sklearn.metrics import classification_report


def extract_color_histogram(image, bins=(8, 8, 8)):

	# extract a 3D color histogram from the HSV color space using

	# the supplied number of `bins` per channel

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,

		[0, 180, 0, 256, 0, 256])



	# handle normalizing the histogram if we are using OpenCV 2.4.X


	if imutils.is_cv2():

		hist = cv2.normalize(hist)



	# otherwise, perform "in place" normalization in OpenCV 3

	else:

		cv2.normalize(hist, hist)



	# return the flattened histogram as the feature vector

	return hist.flatten()


# grab the list of images that we'll be describing


imagePaths=paths.list_images(r'D:\ChinaSet_AllFiles\ChinaSet_AllFiles\CXR_png')


# initialize the features matrix,

# and labels list

features = []

labels = []



# loop over the input images

for (i, imagePath) in enumerate(imagePaths):

	# load the image and extract the class label 

	image = cv2.imread(imagePath)
    
	label = imagePath[-5]



	# extract a color

	# histogram to characterize the color distribution of the pixels

	# in the image


	hist = extract_color_histogram(image)


	# update features, and labels matricies,

	# respectively


	features.append(hist)

	labels.append(label)


# show some information on the memory consumed by the features matrix

features = np.array(features)

labels = np.array(labels)

print("[INFO] features matrix: {:.2f}MB".format(

	features.nbytes / (1024 * 1000.0)))



# partition the data into training and testing splits, using 75%

# of the data for training and the remaining 25% for testing

(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(

	features, labels, test_size=0.2, random_state=42)


#model = KNeighborsClassifier(n_neighbors=20,
#
#	n_jobs=-1)
#
#model.fit(trainRI, trainRL)


# train and evaluate a k-NN classifer on the histogram

# representations


from keras.models import Sequential
from keras.layers import Dense


print("[INFO] evaluating histogram accuracy...")


model = Sequential()
model.add(Dense(100,input_dim=features.shape[1],activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(trainFeat,trainLabels,epochs=60,batch_size=16,validation_data=(testFeat,testLabels),verbose=2)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#acc = model.score(testFeat, testLabels)
scores = model.evaluate(testFeat,testLabels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

score_feat=model.predict_classes(testFeat)
print("Detailed classification report")
print(classification_report(testLabels,score_feat))