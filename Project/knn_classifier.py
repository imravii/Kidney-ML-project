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


def image_to_feature_vector(image, size=(32, 32)):

	# resize the image to a fixed size, then flatten the image into

	# a list of raw pixel intensities

	return cv2.resize(image, size).flatten()



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


print("[INFO] describing images...")

imagePaths=paths.list_images(r'D:\ChinaSet_AllFiles\ChinaSet_AllFiles\CXR_png')


# initialize the raw pixel intensities matrix, the features matrix,

# and labels list


rawImages = []

features = []

labels = []



# loop over the input images

for (i, imagePath) in enumerate(imagePaths):

	# load the image and extract the class label 


	image = cv2.imread(imagePath)
    
	label = imagePath[-5]



	# extract raw pixel intensity "features", followed by a color

	# histogram to characterize the color distribution of the pixels

	# in the image


	pixels = image_to_feature_vector(image)

	hist = extract_color_histogram(image)



	# update the raw images, features, and labels matricies,

	# respectively


	rawImages.append(pixels)

	features.append(hist)

	labels.append(label)


# show some information on the memory consumed by the raw images

# matrix and features matrix


rawImages = np.array(rawImages)

features = np.array(features)

labels = np.array(labels)


print("[INFO] pixels matrix: {:.2f}MB".format(

	rawImages.nbytes / (1024 * 1000.0)))

print("[INFO] features matrix: {:.2f}MB".format(

	features.nbytes / (1024 * 1000.0)))



# partition the data into training and testing splits, using 80%

# of the data for training and the remaining 20% for testing


(trainRI, testRI, trainRL, testRL) = train_test_split(

	rawImages, labels, test_size=0.2, random_state=42)

(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(

	features, labels, test_size=0.2, random_state=42)



# train and evaluate a k-NN classifer on the raw pixel intensities


print("[INFO] evaluating raw pixel accuracy...")

model = KNeighborsClassifier(n_neighbors=20,
	n_jobs=-1)

model.fit(trainRI, trainRL)

acc = model.score(testRI, testRL)

score_ri=model.predict_classes(testRI)

print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
print("Detailed classification report")
print(classification_report(testRL,score_ri))

# train and evaluate a k-NN classifer on the histogram

# representations

print("[INFO] evaluating histogram accuracy...")

model = KNeighborsClassifier(n_neighbors=20,
	n_jobs=-1)

model.fit(trainFeat, trainLabels)

acc = model.score(testFeat, testLabels)

score_feat=model.predict_classes(testFeat)
print("Detailed classification report")
print(classification_report(testLabels,score_feat))
