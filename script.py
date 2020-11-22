#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rickysu
"""

import numpy as np
import os

# TODO
'''
Yes, there's a lot you can do to clean this up, like modularizing it or making
functions. This is just a script to get you started. It's not supposed to be pretty.
When's the last time someone handed 'pretty' code to you, huh?
'''

# Exploratory
'''
See number of samples
After clustering, how many samples in each cluster
Clusters with few samples should augment their samples with ImageDataGenerator
    https://keras.io/api/preprocessing/image/#imagedatagenerator-class
    from keras.preprocessing.image import ImageDataGenerator
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
'''

# Select and create CNN model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

model = VGG16(weights='imagenet', include_top=False)
# full_model = VGG16(weights='imagenet') # this is just for fun


# Separate images to train and val
folder = "./train_small/"
train_files = os.listdir(folder)


# Load the training images
train_loaded = []
for file_name in train_files:
    img_path = folder + file_name
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_loaded.append(x)


# Extract features (flatten also)
train_features = []
for loaded_file in train_loaded:
    train_features.append(np.ndarray.flatten(model.predict(loaded_file)))


# Select and create clustering model
from sklearn.cluster import KMeans
from scipy import stats

# Always 42!
kmeans = KMeans(n_clusters=2, random_state=42).fit(train_features)


# Get the predicted cluster label for dogs and cats
dog_clusters = []
cat_clusters = []

for i in range(len(train_features)):
    current_file = train_files[i]
    feature = train_features[i]
    predicted_cluster = kmeans.predict([feature])
    if 'dog' in current_file:
        dog_clusters.append(predicted_cluster)
    else:
        cat_clusters.append(predicted_cluster)

dog_cluster = stats.mode(dog_clusters)[0][0][0]
cat_cluster = stats.mode(cat_clusters)[0][0][0]
print("Dog cluster is: ", str(dog_cluster))
print("Cat cluster is: ", str(cat_cluster))


# Create true cluster labels (based on name of the file)
true_labels = []
for current_file in train_files:
    if 'dog' in current_file:
        true_labels.append(dog_cluster)
    else:
        true_labels.append(cat_cluster)


# Create predicted cluster labels
predicted_labels = []
for feature in train_features:
    predicted_labels.append(kmeans.predict([feature])[0])


# Get accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(true_labels,predicted_labels))

