# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import random
import time
from tqdm.notebook import tqdm
# from keras.preprocessing.image import load_img
# import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# %%
# TRAIN_DIR = 'dataset1/train'
# TEST_DIR = 'dataset1/test'
IMAGE_SIZE = 48
class_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
label_to_class = {v: k for k, v in class_labels.items()}

label_colors = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 255, 255),  # Yellow
    'fear': (255, 0, 0),       # Blue
    'happy': (0, 255, 0),      # Green
    'neutral': (255, 255, 0),  # Cyan
    'sad': (255, 0, 255),      # Magenta
    'surprise': (0, 255, 255)  # Light Blue

}

# model = tf.keras.models.load_model('cnnmodel.h5')
model = keras.models.load_model('cnnmodel.h5')
# %%
# test_paths = []
# test_labels = []

# for label in os.listdir(TEST_DIR):
#     if (label == '.DS_Store'):
#         continue
#     for img in os.listdir(os.path.join(TEST_DIR, label)):
#         test_paths.append(os.path.join(TEST_DIR, label, img))
#         test_labels.append(label_to_class[label])
#     print(f'{label}: done')

# %%
# test_labels = np.array(test_labels)
# test_images = []
# for path in tqdm(test_paths):
#     img = load_img(path,target_size=(IMAGE_SIZE, IMAGE_SIZE), color_mode='grayscale')
#     img = np.array(img)
#     test_images.append(img)
# test_images = np.array(test_images)
# test_images = test_images/255.0

# %%
# model.evaluate(test_images, test_labels)

# %%
# n = random.randint(0, test_images.shape[0]- 1)
# image = test_images[n].reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
# og_label = class_labels[test_labels[n]]
# pre_label = class_labels[model.predict(image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)).argmax()]
# plt.imshow(image[:,:,0], cmap='gray')
# plt.title(f'Original: {og_label}, Predicted: {pre_label}')
# plt.show()

# %%
import cv2

# %%
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# %%
cam = cv2.VideoCapture(0)
time.sleep(2)
while True:
    i, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

        
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
        face = np.array(face)       
        face = face/255.0
        face = face.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

        label = model.predict(face).argmax()
        label = class_labels[label]
        rectangle_color = label_colors.get(label, (0, 0, 0))
        cv2.rectangle(img, (x, y), (x+w, y+h), rectangle_color, 2)
        
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, rectangle_color, 2)
    cv2.imshow('img', img)
    cv2.waitKey(27)



