#20180805014 Ahmet Aydeniz
"""
Untitled1.ipynb

"""
import os
import shutil
from keras import layers
from keras import models
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas
import matplotlib.pyplot as plt
import tensorflow as tf
tf.keras.utils.get_custom_objects()
from keras.optimizers import Adam
from tensorflow import keras
from keras.models import load_model
from keras import optimizers
from tensorflow.keras.optimizers import RMSprop
original_Marino_dir='C:\\Users\\Aydeniz\\Desktop\deeplearning\\SheepFaceImages\\Marino'
original_PollDorset_dir='C:\\Users\\Aydeniz\\Desktop\\deeplearning\\SheepFaceImages\\PollDorset'
original_Suffolk_dir='C:\\Users\\Aydeniz\\Desktop\\deeplearning\\SheepFaceImages\\Suffolk'
original_WhiteSuffolk_dir='C:\\Users\\Aydeniz\\Desktop\deeplearning\\SheepFaceImages\\WhiteSuffolk'
base_dir=('C:\\Users\Aydeniz\Desktop\deepout')
if os.path.exists(base_dir):shutil.rmtree(base_dir)
os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
# Yeni Bölüm"""
train_Marino_dir = os.path.join(train_dir, 'Marino')
os.mkdir(train_Marino_dir)
train_PollDorset_dir = os.path.join(train_dir, 'PollDorset')
os.mkdir(train_PollDorset_dir)
train_Suffolk_dir = os.path.join(train_dir, 'Suffolk')
os.mkdir(train_Suffolk_dir)
train_WhiteSuffolk_dir = os.path.join(train_dir, 'WhiteSuffolk')
os.mkdir(train_WhiteSuffolk_dir)
validation_Marino_dir = os.path.join(validation_dir, 'Marino')
os.mkdir(validation_Marino_dir)
validation_PollDorset_dir = os.path.join(validation_dir, 'PollDorset')
os.mkdir(validation_PollDorset_dir)
validation_Suffolk_dir = os.path.join(validation_dir, 'Suffolk')
os.mkdir(validation_Suffolk_dir)
validation_WhiteSuffolk_dir = os.path.join(validation_dir, 'WhiteSuffolk')
os.mkdir(validation_WhiteSuffolk_dir)
test_Marino_dir = os.path.join(test_dir, 'Marino')
os.mkdir(test_Marino_dir)
test_PollDorset_dir = os.path.join(test_dir, 'PollDorset')
os.mkdir(test_PollDorset_dir)
test_Suffolk_dir = os.path.join(test_dir, 'Suffolk')
os.mkdir(test_Suffolk_dir)
test_WhiteSuffolk_dir = os.path.join(test_dir, 'WhiteSuffolk')
os.mkdir(test_WhiteSuffolk_dir)
### 1.KISIM - B - Resimleri oluşturduğum training, validation ve testklasörlerine gönderme
for i in range(1,301):source ="C:\\Users\\Aydeniz\\Desktop\\deeplearning\\SheepFaceImages\\Marino\\"+str(i)+".jpg"
destination ="C:\\Users\\Aydeniz\\Desktop\\deepout\\train\\Marino\\"+str(i)+".jpg"
shutil.copyfile(source, destination)
for i in range(301,361):source ="C:\\Users\\Aydeniz\\Desktop\\deeplearning\\SheepFaceImages\\Marino\\"+str(i)+".jpg"
destination ="C:\\Users\\Aydeniz\\Desktop\\deepout\\validation\\Marino\\"+str(i)+".jpg"
shutil.copyfile(source, destination)
for i in range(361,421):source ="C:\\Users\\Aydeniz\\Desktop\\deeplearning\\SheepFaceImages\\Marino\\"+str(i)+".jpg"
destination ="C:\\Users\\Aydeniz\\Desktop\\deepout\\test\\Marino\\"+str(i)+".jpg"
shutil.copyfile(source, destination)
#-----------------------------------------------------------------------------

for i in range(1,301):source ="C:\\Users\\Aydeniz\\Desktop\\deeplearning\\SheepFaceImages\\PollDorset\\"+str(i)+".jpg"
destination ="C:\\Users\\Aydeniz\\Desktop\\deepout\\train\\PollDorset\\"+str(i)+".jpg"
shutil.copyfile(source, destination)
for i in range(301,361):source ="C:\\Users\\Aydeniz\\Desktop\\deeplearning\\SheepFaceImages\\PollDorset\\"+str(i)+".jpg"
destination ="C:\\Users\\Aydeniz\\Desktop\\deepout\\validation\\PollDorset\\"+str(i)+".jpg"
shutil.copyfile(source, destination)
for i in range(361,421):source ="C:\\Users\\Aydeniz\\Desktop\\deeplearning\\SheepFaceImages\\PollDorset\\"+str(i)+".jpg"
destination ="C:\\Users\\Aydeniz\\Desktop\\deepout\\test\\PollDorset\\"+str(i)+".jpg"
shutil.copyfile(source, destination)
#-----------------------------------------------------------------------------

for i in range(1,301):source ="C:\\Users\\Aydeniz\\Desktop\\deeplearning\\SheepFaceImages\\Suffolk\\"+str(i)+".jpg"
destination ="C:\\Users\\Aydeniz\\Desktop\\deepout\\train\\Suffolk\\"+str(i)+".jpg"
shutil.copyfile(source, destination)
for i in range(301,361):source ="C:\\Users\\Aydeniz\\Desktop\\deeplearning\\SheepFaceImages\\Suffolk\\"+str(i)+".jpg"
destination ="C:\\Users\\Aydeniz\\Desktop\\deepout\\validation\\Suffolk\\"+str(i)+".jpg"
shutil.copyfile(source, destination)
for i in range(361,421):source ="C:\\Users\\Aydeniz\\Desktop\\deeplearning\\SheepFaceImages\\Suffolk\\"+str(i)+".jpg"
destination ="C:\\Users\\Aydeniz\\Desktop\\deepout\\test\\Suffolk\\"+str(i)+".jpg"
shutil.copyfile(source, destination)
#-----------------------------------------------------------------------------

for i in range(1,301):source ="C:\\Users\\Aydeniz\\Desktop\\deeplearning\\SheepFaceImages\\WhiteSuffolk\\"+str(i)+".jpg"
destination ="C:\\Users\\Aydeniz\\Desktop\\deepout\\train\\WhiteSuffolk\\"+str(i)+".jpg"
shutil.copyfile(source, destination)
for i in range(301,361):source ="C:\\Users\\Aydeniz\\Desktop\\deeplearning\\SheepFaceImages\\WhiteSuffolk\\"+str(i)+".jpg"
destination ="C:\\Users\\Aydeniz\\Desktop\\deepout\\validation\\WhiteSuffolk\\"+str(i)+".jpg"
shutil.copyfile(source, destination)
for i in range(361,421):source ="C:\\Users\\Aydeniz\\Desktop\\deeplearning\\SheepFaceImages\\WhiteSuffolk\\"+str(i)+".jpg"
destination ="C:\\Users\\Aydeniz\\Desktop\\deepout\\test\\WhiteSuffolk\\"+str(i)+".jpg"
shutil.copyfile(source, destination)
print('common sense baseline tum 4 classtaki veri sayısı esit = 420 olduguicin = 1/4')
print('total training Marino images:', len(os.listdir(train_Marino_dir)))
print('total valid Marino images:', len(os.listdir(validation_Marino_dir)))
print('total test Marino images:', len(os.listdir(test_Marino_dir)))
print('total training PollDorset images:',
len(os.listdir(train_PollDorset_dir)))
print('total valid PollDorset images:',
len(os.listdir(validation_PollDorset_dir)))
print('total test PollDorset images:', len(os.listdir(test_PollDorset_dir)))
print('total training Suffolk images:', len(os.listdir(train_Suffolk_dir)))
print('total valid Suffolk images:', len(os.listdir(validation_Suffolk_dir)))
print('total test Suffolk images:', len(os.listdir(test_Suffolk_dir)))
print('total training WhiteSuffolk images:',
len(os.listdir(train_WhiteSuffolk_dir)))
print('total valid WhiteSuffolk images:',
len(os.listdir(validation_WhiteSuffolk_dir)))
print('total test WhiteSuffolk images:',
len(os.listdir(test_WhiteSuffolk_dir)))
import cv2
from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(156, 181,
3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
optimizer = optimizers.RMSprop(lr=1e-4),
metrics =['acc'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(156, 181),
batch_size=20,
class_mode='categorical')
validation_generator=test_datagen.flow_from_directory(
validation_dir,
target_size=(156, 181),
batch_size=20,
class_mode='categorical')
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batc h shape:', labels_batch.shape)
    break
model.compile(loss = "categorical_crossentropy",
optimizer = RMSprop(learning_rate
=1e-4),
metrics = ["acc"])
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
for data_batch, labels_batch in validation_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
history = model.fit(
train_generator,
steps_per_epoch=50,
validation_data=validation_generator,
validation_steps=10,
epochs=28)
model.save('koyunlar')
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
datagen = ImageDataGenerator(
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')
from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150,
3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))
from keras import optimizers
train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(150, 150),
batch_size=32,
class_mode='categorical')
validation_generator=test_datagen.flow_from_directory(
validation_dir,
target_size=(150, 150),
batch_size=32,
class_mode='categorical')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
history = model.fit(
train_generator,steps_per_epoch=100,
epochs=28,
validation_data=validation_generator,
validation_steps=50)
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
model.save('koyunlar')
test_generator=test_datagen.flow_from_directory(
test_dir,
target_size=(150, 150),
batch_size=20,
class_mode='categorical')
test_loss, test_acc = model.evaluate(test_generator, steps=18)
print('test acc:', test_acc)

