
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
original_datasetAfrican_dir = 'C:\\AsianAfrican\\train\\African'
original_datasetAsian_dir = 'C:\\AsianAfrican\\train\\Asian'
base_dir = 'C:\\AsianAfrican\\base'
if os.path.exists(base_dir):
  shutil.rmtree(base_dir)

os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_African_dir = os.path.join(train_dir, 'African')
os.mkdir(train_African_dir)

train_Asian_dir = os.path.join(train_dir, 'Asian')
os.mkdir(train_Asian_dir)

validation_African_dir = os.path.join(validation_dir, 'African')
os.mkdir(validation_African_dir)

validation_Asian_dir = os.path.join(validation_dir, 'Asian')
os.mkdir(validation_Asian_dir)

test_African_dir = os.path.join(test_dir, 'African')
os.mkdir(test_African_dir)

test_Asian_dir = os.path.join(test_dir, 'Asian')
os.mkdir(test_Asian_dir)

fnames = ['{}.jpg'.format(i) for i in range(1,351)]
for fname in fnames:
    src = os.path.join(original_datasetAfrican_dir, fname)
    dst = os.path.join(train_African_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(351, 451)]
for fname in fnames:
    src = os.path.join(original_datasetAfrican_dir, fname)
    dst = os.path.join(validation_African_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(451, 518)]
for fname in fnames:
    src = os.path.join(original_datasetAfrican_dir, fname)
    dst = os.path.join(test_African_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(1,351)]
for fname in fnames:
    src = os.path.join(original_datasetAsian_dir, fname)
    dst = os.path.join(train_Asian_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['{}.jpg'.format(i) for i in range(351, 451)]
for fname in fnames:
    src = os.path.join(original_datasetAsian_dir, fname)
    dst = os.path.join(validation_Asian_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(451, 512)]
for fname in fnames:
    src = os.path.join(original_datasetAsian_dir, fname)
    dst = os.path.join(test_Asian_dir, fname)
    shutil.copyfile(src, dst)

print('total training asian images:', len(os.listdir(train_Asian_dir)))
print('total training african images:', len(os.listdir(train_African_dir)))

print('total valid asian images:', len(os.listdir(validation_Asian_dir)))
print('total valid african images:', len(os.listdir(validation_African_dir)))

print('total test asian images:', len(os.listdir(test_Asian_dir)))
print('total test african images:', len(os.listdir(test_African_dir)))


from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 194, 3)))

model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.summary()

from keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(256, 194),
batch_size=20,
class_mode='categorical')

validation_generator=test_datagen.flow_from_directory(
validation_dir,
target_size=(256, 194),
batch_size=20,
class_mode='categorical')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit(
train_generator,
steps_per_epoch=30,
epochs=17,
validation_data=validation_generator,
validation_steps=15)

model.save('Alikaya')

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

history = model.fit(
train_generator,
steps_per_epoch=30,
epochs=17,
validation_data=validation_generator,
validation_steps=15)

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

model.save('asiaveafricanfil_cesitleri')

test_generator=test_datagen.flow_from_directory(
test_dir,
target_size=(259, 194),
batch_size=20,
class_mode='categorical')

test_loss, test_acc = model.evaluate(test_generator, steps=18)
print('test acc:', test_acc)