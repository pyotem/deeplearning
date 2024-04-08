
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

original_datasetbird_dir='C:\\hayvanlar\\animall\\animal_images\\bird'
original_datasetcat_dir='C:\\hayvanlar\\animall\\animal_images\\cat'
original_datasetdog_dir='C:\\hayvanlar\\animall\\animal_images\\dog'
original_datasetfish_dir='C:\\hayvanlar\\animall\\animal_images\\fish'
original_datasetrabbit_dir='C:\\hayvanlar\\animall\\animal_images\\rabbit'
original_datasetother_dir='C:\\hayvanlar\\animall\\animal_images\\other'
base_dir=('C:\\hayvanlar\\animall\\base')
if os.path.exists(base_dir):
 shutil.rmtree(base_dir)

os.mkdir(base_dir)



train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

#111111111111111111111111111111111111111111111111111111

train_bird_dir = os.path.join(train_dir, 'train_bird')
os.mkdir(train_bird_dir)

train_cat_dir = os.path.join(train_dir, 'train_cat')
os.mkdir(train_cat_dir)

train_dog_dir = os.path.join(train_dir, 'train_dog')
os.mkdir(train_dog_dir)

train_fish_dir = os.path.join(train_dir, 'train_fish')
os.mkdir(train_fish_dir)

train_rabbit_dir = os.path.join(train_dir, 'train_rabbit')
os.mkdir(train_rabbit_dir)

train_other_dir = os.path.join(train_dir, 'train_other')
os.mkdir(train_other_dir)

#111111111111111111111111111111111111111111111111111111

validation_bird_dir = os.path.join(validation_dir, 'validation_bird')
os.mkdir(validation_bird_dir)

validation_cat_dir = os.path.join(validation_dir, 'validation_cat')
os.mkdir(validation_cat_dir)

validation_dog_dir = os.path.join(validation_dir, 'validation_dog')
os.mkdir(validation_dog_dir)

validation_fish_dir = os.path.join(validation_dir, 'validation_fish')
os.mkdir(validation_fish_dir)

validation_rabbit_dir = os.path.join(validation_dir, 'validation_rabbit')
os.mkdir(validation_rabbit_dir)

validation_other_dir = os.path.join(validation_dir, 'validation_other')
os.mkdir(validation_other_dir)


#111111111111111111111111111111111111111111111111111111

test_bird_dir = os.path.join(test_dir, 'test_bird')
os.mkdir(test_bird_dir)

test_cat_dir = os.path.join(test_dir, 'test_cat')
os.mkdir(test_cat_dir)

test_dog_dir = os.path.join(test_dir, 'test_dog')
os.mkdir(test_dog_dir)

test_fish_dir = os.path.join(test_dir, 'test_fish')
os.mkdir(test_fish_dir)

test_rabbit_dir = os.path.join(test_dir, 'test_rabbit')
os.mkdir(test_rabbit_dir)

test_other_dir = os.path.join(test_dir, 'test_other')
os.mkdir(test_other_dir)




#111111111111111111111111111111111111111111111111111111


fnames = ['{}.jpg' .format(i) for i in range(1,600)]
for fname in fnames:
  src = os.path.join(original_datasetbird_dir, fname)
  dst = os.path.join(train_bird_dir, fname)
  shutil.copyfile(src, dst)
fnames = ['{}.jpg' .format(i) for i in range(600, 685)]
for fname in fnames:
  src = os.path.join(original_datasetbird_dir, fname)
  dst = os.path.join(validation_bird_dir, fname)
  shutil.copyfile(src, dst)
fnames = ['{}.jpg' .format(i) for i in range(685,733)]
for fname in fnames:
  src = os.path.join(original_datasetbird_dir ,fname)
  dst = os.path.join(test_bird_dir, fname)
  shutil.copyfile(src, dst)


#111111111111111111111111111111111111111111111111111111


fnames = ['{}.jpg' .format(i) for i in range(1,140)]
for fname in fnames:
  src = os.path.join(original_datasetcat_dir, fname)
  dst = os.path.join(train_cat_dir, fname)
  shutil.copyfile(src, dst)
fnames = ['{}.jpg' .format(i) for i in range(140, 180)]
for fname in fnames:
  src = os.path.join(original_datasetcat_dir, fname)
  dst = os.path.join(validation_cat_dir, fname)
  shutil.copyfile(src, dst)
fnames = ['{}.jpg' .format(i) for i in range(180,218)]
for fname in fnames:
  src = os.path.join(original_datasetcat_dir, fname)
  dst = os.path.join(test_cat_dir, fname)
  shutil.copyfile(src, dst)


#111111111111111111111111111111111111111111111111111111


fnames = ['{}.jpg' .format(i) for i in range(1,2600)]
for fname in fnames:
  src = os.path.join(original_datasetdog_dir, fname)
  dst = os.path.join(train_dog_dir, fname)
  shutil.copyfile(src, dst)
fnames = ['{}.jpg' .format(i) for i in range(2600, 3100)]
for fname in fnames:
  src = os.path.join(original_datasetdog_dir, fname)
  dst = os.path.join(validation_dog_dir, fname)
  shutil.copyfile(src, dst)
fnames = ['{}.jpg' .format(i) for i in range(3100,3534)]
for fname in fnames:
  src = os.path.join(original_datasetdog_dir, fname)
  dst = os.path.join(test_dog_dir, fname)
  shutil.copyfile(src, dst)


#111111111111111111111111111111111111111111111111111111




fnames = ['{}.jpg' .format(i) for i in range(1,1450)]
for fname in fnames:
  src = os.path.join(original_datasetfish_dir, fname)
  dst = os.path.join(train_fish_dir, fname)
  shutil.copyfile(src, dst)
fnames = ['{}.jpg' .format(i) for i in range(1450, 1600)]
for fname in fnames:
  src = os.path.join(original_datasetfish_dir, fname)
  dst = os.path.join(validation_fish_dir, fname)
  shutil.copyfile(src, dst)
fnames = ['{}.jpg' .format(i) for i in range(1600,1719)]
for fname in fnames:
  src = os.path.join(original_datasetfish_dir, fname)
  dst = os.path.join(test_fish_dir, fname)
  shutil.copyfile(src, dst)


#111111111111111111111111111111111111111111111111111111




fnames = ['{}.jpg' .format(i) for i in range(1,300)]
for fname in fnames:
  src = os.path.join(original_datasetrabbit_dir, fname)
  dst = os.path.join(train_rabbit_dir, fname)
  shutil.copyfile(src, dst)
fnames = ['{}.jpg' .format(i) for i in range(300, 400)]
for fname in fnames:
  src = os.path.join(original_datasetrabbit_dir, fname)
  dst = os.path.join(validation_rabbit_dir, fname)
  shutil.copyfile(src, dst)
fnames = ['{}.jpg' .format(i) for i in range(400,470)]
for fname in fnames:
  src = os.path.join(original_datasetrabbit_dir, fname)
  dst = os.path.join(test_rabbit_dir, fname)
  shutil.copyfile(src, dst)


#111111111111111111111111111111111111111111111111111111




fnames = ['{}.jpg' .format(i) for i in range(1,4300)]
for fname in fnames:
  src = os.path.join(original_datasetother_dir, fname)
  dst = os.path.join(train_other_dir, fname)
  shutil.copyfile(src, dst)
fnames = ['{}.jpg' .format(i) for i in range(4300, 4900)]
for fname in fnames:
  src = os.path.join(original_datasetother_dir, fname)
  dst = os.path.join(validation_other_dir, fname)
  shutil.copyfile(src, dst)
fnames = ['{}.jpg' .format(i) for i in range(4900,5285)]
for fname in fnames:
  src = os.path.join(original_datasetother_dir, fname)
  dst = os.path.join(test_other_dir, fname)
  shutil.copyfile(src, dst)


#111111111111111111111111111111111111111111111111111111

print('total training building images:', len(os.listdir(train_bird_dir)))
print('total training building images:', len(os.listdir(train_cat_dir)))
print('total training building images:', len(os.listdir(train_dog_dir)))
print('total training building images:', len(os.listdir(train_fish_dir)))
print('total training building images:', len(os.listdir(train_rabbit_dir)))
print('total training building images:', len(os.listdir(train_other_dir)))





print('total valid building images:', len(os.listdir(validation_bird_dir)))
print('total valid building images:', len(os.listdir(validation_cat_dir)))
print('total valid building images:', len(os.listdir(validation_dog_dir)))
print('total valid building images:', len(os.listdir(validation_fish_dir)))
print('total valid building images:', len(os.listdir(validation_rabbit_dir)))
print('total valid building images:', len(os.listdir(validation_other_dir)))



print('total test building images:', len(os.listdir(test_bird_dir)))
print('total test building images:', len(os.listdir(test_cat_dir)))
print('total test building images:', len(os.listdir(test_dog_dir)))
print('total test building images:', len(os.listdir(test_fish_dir)))
print('total test building images:', len(os.listdir(test_rabbit_dir)))
print('total test building images:', len(os.listdir(test_other_dir)))



from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(142, 107, 3)))

model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))
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
target_size=(142, 107),
batch_size=64,
class_mode='categorical')

validation_generator=test_datagen.flow_from_directory(
validation_dir,
target_size=(142, 107),
batch_size=64,
class_mode='categorical')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit(
train_generator,
steps_per_epoch=100,
epochs=10,
validation_data=validation_generator,
validation_steps=80)

model.save('hayvanlar')

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
steps_per_epoch=100,
epochs=10,
validation_data=validation_generator,
validation_steps=80)

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

model.save('asiavehayvanfil_cesitleri')

test_generator=test_datagen.flow_from_directory(
test_dir,
target_size=(142, 107),
batch_size=64,
class_mode='categorical')

test_loss, test_acc = model.evaluate(test_generator, steps=18)
print('test acc:', test_acc)
