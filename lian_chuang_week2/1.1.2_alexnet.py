import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import os
import shutil
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Lambda, Conv2D, MaxPool2D, BatchNormalization, Dense, Flatten, Dropout
from keras.models import Model
from keras.utils import to_categorical

original_dataset_dir = R'C:\Users\xiong35\Desktop\train\train'

base_dir = R'C:\Users\xiong35\Desktop\small_dir'
# os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
# os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
# os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
# os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
# os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
# os.mkdir(test_dogs_dir)

# fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_cats_dir, fname)
#     shutil.copyfile(src, dst)

# fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_cats_dir, fname)
#     shutil.copyfile(src, dst)

# fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_cats_dir, fname)
#     shutil.copyfile(src, dst)


# fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_dogs_dir, fname)
#     shutil.copyfile(src, dst)

# fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_dogs_dir, fname)
#     shutil.copyfile(src, dst)

# fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_dogs_dir, fname)
#     shutil.copyfile(src, dst)


print("total training cat images:", len(os.listdir(train_cats_dir)))
print("total training dog images:", len(os.listdir(train_dogs_dir)))
print("total validation cat images:", len(os.listdir(validation_cats_dir)))
print("total validation dog images:", len(os.listdir(validation_dogs_dir)))
print("total test cat images:", len(os.listdir(test_cats_dir)))
print("total test dog images:", len(os.listdir(test_dogs_dir)))


################我是分割线################


inputs = Input((150, 150, 3))
Conv_1 = Conv2D(48, (11, 11), strides=4, activation='relu',
                kernel_initializer='uniform', padding='valid')(inputs)
BN_1 = BatchNormalization()(Conv_1)
MaxPool_1 = MaxPool2D((3, 3), strides=2, padding='valid')(BN_1)

Conv_2 = Conv2D(128, (5, 5), strides=1, padding='same',
                activation='relu', kernel_initializer='uniform')(MaxPool_1)
BN_2 = BatchNormalization()(Conv_2)
MaxPool_2 = MaxPool2D((3, 3), strides=2, padding='valid')(BN_2)

Conv_3 = Conv2D(192, (3, 3), strides=1, padding='same',
                activation='relu', kernel_initializer='uniform')(MaxPool_2)
Conv_4 = Conv2D(192, (3, 3), strides=1, padding='same',
                activation='relu', kernel_initializer='uniform')(Conv_3)
Conv_5 = Conv2D(128, (3, 3), strides=1, padding='same',
                activation='relu', kernel_initializer='uniform')(Conv_4)
MaxPool_3 = MaxPool2D((3, 3), strides=2, padding='valid')(Conv_5)

Flatten_1 = Flatten()(MaxPool_3)
Dense_1 = Dense(256, activation='relu')(Flatten_1)
Dropout_1 = Dropout(0.5)(Dense_1)

Dense_2 = Dense(256, activation='relu')(Dropout_1)
Dropout_2 = Dropout(0.5)(Dense_2)
outputs = Dense(1, activation='sigmoid')(Dropout_2)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('my_alexnet.h5', verbose=1, save_best_only=True)

callbacks = [earlystopper, checkpointer]
model.summary()
input()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])


train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=callbacks)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
