import keras
from keras.layers import *
from keras.models import *
from keras import backend as K
from keras.regularizers import l2
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


num_classes = 10
num_dense_block = 2

growth_rate = 16
num_filter = 32

dropout_rate = 0.2
weight_decay = 1e-4
subsample_initial_block = True
input_shape = (32, 32, 3)


def train(model):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_test /= 255
    train_datagan = ImageDataGenerator(rescale=1./255, )

    callbacks = [keras.callbacks.TensorBoard(log_dir='my_log')]

    history = model.fit_generator(train_datagan.flow(x_train, y_train, batch_size=32),
                                  steps_per_epoch=25, epochs=1,
                                  validation_data=(x_test, y_test),
                                  shuffle=True,
                                  callbacks=callbacks)

    history = history.history
    acc = history['acc']
    loss = history['loss']
    epochs = range(1, len(acc)+1)
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.legend()
    plt.savefig('densenet_acc.png')
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.legend()
    plt.savefig('densenet_loss.png')


def conv_block(ip, num_filter, dropout_rate=None, weight_decay=1e-4):

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    x = Conv2D(num_filter, (3, 3), kernel_initializer='he_normal',
               padding='same', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(ip, num_filter, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    x = Conv2D(int(num_filter * 0.5), (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def build_model():
    num_filter = 32
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filter, (3, 3), kernel_initializer='he_normal', padding='same',
               strides=1,  kernel_regularizer=l2(weight_decay))(inputs)

    for _ in range(3):
        cb = conv_block(x, growth_rate, dropout_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        num_filter += growth_rate

    x = transition_block(
        x, num_filter,  weight_decay=weight_decay)
    num_filter = int(num_filter * 0.7)

    for _ in range(3):
        cb = conv_block(x, growth_rate, dropout_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        num_filter += growth_rate

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, x)

    model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  metrics=['acc'])
    return model


model = build_model()

train(model)
