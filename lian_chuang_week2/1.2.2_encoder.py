import keras
from keras.layers import Dense, Input
from keras.datasets import mnist
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

x_train_nosiy = x_train + 0.1 * \
    np.random.normal(loc=0., scale=1., size=x_train.shape)
x_train_nosiy = np.clip(x_train_nosiy, 0., 1.)


input_img = Input(shape=(784,))

encode = Dense(128, activation='relu')(input_img)
encode = Dense(64, activation='relu')(encode)
encode = Dense(10, activation='relu')(encode)
encoder_output = Dense(3)(encode)

decode = Dense(10, activation='relu')(encoder_output)
decode = Dense(64, activation='relu')(decode)
decode = Dense(128, activation='relu')(decode)
decode = Dense(784, activation='sigmoid')(decode)

auto_encoder = Model(inputs=input_img, outputs=decode)

encoder = Model(inputs=input_img, outputs=encoder_output)

auto_encoder.compile(optimizer='adam', loss='mse')

auto_encoder.fit(x_train_nosiy, x_train, epochs=120, batch_size=256, shuffle=True)


results = encoder.predict(x_test)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(results[:, 0], results[:, 1],
             results[:, 2], c=y_test, s=5, alpha=0.5)
plt.show()
