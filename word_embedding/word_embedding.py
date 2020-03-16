
from keras.models import Sequential
from keras import preprocessing
from keras.datasets import imdb
from keras.layers import Embedding, Dense, Flatten

# num of words, dimension to embed
embedding_layer = Embedding(1000, 64)

# consider embedding layer as a dictionary
# give it the index of the word
# it returns the vector of the word

# embedding layer takes a 2D tensor: (samples, sequence_length)
# in one batch, the length of the sequence shuld be the same
# the short ones are embeded with 0

# embedding layer returns a 3D tensor:
# (samples, sequence_length, dimensionality)


# train imdb embedding

max_features = 10000
max_len = 20

(x_train, y_train), (x_test, y_test) = \
    imdb.load_data(num_words=max_features)

# change the int list into a (samples, max_len) tensor
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(10000, 8, input_length=max_len))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# model.summary()

history = model.fit(x_train, y_train, epochs=10,
                    batch_size=32, validation_split=0.2)
