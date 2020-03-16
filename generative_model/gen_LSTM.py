
import sys
import random
from keras import layers
import keras
import numpy as np

path = keras.utils.get_file(
    'nietzsche,txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

text = open(path).read().lower()
print('Corpus len: ', len(text))

# extract a sequence, whose length is maxlen
# apply one-hot encoding to it
# and pack it into a 3D tensor:
# (sequence, maxlen, unique_char)
# and prepare a vec y, containing targets

maxlen = 80     # text len per sample
step = 3        # step between samples
sentences = []
next_chars = []

for i in range(0, len(text)-maxlen, step):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])

print('num of sentences: ', len(sentences))

unique_chars = sorted(list(set(text)))
char_indices = dict((char, unique_chars.index(char))
                    for char in unique_chars)

print('num of unique characters: ', len(unique_chars))
print(unique_chars)

# one-hot
x = np.zeros((len(sentences), maxlen, len(unique_chars)),
             dtype=np.bool)
y = np.zeros((len(sentences), len(unique_chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


### build model ###

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(unique_chars))))
model.add(layers.Dense(len(unique_chars), activation='softmax'))
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


## reweight the original P ##

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds+1e-7)/（temperature+1e-7）
    exp_preds = np.exp(preds)
    preds = exp_preds/(np.sum(exp_preds)+1e-7)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


random.seed(7)


model.fit(x, y, batch_size=128, epochs=60)
model.save('/root/MySource/gen_LSTM_60.h5')

for sent in range(10):
    start_index = random.randint(0, len(text)-maxlen-1)
    generated_text = text[start_index:start_index+maxlen]
    print()
    print('---generate with: "', generated_text, '"')
    for temperature in [0.2, 0.5, 1.0]:
        print()
        print('------temperature: ', temperature)
        sys.stdout.write(generated_text)

        for i in range(500):
            sampled = np.zeros((1, maxlen, len(unique_chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = unique_chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
