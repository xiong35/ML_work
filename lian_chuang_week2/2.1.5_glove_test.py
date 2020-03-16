
from keras.models import load_model
import keras.backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Reshape, Add
import numpy as np
import pandas as pd

word2id = pd.read_csv(R'C:\Users\xiong35\Desktop\word2id.csv')
id2word = pd.read_csv(R'C:\Users\xiong35\Desktop\id2word.csv')
num_of_words = word2id.shape[1]


def glove_model(vocab_size=num_of_words, vector_dim=64):

    input_target = Input((1,), name='central_word_id')
    input_context = Input((1,), name='context_word_id')

    central_embedding = Embedding(
        vocab_size, vector_dim, input_length=1, name='central_embedding')
    central_bias = Embedding(
        vocab_size, 1, input_length=1, name='central_bias')

    context_embedding = Embedding(
        vocab_size, vector_dim, input_length=1, name='context_embedding')
    context_bias = Embedding(
        vocab_size, 1, input_length=1, name='context_bias')

    vector_target = central_embedding(input_target)
    vector_context = context_embedding(input_context)

    bias_target = central_bias(input_target)
    bias_context = context_bias(input_context)

    dot_product = Dot(axes=-1)([vector_target, vector_context])
    dot_product = Reshape((1,))(dot_product)
    bias_target = Reshape((1,))(bias_target)
    bias_context = Reshape((1,))(bias_context)

    prediction = Add()([dot_product, bias_target, bias_context])

    model = Model(inputs=[input_target, input_context], outputs=prediction)
    model.compile(loss=my_loss, optimizer=Adam())

    model.summary()
    return model


def my_loss(y_true, y_pred):
    return K.sum(K.pow(K.clip(y_true / 100, 0.0, 1.0), 3/4) *
                 K.square(y_pred - K.log(y_true+1e-6)), axis=-1)


model = glove_model()
model.load_weights('my_glove_w.h5')

embeddings = model.get_weights()[0]+model.get_weights()[1]
normalized_embeddings = embeddings / \
    (embeddings**2).sum(axis=1).reshape((-1, 1))**0.5


def most_similar(word, k=10):
    v = normalized_embeddings[int(word2id[word])]
    sims = np.dot(normalized_embeddings, v)
    sort = sims.argsort()[::-1]
    sort = sort[sort > 0]
    return [(id2word[str(i)], sims[i]) for i in sort[:k]]


def calculate(word_list):
    v0_id = word2id[word_list[0]]
    v1_id = word2id[word_list[1]]
    v2_id = word2id[word_list[2]]
    v0 = embeddings[int(v0_id)]
    v1 = embeddings[int(v1_id)]
    v2 = embeddings[int(v2_id)]
    v_pred = v0 - v1 + v2
    sims = np.dot(normalized_embeddings, v_pred)
    sort = sims.argsort()[::-1]
    sort = sort[sort > 0]
    return [(id2word[str(i)], sims[i]) for i in sort[:10]]


def predict():
    while True:
        word = input("enter a word: ")
        if word == 'q':
            break
        if word in word2id:
            print(pd.Series(most_similar(word)))
        else:
            print('not in')
    while True:
        input_words = input('enter 3 words:\nI\'ll show U w0 - w1 +w2: ')
        input_words = input_words.split()
        wrong = False
        for word in input_words:
            if word not in word2id:
                print(word, ' not in')
                wrong = True
        if not wrong:
            print(pd.Series(calculate(input_words)))


predict()
