
# 用LSTM生成尼采风格文章 

## 引入相关依赖

    import sys
    import random
    from keras import layers
    import keras
    import numpy as np

## 下载并处理语料

    path = keras.utils.get_file(
        'nietzsche,txt',
        origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

    text = open(path).read().lower()
    print('Corpus len: ', len(text))

## 将语料处理成模型输入

    # extract a sequence, whose length is maxlen
    # apply one-hot encoding to it
    # and pack it into a 3D tensor:
    # (sequence, maxlen, unique_char)
    # and prepare a vec y, containing targets
    maxlen = 80     # text len per sample
    step = 3        # step between samples
    sentences = []
    next_chars = []

将语料提取为一组句子

    for i in range(0, len(text)-maxlen, step):
        sentences.append(text[i:i+maxlen])
        next_chars.append(text[i+maxlen])

    print('num of sentences: ', len(sentences))

找到所有不同的字符

    unique_chars = sorted(list(set(text)))
    char_indices = dict((char, unique_chars.index(char))
                        for char in unique_chars)

    print('num of unique characters: ', len(unique_chars))
    print(unique_chars)

将字符进行one-hot编码

    # one-hot
    x = np.zeros((len(sentences), maxlen, len(unique_chars)),
                dtype=np.bool)
    y = np.zeros((len(sentences), len(unique_chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

## 建立模型

    ### build model ###
    model = keras.models.Sequential()
    model.add(layers.LSTM(128, input_shape=(maxlen, len(unique_chars))))
    model.add(layers.Dense(len(unique_chars), activation='softmax'))
    optimizer = keras.optimizers.RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

为了让输出不那么死板，我们调整输出方式，使其有一些随机性

    ## reweight the original P ##
    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds+1e-7)/（temperature+1e-7）
        exp_preds = np.exp(preds)
        preds = exp_preds/(np.sum(exp_preds)+1e-7)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

## 训练并保存模型

    model.fit(x, y, batch_size=128, epochs=60)
    model.save('/root/MySource/gen_LSTM_60.h5')

## 让模型进行输出

    random.seed(7)
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

部分结果如下：

![same](http://q5ioolwed.bkt.clouddn.com/samesamesame.png)

可见尼采很喜欢same这个词。。。。  
看点靠谱的！  

![transleted_gen](http://q5ioolwed.bkt.clouddn.com/translated_gen.png)

写的其实还蛮好的哈哈哈哈