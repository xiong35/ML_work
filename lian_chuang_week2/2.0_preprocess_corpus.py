
import os
import pandas as pd


def get_stopwords():
    stopwords = []
    stopwords = [' '+line.strip()+' ' for line in open(
        R'C:\Users\xiong35\Desktop\corpus\stopped_word.txt').readlines()]
    return stopwords


def preprocess(fname, new_file):
    f = open(fname, 'r')
    lines = f.readlines()
    new_f = open(new_file, 'a')
    stopwords = get_stopwords()
    to_replace = [',', '.', '-', '_', '?', '!', ':',
                  '”', '“', '’', '‘', '[', ']', '–', ';']
    for i, line in enumerate(lines):
        if i % 10 == 0:
            print('{} is done'.format(i))
            new_f.write('\n')
        line = line.strip('\n').lower()+' '
        for word in stopwords:
            if word in line:
                line = line.replace(word, ' ')
        for mark in to_replace:
            line = line.replace(mark, ' ')
        for num in '0123456789':
            line = line.replace(num, '')
        new_f.write(line)
    f.close()
    new_f.close()


fname = R'C:\Users\xiong35\Desktop\corpus\Holmose.txt'
new_file = R'C:\Users\xiong35\Desktop\corpus\new_h.txt'

if not os.path.exists(new_file):
    preprocess(fname, new_file)


corpus = R'C:\Users\xiong35\Desktop\corpus\new_h.txt'
min_count = 9


with open(corpus) as fr:
    lines = fr.readlines()

all_words = {}

for i, line in enumerate(lines):
    words = line.strip('\n').split()
    print('line: %d' % i)
    for word in words:
        if word in all_words:
            all_words[word] += 1
        else:
            all_words[word] = 1


all_words = {i: j for i, j in all_words.items() if j >= min_count}
# id == 0: UKN
# {id: [word]}
id2word = {i+1: [j] for i, j in enumerate(all_words)}
# {word: [id]}
word2id = {j[0]: [i] for i, j in id2word.items()}
id2word = pd.DataFrame(id2word)
word2id = pd.DataFrame(word2id)
id2word.to_csv(R'C:\Users\xiong35\Desktop\id2word.csv')
word2id.to_csv(R'C:\Users\\xiong35\Desktop\word2id.csv')
