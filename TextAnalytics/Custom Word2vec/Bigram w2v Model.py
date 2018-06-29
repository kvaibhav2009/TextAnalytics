import os,sys
import codecs
from gensim.models import Phrases
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
import re
import multiprocessing
from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
import gensim
from collections import Counter
from tqdm import tqdm

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield word_tokenize(line.encode('ascii', 'ignore').decode('ascii'))

class PhraseItertor(object):
    def __init__(self, my_phraser, data):
        self.my_phraser, self.data = my_phraser, data

    def __iter__(self):
        yield self.my_phraser[self.data]


sentences = MySentences('Corpus/')
bigram_transformer = gensim.models.Phrases(sentences)

bigram = gensim.models.phrases.Phraser(bigram_transformer)
corpus = PhraseItertor(bigram, sentences)

model = gensim.models.Word2Vec(corpus, size=300, window=5, workers=8)


x=bigram.add_vocab([iterator.filteredText])

list(bigram[iterator.filteredText])[:5]


bigram_counter = Counter()
for key in bigram.vocab.keys():
    if key not in stopwords.words("english"):
        if len(key.split("_")) > 1:
            bigram_counter[key] += bigram.vocab[key]
for key, counts in bigram_counter.most_common(20):
    print '{0: <20} {1}'.format(key.encode("utf-8"), counts)


bigi_counter = dict()
for key in bigram.vocab.keys():
    if key not in stopwords.words("english") and bigram.vocab[key]>500:
        if len(key.split("_")) > 1:
            bigi_counter[key] = bigram.vocab[key]


file = codecs.open("bigram_vocab.txt", "w", "utf-8")
content=""
for k in bigi_counter.keys():
      content=content+"\n"+k

file.write(content)

file.close()



SoftModel_bigram_w2v = Word2Vec(size=300, min_count=10)
SoftModel_bigram_w2v.build_vocab([x for x in tqdm(iterator.filteredText)])

SoftModel_bigram_w2v.train([x for x in tqdm(iterator.filteredText)],total_examples=iterator.filteredText.__len__(),word_count=5,epochs=150)
