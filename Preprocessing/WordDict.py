import sys
sys.path.append('../')

from Config import config
import pickle
import string


class WordDict():
    def __init__(self):
        self.En2IndexDic = {}
        self.Es2IndexDic = {}

        self.special_word = ['<Padding>','<UNKNOWN>', '<DIGITS>', 'id']
        for i, word in enumerate(self.special_word):
            self.En2IndexDic[word] = i
            self.Es2IndexDic[word] = i

    def add_word(self, word, tag):
        if word.isdigit() and len(word) < 8:
            if tag == 'es':
                self.Es2IndexDic[word] = 2
            elif tag == 'en':
                self.En2IndexDic[word] = 2

        elif any(ch in string.digits for ch in word)==True and len(word) >= 8:
            if tag == 'es':
                self.Es2IndexDic[word] = 3
            elif tag == 'en':
                self.En2IndexDic[word] = 3

        else:
            if tag == 'es':
                if self.Es2IndexDic.get(word) == None:
                    self.Es2IndexDic[word] = len(self.Es2IndexDic)
            elif tag == 'en':
                if self.En2IndexDic.get(word) == None:
                    self.En2IndexDic[word] = len(self.En2IndexDic)

    def get_index(self, word, tag):
        if tag == 'es':
            return self.Es2IndexDic.get(word)
        elif tag == 'en':
            return self.En2IndexDic.get(word)

    def get_size(self, tag):
        if tag == 'es':
            return len(self.Es2IndexDic)
        elif tag == 'en':
            return len(self.En2IndexDic)

    def saveWord2IndexDic(self, tag):
        if tag == 'es':
            path = config.cache_prefix_path + 'Es2IndexDic.pkl'
            with open(path, 'wb') as pkl:
                pickle.dump(self.Es2IndexDic, pkl)
        elif tag == 'en':
            path = config.cache_prefix_path + 'En2IndexDic.pkl'
            with open(path, 'wb') as pkl:
                pickle.dump(self.En2IndexDic, pkl)

    def loadWord2IndexDic(self, tag):
        if tag == 'es':
            self.Es2IndexDic.clear()
            path = config.cache_prefix_path + 'Es2IndexDic.pkl'
            with open(path, 'rb') as pkl:
                self.Es2IndexDic = pickle.load(pkl)
                return self.Es2IndexDic
        elif tag == 'en':
            self.En2IndexDic.clear()
            path = config.cache_prefix_path + 'En2indexDic.pkl'
            with open(path, 'rb') as pkl:
                self.En2IndexDic = pickle.load(pkl)
                return self.En2IndexDic



