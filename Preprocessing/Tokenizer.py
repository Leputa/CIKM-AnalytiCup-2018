import sys
sys.path.append('../')

import re
import string
from nltk.stem import WordNetLemmatizer
import pattern.es as lemEsp

from Config import config

class Tokenizer():
    def __init__(self):
        self.punc = string.punctuation
        self.stop_words = []
        self.wnl = WordNetLemmatizer()


        stop_words_path = config.data_prefix_path + 'spanish.txt'
        with open(stop_words_path, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                self.stop_words.append(line.strip())

    def es_str_clean(self, string):
        string = string.strip().lower()
        string = re.sub(r"\[([^\]]+)\]", " ", string)
        string = re.sub(r"\(([^\)]+)\)", " ", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"/", " / ", string)
        string = re.sub(r"¿", " ¿ ", string)
        string = re.sub(r"¡", " ¡ ", string)
        string = re.sub(r"^[!]", " ! ", string)
        string = re.sub(r"^[?]", " ¿ ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r";", " ; ", string)
        string = re.sub(r"\s{2,}", " ", string)

        # 词形还原
        string = ' '.join(lemEsp.Sentence(lemEsp.parse(string, lemmata=True)).lemmata)
        word_list = []
        for str in string.split('\n'):
            word_list.extend(str.split(' '))

        words = [word for word in word_list if word not in self.punc]

        return words

    def en_str_clean(self, string):
        string = re.sub(r"\[([^\]]+)\]", " ", string)
        string = re.sub(r"\(([^\)]+)\)", " ", string)
        string = re.sub(r"[^A-Za-z0-9,!?.;]", " ", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"/", " / ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r";", " ; ", string)
        string = re.sub(r"\s{2,}", " ", string)

        word_list = string.strip().lower().split(" ")
        words = [word for word in word_list if word not in self.punc]

        return [self.wnl.lemmatize(word) for word in words]


if __name__ == '__main__':
    tokenizer = Tokenizer()
    print(tokenizer.es_str_clean('¡Hola! Cerré la disputa el 21 de mayo de 2017 y dice que se realizará el reembolso.'))
