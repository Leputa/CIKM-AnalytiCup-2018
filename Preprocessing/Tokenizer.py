import sys
sys.path.append('../')

import re
import string
from nltk.stem import WordNetLemmatizer
import pattern.es as lemEsp

from Config import config
from gensim.models.keyedvectors import KeyedVectors

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

    def es_str_clean(self, text):
        text= text.strip().lower()
        text = ' ' + text + ' '

        # text = re.sub(r"á", "a", text)
        # text = re.sub(r"ó", "o", text)
        # text = re.sub(r"é", "e", text)
        # text = re.sub(r"í", "i", text)
        # text = re.sub(r"ú", "u", text)

        # acronym
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"can′t", "can not", text)

        # punctuation
        text = re.sub(r"\[([^\]]+)\]", " ", text)
        text = re.sub(r"\(([^\)]+)\)", " ", text)
        text = re.sub(r"\.", " . ", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"/", " / ", text)
        text = re.sub(r"¿", " ¿ ", text)
        text = re.sub(r"？", "", text)
        text = re.sub(r"¡", " ¡ ", text)
        text = re.sub(r"´", " ", text)
        text = re.sub(r"`", " ", text)
        text = re.sub(r"-", " - ", text)
        text = re.sub(r"^[!]", " ! ", text)
        text = re.sub(r"^[?]", " ¿ ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r";", " ; ", text)
        text = re.sub(r"\s{2,}", " ", text)

        # unit
        text = re.sub(r"(\d+)h ", lambda m: m.group(1) + ' hora ', text)
        text = re.sub(r"(\d+)% ", lambda m: m.group(1) + ' por ciento ', text)
        text = re.sub(r"(\d+)cm ", lambda m: m.group(1) + ' cm ', text)
        text = re.sub(r"(\d+)€ ", lambda m: m.group(1) + ' euro ', text)

        # digit
        text = re.sub(r" 0 ", " cero ", text)
        text = re.sub(r" 1 ", " uno ", text)
        text = re.sub(r" 2 ", " dos ", text)
        text = re.sub(r" 3 ", " tres ", text)
        text = re.sub(r" 4 ", " cuatro ", text)
        text = re.sub(r" 5 ", " cinco ", text)
        text = re.sub(r" 6 ", " seis ", text)
        text = re.sub(r" 7 ", " siete ", text)
        text = re.sub(r" 8 ", " ocho ", text)
        text = re.sub(r" 9 ", " nueve ", text)

        text = re.sub(r" 10 ", " veinte ", text)
        text = re.sub(r" 20 ", " treinta ", text)
        text = re.sub(r" 30 ", " tres ", text)
        text = re.sub(r" 40 ", " cuarenta ", text)
        text = re.sub(r" 50 ", " cincuenta ", text)
        text = re.sub(r" 60 ", " sesenta ", text)
        text = re.sub(r" 70 ", " setenta ", text)
        text = re.sub(r" 80 ", " ochenta ", text)
        text = re.sub(r" 90 ", " noventa ", text)


        text = re.sub(r" 100 ", " cien ", text)
        text = re.sub(r" 200 ", " doscientos ", text)
        text = re.sub(r" 500 ", " quinientos ", text)
        text = re.sub(r" 600 ", " seiscientos ", text)
        text = re.sub(r" 1000 ", " mil ", text)

        text = re.sub(r" 2nd ", " segundo ", text)

        text = re.sub(r" 2016 ", " año pasado ", text)
        text = re.sub(r" 2017 ", " este año ", text)

        # symbol replacement
        text = re.sub(r"&", " and ", text)
        text = re.sub(r"\|", " or ", text)
        text = re.sub(r"=", " equal ", text)
        text = re.sub(r"\+", " plus ", text)

        text = re.sub(r" e \- mail ", " email ", text)
        text = re.sub(r" e\-mail ", " email ", text)
        text = re.sub(r" imail ", " email ", text)
        text = re.sub(r" 3d ", " tridimensional ", text)
        text = re.sub(r" mp4 ", " el jugadores ", text)
        text = re.sub(r" trademanager ", " alibaba ", text)
        text = re.sub(r" hmmmmm ", " bueno ", text)
        text = re.sub(r' :\) ', " alegría ", text)
        text = re.sub(r' :o ', " sorprendieron ", text)

        # 词形还原
        string = ' '.join(lemEsp.Sentence(lemEsp.parse(text, lemmata=True)).lemmata)
        word_list = []
        for str in string.split('\n'):
            word_list.extend(str.split(' '))
        words = [word for word in word_list if word not in self.punc]

        return words

    def en_str_clean(self, text):
        """
        Clean text
        :param text: the string of text
        :return: text string after cleaning
        """
        text= text.strip().lower()
        text = ' ' + text + ' '

        # unit
        text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', text)  # e.g. 4kgs => 4 kg
        text = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', text)  # e.g. 4kg => 4 kg
        text = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', text)  # e.g. 4k => 4000
        text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
        text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)

        # acronym
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"what\'s", "what is", text)
        text = re.sub(r"What\'s", "what is", text)
        text = re.sub(r"\'ve ", " have ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"i\'m", "i am ", text)
        text = re.sub(r"I\'m", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"c\+\+", "cplusplus", text)
        text = re.sub(r"c \+\+", "cplusplus", text)
        text = re.sub(r"c \+ \+", "cplusplus", text)
        text = re.sub(r"c#", "csharp", text)
        text = re.sub(r"f#", "fsharp", text)
        text = re.sub(r"g#", "gsharp", text)
        text = re.sub(r" e mail ", " email ", text)
        text = re.sub(r" e \- mail ", " email ", text)
        text = re.sub(r" e\-mail ", " email ", text)
        text = re.sub(r",000", '000', text)
        text = re.sub(r"\'s", " ", text)

        # spelling correction
        text = re.sub(r"ph\.d", "phd", text)
        text = re.sub(r"PhD", "phd", text)
        text = re.sub(r"pokemons", "pokemon", text)
        text = re.sub(r"pokémon", "pokemon", text)
        text = re.sub(r"pokemon go ", "pokemon-go ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" 9 11 ", " 911 ", text)
        text = re.sub(r" j k ", " jk ", text)
        text = re.sub(r" fb ", " facebook ", text)
        text = re.sub(r"facebooks", " facebook ", text)
        text = re.sub(r"facebooking", " facebook ", text)
        text = re.sub(r"insidefacebook", "inside facebook", text)
        text = re.sub(r"donald trump", "trump", text)
        text = re.sub(r"the big bang", "big-bang", text)
        text = re.sub(r"the european union", "eu", text)
        text = re.sub(r" usa ", " america ", text)
        text = re.sub(r" us ", " america ", text)
        text = re.sub(r" u s ", " america ", text)
        text = re.sub(r" U\.S\. ", " america ", text)
        text = re.sub(r" US ", " america ", text)
        text = re.sub(r" American ", " america ", text)
        text = re.sub(r" America ", " america ", text)
        text = re.sub(r" quaro ", " quora ", text)
        text = re.sub(r" mbp ", " macbook-pro ", text)
        text = re.sub(r" mac ", " macbook ", text)
        text = re.sub(r"macbook pro", "macbook-pro", text)
        text = re.sub(r"macbook-pros", "macbook-pro", text)
        text = re.sub(r" 1 ", " one ", text)
        text = re.sub(r" 2 ", " two ", text)
        text = re.sub(r" 3 ", " three ", text)
        text = re.sub(r" 4 ", " four ", text)
        text = re.sub(r" 5 ", " five ", text)
        text = re.sub(r" 6 ", " six ", text)
        text = re.sub(r" 7 ", " seven ", text)
        text = re.sub(r" 8 ", " eight ", text)
        text = re.sub(r" 9 ", " nine ", text)
        text = re.sub(r"googling", " google ", text)
        text = re.sub(r"googled", " google ", text)
        text = re.sub(r"googleable", " google ", text)
        text = re.sub(r"googles", " google ", text)
        text = re.sub(r" rs(\d+)", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"(\d+)rs", lambda m: ' rs ' + m.group(1), text)
        text = re.sub(r"the european union", " eu ", text)
        text = re.sub(r"dollars", " dollar ", text)

        # punctuation
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"-", " - ", text)
        text = re.sub(r"/", " / ", text)
        text = re.sub(r"\\", " \ ", text)
        text = re.sub(r"=", " = ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r"\.", " . ", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"\?", " ? ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\"", " \" ", text)
        text = re.sub(r"&", " & ", text)
        text = re.sub(r"\|", " | ", text)
        text = re.sub(r";", " ; ", text)
        text = re.sub(r"\(", " ( ", text)
        text = re.sub(r"\)", " ( ", text)

        # symbol replacement
        text = re.sub(r"&", " and ", text)
        text = re.sub(r"\|", " or ", text)
        text = re.sub(r"=", " equal ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r"₹", " rs ", text)  # 测试！
        text = re.sub(r"\$", " dollar ", text)

        word_list = text.strip().lower().split(" ")
        words = [word for word in word_list if word not in self.punc]

        return [self.wnl.lemmatize(word) for word in words]


if __name__ == '__main__':
    tokenizer = Tokenizer()
    print(tokenizer.es_str_clean('procesadar'))
