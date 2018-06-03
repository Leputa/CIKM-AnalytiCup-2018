import re

class Tokenizer():
    def es_str_clean(self, string):
        string = re.sub(r"\[([^\]]+)\]", " ", string)
        string = re.sub(r"\(([^\)]+)\)", " ", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"/", " / ", string)
        string = re.sub(r"¿", "", string)
        string = re.sub(r"¡", "", string)
        string = re.sub(r"^[?!]", "", string)
        string = re.sub(r"\?", " . ", string)
        string = re.sub(r";", " ; ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower().split(" ")

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

        return string.strip().lower().split(" ")


