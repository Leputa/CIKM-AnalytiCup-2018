import sys
typ = sys.getfilesystemencoding()
sys.path.append('../')

import urllib.request
import urllib.parse
import pyperclip
from tkinter import *
from tqdm import tqdm

from Config import config


class GoogleTranslation():
    def load_data(self):
        data = []
        with open(config.TEST_FiLE, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                data.append(line)
        return data

    def translate(self, text, f='en', t='es'):
        # f: 目标语言
        # t: 源语言
        url_google = 'http://translate.google.cn/translate_t'
        reg_text = re.compile(r'(?<=TRANSLATED_TEXT=).*?;')
        user_agent = r'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) ' \
                     r'Chrome/44.0.2403.157 Safari/537.36'
        '''user_agent = 'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 2.0.50727)'''''
        values = {'hl': 'en', 'ie': 'utf-8', 'text': text, 'langpair': '%s|%s' % (t, f)}
        value = urllib.parse.urlencode(values)

        req = urllib.request.Request(url_google + '?' + value)

        req.add_header('User-Agent', user_agent)
        response = urllib.request.urlopen(req)
        content = response.read().decode('utf-8')
        data = reg_text.search(content)
        result = data.group(0).strip(';').strip('\'')
        return result

    def translate_data(self):
        data_es = self.load_data()
        with open(config.data_prefix_path + 'test_en.txt', 'w', encoding='utf-8') as fr:
            for sentence in tqdm(data_es):
                s_list = sentence.strip().split('\t')
                fr.write(self.translate(s_list[0]) + '\t' + self.translate(s_list[1]) + '\n')

if __name__ == '__main__':
    G = GoogleTranslation()
    G.translate_data()
