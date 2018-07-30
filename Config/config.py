import os

data_prefix_path = '../Data/'
save_prefix_path = '../Save/'
cache_prefix_path = '../Cache/'
output_prefix_path = '../Output/'

EN_TRAIN_FILE = os.path.join(data_prefix_path, 'cikm_english_train_20180516.txt')
ES_TRAIN_FILE = os.path.join(data_prefix_path, 'cikm_spanish_train_20180516.txt')

TEST_FILE_A = os.path.join(data_prefix_path, 'cikm_test_a_20180516.txt')
TEST_FILE_B = os.path.join(data_prefix_path, 'cikm_test_b_20180730.txt')
TRANSLATE_FILE = os.path.join(data_prefix_path, 'cikm_unlabel_spanish_train_20180516.txt')

TOKEN_TRAIN = os.path.join(cache_prefix_path, 'token_train.pkl')
TOKEN_VAL = os.path.join(cache_prefix_path, 'token_dev.pkl')
TOKEN_TEST_A = os.path.join(cache_prefix_path, 'token_test_a.pkl')
TOKEN_TEST_B = os.path.join(cache_prefix_path, 'token_test_b.pkl')

EN_EMBEDDING_MATRIX = os.path.join(data_prefix_path, 'wiki.en.vec')
ES_EMBEDDING_MATRIX = os.path.join(data_prefix_path, 'wiki.es.vec')




