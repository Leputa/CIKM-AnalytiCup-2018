import numpy as np
import gc

import os
import sys
sys.path.append("../")

from scipy.stats import rankdata
from Config import config

def get_subs(fileNames):
    predict_list = []
    for file in fileNames:
        with open(config.output_prefix_path + file, 'r') as fr:
            predict_list.append(np.array([float(line[:-1]) for line in fr.readlines()]))
    return predict_list



if __name__ == '__main__':
    fileNames = ['es_ABCNN3-submit.txt', '960_xgboost_human_feature-summit.txt','3_cv_es_LexDecomp-submit_0.001.txt', '3_cv_es_LexDecomp-submit_0.0009.txt']

    predict_list = get_subs(fileNames)
    gc.collect()

    print("Rank averaging on ", len(predict_list), " files")
    predictions = np.zeros_like(predict_list[0])
    # for predict in predict_list:
    #     predictions = np.add(predictions, rankdata(predict)/predictions.shape[0])
    #     gc.collect()
    # predictions /= len(predict_list)
    predictions = predict_list[0] * 0.27 + predict_list[1] * 0.23 +  predict_list[2] * 0.3 + predict_list[3] * 0.2

    with open(config.output_prefix_path + 'blending.txt', 'w') as fr:
        for sub in predictions:
            fr.write(str(sub) + '\n')
