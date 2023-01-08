import faiss
import json
import argparse
import numpy as np
import time
import os
import pickle
import torch
from itertools import chain
from tqdm import tqdm
from faiss import normalize_L2

from collections import defaultdict
from encode_corpus import Encoder
from datasets import load_dataset

'''
todo:用模型完成文档检索任务，输出格式：query number, fixed string Q0, document id, rank, retrieval score, run id
一个例子：1 Q0 24135#2 2 80.4531 teamX-runY
topK取100
'''


class Searcher(object):
    def __init__(self, encoder, index_file, topk):
        self.index_file = index_file
        self.encoder = encoder
        self.topk = topk
        self.build_index()

    def build_index(self):
        p_reps_0, p_lookup_0 = self.pickle_load(self.index_file) # p_reps_0就是corpus的embedding， p_lookup_0是他们的docid
        self.look_up = p_lookup_0
        index = faiss.IndexFlatL2(p_reps_0.shape[1])
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, index)  # index
        self.add(p_reps_0)

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def pickle_load(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def searcher(self, query):
        res = {}
        qvec = self.encoder.predict([query])
        scores, indices = self.index.search(qvec, self.topk)
        scores = scores[0] # 不知道为啥包着一层list
        doc_ids = [self.look_up[x] for x in indices[0]]
        for i in range(len(doc_ids)):
            res[doc_ids[i]] = scores[i]
        res = sorted(res.items(), key=lambda x : x[1], reverse=True) # 按照score进行排序
        return res

    def read_query_file(self, input_file):
        query_dataset = load_dataset('json', data_files=input_file)['train']
        query_lst = []
        for i in range(len(query_dataset)):
            query_id = query_dataset[i]['query_id']
            query = query_dataset[i]['query']
            query_lst.append((query_id, query))
        return query_lst

    def process(self, inputFile, outputFile):
        query_lst = self.read_query_file(inputFile)
        print(">>> Now is searching ... ")
        fw = open(outputFile, "w")
        for queryl in tqdm(query_lst):
            query_id, query = queryl
            doc2score = self.searcher(query)
            # print ("scores: ", scores)
            # print ("indices: ", indices)
            for i in range(len(doc2score)):
                fw.write(' '.join([query_id, 'Q0', doc2score[i][0], str(i+1), str(doc2score[i][1]), 'team111-run1\n']))
        fw.close()


if __name__ == "__main__":

    with open('config/retrieval.json') as f:
        config = json.load(f)

    # todo：有些language的语料库没完成index
    unindexed_language = ['en', 'fr']
    languages = ['ar', 'bn', 'fi', 'id', 'ja', 'ko', 'sw', 'te', 'th', 'zh']
    encoder = Encoder(config['model_path'], config['pool_type'], config['max_seq_len'])
    for language in languages:
        print('------ {} dev 开始检索 ------'.format(language))
        index_file = '{}/{}/indexed_corpus.ct'.format(config['index_file'], language)
        input_file = '{}/{}/dev.jsonl'.format(config['input_file'], language)
        output_file = '{}/{}/{}_dev.txt'.format(config['output_file'], language, language)

        if not os.path.exists('{}/{}'.format(config['output_file'], language)):
            os.makedirs('{}/{}'.format(config['output_file'], language))
        if not os.path.exists(output_file):
            os.mknod(output_file)

        searcher = Searcher(encoder, index_file, config['topK'])
        searcher.process(input_file, output_file)
        print('------ {} dev 检索完成 ------'.format(language))

        if os.path.exists('{}/{}/testA.jsonl'.format(config['input_file'], language)): # 如果存在testA，那么也要search
            print('------ {} testA 开始检索 ------'.format(language))
            input_file = '{}/{}/testA.jsonl'.format(config['input_file'], language)
            output_file = '{}/{}/{}_testA.txt'.format(config['output_file'], language, language)
            if not os.path.exists(output_file):
                os.mknod(output_file)

            searcher = Searcher(encoder, index_file, config['topK'])
            searcher.process(input_file, output_file)
            print('------ {} testA 检索完成 ------'.format(language))