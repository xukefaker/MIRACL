import argparse
import json
import os
import random

from transformers import AutoTokenizer
from multiprocessing import Pool
from tqdm import tqdm
import glob
from datasets import load_dataset


def encode_one(raw_data_path, output_path):
    data_dic = load_dataset('json', data_files=raw_data_path)

    for split, data in data_dic.items():

        query = data['query']
        positives = data['positive_passages']
        negatives = data['negative_passages'] # 根据id取对应的line
        spans = [] # 一个element就是经过处理的query, positives, negative
        for i in range(len(query)):
            temp = []
            temp.append(query[i])
            # for j in range(len(positives[i])):
            #     temp.append(positives[i][j]['text'])
            temp.append(positives[i][0]['text']) # 为了简单起见，只取一个positive
            if len(negatives[i]) == 0:
                neg_index = random.randint(0, len(query) - 1)
                while len(positives[random.randint(0, len(query) - 1)]) == 0:
                    neg_index = random.randint(0, len(query) - 1)
                temp.append(positives[neg_index][0]['text']) # 有些query没有negative，就随机挑一个其他query的positive
            else:
                temp.append(negatives[i][0]['text'])

            tokenized = [
                tokenizer(
                    s.lower(),
                    add_special_tokens=False,
                    truncation=True,
                    max_length=510,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"] for s in temp
            ]
            spans.append(tokenized)
            if i % 500 == 0:
                print('------ {} 已处理 {} ------'.format(raw_data_path, i))

        res = [json.dumps({'spans' : spans[k]}) for k in range(len(spans))]
    return res
    # with open(output_path, 'w') as output:
    #     for x in res:
    #         if x is None:
    #             continue
    #         output.write(x + "\n")


def get_index(arr, item):
    return [i for i in range(len(arr)) if arr[i] == item]

with open('config/create_train.json') as f:
    config = json.load(f)


tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'], use_fast=True)
languages = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'sw', 'te', 'th', 'zh'] # ru数据好像有问题，暂时跳过
# raw_data_path =  glob.glob('{}/*/*.jsonl'.format(config['raw_data']))
# raw_data_path.sort()
# print(data['train'][0].keys())

res = []
for language in languages:
    raw_data_paths = glob.glob('{}/{}/train.jsonl'.format(config['raw_data'], language))

    for raw_data_path in raw_data_paths:
        output_path = '{}/{}/{}.json'.format(config['output_path'], language, raw_data_path[get_index(raw_data_path, '/')[-1] + 1 : get_index(raw_data_path, '.')[0]])
        if not os.path.exists('{}/{}'.format(config['output_path'], language)):
            os.makedirs('{}/{}'.format(config['output_path'], language))
        if not os.path.exists(output_path):
            os.mknod(output_path)
        res.extend(encode_one(raw_data_path, output_path))

        # with open(output_path, 'w') as f:
        #     with Pool() as p:
        #         all_tokenized = p.imap_unordered(
        #             encode_one,
        #             tqdm(open(raw_data_path), ascii=True),
        #             chunksize=500,
        #         )
        #         for x in all_tokenized:
        #             if x is None:
        #                 continue
        #             f.write(x + "\n")

        # print('------{} 输出完成 ------'.format(output_path))
    with open('data/train.json', 'w') as output:
        for x in res:
            if x is None:
                continue
            output.write(x + "\n")
