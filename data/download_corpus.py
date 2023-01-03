import os.path

from datasets import get_dataset_config_names
from datasets import load_dataset
import json

with open('../config/download.json') as f:
    config = json.load(f)


languages = ['ar', 'bn', 'zh', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'en']
hf_token = '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'

for language in languages:
    miracl_corpus = load_dataset('miracl/miracl-corpus', language, cache_dir=config['cache'])['train']
    if not os.path.exists('{}/{}'.format(config['corpus_dir'], language)):
        os.makedirs('{}/{}'.format(config['corpus_dir'], language))
    if not os.path.exists('{}/{}/{}-corpus.jsonl'.format(config['corpus_dir'], language, language)):
        os.mknod('{}/{}/{}-corpus.jsonl'.format(config['corpus_dir'], language, language))
    miracl_corpus.to_json('{}/{}/{}-corpus.jsonl'.format(config['corpus_dir'], language, language))