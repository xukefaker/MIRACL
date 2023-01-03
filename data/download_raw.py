import os.path

from datasets import get_dataset_config_names
from datasets import load_dataset
import json

with open('../config/download.json') as f:
    config = json.load(f)


languages = ['ar', 'bn', 'zh', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'en']
hf_token = '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'
for language in languages:
    datadic = load_dataset('miracl/miracl', language, use_auth_token=hf_token, cache_dir=config['cache'])
    for split, data in datadic.items():
        if not os.path.exists('{}/{}'.format(config['raw_dir'], language)):
            os.makedirs('{}/{}'.format(config['raw_dir'], language))
        if not os.path.exists('{}/{}/{}.jsonl'.format(config['raw_dir'], language, split)):
            os.mknod('{}/{}/{}.jsonl'.format(config['raw_dir'], language, split))
        data.to_json('{}/{}/{}.jsonl'.format(config['raw_dir'], language, split))

