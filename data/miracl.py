# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import json
import datasets
from collections import defaultdict
from dataclasses import dataclass

_CITATION = '''
'''

languages = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh']
new_languages = ['es', 'fa', 'fr', 'hi', 'zh']
non_surprise_languages = languages

_DESCRIPTION = 'dataset load script for MIRACL'

_DATASET_URLS = {
    lang: {
        'train': [
            f'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-train.tsv',
            f'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-train.tsv',
        ],
        'dev': [
            f'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-dev.tsv',
            f'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-dev.tsv',
        ],
    } for lang in languages
}
for lang in languages:
    if lang in new_languages:
        continue
    _DATASET_URLS[lang]['testA'] = [
        f'https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-test-a.tsv',
    ]


def load_topic(fn):
    qid2topic = {}
    with open(fn, encoding="utf-8") as f:
        for line in f:
            qid, topic = line.strip().split('\t')
            qid2topic[qid] = topic
    return qid2topic


def load_qrels(fn):
    if fn is None:
        return None

    qrels = defaultdict(dict)
    with open(fn, encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rel = line.strip().split('\t')
            qrels[qid][docid] = int(rel)
    return qrels


class MIRACL(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [datasets.BuilderConfig(
        version=datasets.Version('1.0.0'),
        name=lang, description=f'MIRACL dataset in language {lang}.'
    ) for lang in languages
    ]

    def _info(self):
        features = datasets.Features({
            'query_id': datasets.Value('string'),
            'query': datasets.Value('string'),

            'positive_passages': [{
                'docid': datasets.Value('string'),
                'text': datasets.Value('string'), 'title': datasets.Value('string')
            }],
            'negative_passages': [{
                'docid': datasets.Value('string'),
                'text': datasets.Value('string'), 'title': datasets.Value('string'),
            }],
        })

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage='https://project-miracl.github.io',
            # License for the dataset if available
            license='',
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        lang = self.config.name
        downloaded_files = dl_manager.download_and_extract(_DATASET_URLS[lang])

        splits = [
            datasets.SplitGenerator(
                name='train',
                gen_kwargs={
                    'filepaths': downloaded_files['train'],
                },
            ),
            datasets.SplitGenerator(
                name='dev',
                gen_kwargs={
                    'filepaths': downloaded_files['dev'],
                },
            ),
        ]
        if lang not in new_languages:
            splits.append(datasets.SplitGenerator(
                name='testA',
                gen_kwargs={
                    'filepaths': downloaded_files['testA'],
                },
            ))
        return splits

    def _generate_examples(self, filepaths):
        lang = self.config.name
        miracl_corpus = datasets.load_dataset('miracl/miracl-corpus', lang)['train']
        docid2doc = {doc['docid']: (doc['title'], doc['text']) for doc in miracl_corpus}

        topic_fn, qrel_fn = (filepaths) if len(filepaths) == 2 else (filepaths[0], None)
        qid2topic = load_topic(topic_fn)
        qrels = load_qrels(qrel_fn)
        for qid in qid2topic:
            data = {}
            data['query_id'] = qid
            data['query'] = qid2topic[qid]

            pos_docids = [docid for docid, rel in qrels[qid].items() if rel == 1] if qrels is not None else []
            neg_docids = [docid for docid, rel in qrels[qid].items() if rel == 0] if qrels is not None else []
            data['positive_passages'] = [{
                'docid': docid,
                **dict(zip(['title', 'text'], docid2doc[docid]))
            } for docid in pos_docids]
            data['negative_passages'] = [{
                'docid': docid,
                **dict(zip(['title', 'text'], docid2doc[docid]))
            } for docid in neg_docids]
            yield qid, data