# LASER embeddings

[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/yannvgn/laserembeddings/Python%20package?style=flat-square)](https://github.com/yannvgn/laserembeddings/actions)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/laserembeddings?style=flat-square)
[![PyPI](https://img.shields.io/pypi/v/laserembeddings.svg?style=flat-square)](https://pypi.org/project/laserembeddings/)
[![PyPI - License](https://img.shields.io/pypi/l/laserembeddings.svg?style=flat-square)](https://github.com/yannvgn/laserembeddings/blob/master/LICENSE)

**Out-of-the-box multilingual sentence embeddings.**

![LASER embeddings maps similar sentences in any language to similar language-agnostic embeddings](laserembeddings.gif)

_laserembeddings_ is a pip-packaged, production-ready port of Facebook Research's [LASER](https://github.com/facebookresearch/LASER) (Language-Agnostic SEntence Representations) to compute multilingual sentence embeddings.

✨ **Version 1.1.2 is here! What's new?**
- A compatibility issue with subword-nmt 0.3.8 was fixed (#39) 🐛
- The behavior of `Laser.embed_sentences` was unclear/misleading when the number of language codes received in the `lang` argument did not match the number of sentences to encode. It now raises an error in that case 🐛

## Context

[LASER](https://github.com/facebookresearch/LASER) is a collection of scripts and models created by Facebook Research to compute **multilingual sentence embeddings** for zero-shot cross-lingual transfer. 

What does it mean? LASER is able to transform sentences into **language-independent vectors**. Similar sentences get mapped to close vectors (in terms of cosine distance), regardless of the input language.

That is great, especially if you don't have training sets for the language(s) you want to process: you can build a classifier on top of LASER embeddings, train it on whatever language(s) you have in your training data, and let it classify texts in any language.

**The aim of the package is to make LASER as easy-to-use and easy-to-deploy as possible: zero-config, production-ready, etc., just a two-liner to install.**

👉 👉 👉 For detailed information, have a look at the amazing [LASER repository](https://github.com/facebookresearch/LASER), read its [presentation article](https://code.fb.com/ai-research/laser-multilingual-sentence-embeddings/) and its [research paper](https://arxiv.org/abs/1812.10464). 👈 👈 👈

## Getting started

### Prerequisites

You'll need Python 3.6+ and PyTorch. Please refer to [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

### Installation

```
pip install laserembeddings
```

#### Chinese language

Chinese is not supported by default. If you need to embed Chinese sentences, please install laserembeddings with the "zh" extra. This extra includes [jieba](https://github.com/fxsjy/jieba).

```
pip install laserembeddings[zh]
```

#### Japanese language

Japanese is not supported by default. If you need to embed Japanese sentences, please install laserembeddings with the "ja" extra. This extra includes [mecab-python3](https://github.com/SamuraiT/mecab-python3) and the [ipadic](https://github.com/polm/ipadic-py) dictionary, which is used in the original LASER project.

If you have issues running laserembeddings on Japanese sentences, please refer to [mecab-python3 documentation](https://github.com/SamuraiT/mecab-python3) for troubleshooting.

```
pip install laserembeddings[ja]
```


### Downloading the pre-trained models

```
python -m laserembeddings download-models
```

This will download the models to the default `data` directory next to the source code of the package. Use `python -m laserembeddings download-models path/to/model/directory` to download the models to a specific location.

### Usage

```python
from laserembeddings import Laser

laser = Laser()

# if all sentences are in the same language:

embeddings = laser.embed_sentences(
    ['let your neural network be polyglot',
     'use multilingual embeddings!'],
    lang='en')  # lang is only used for tokenization

# embeddings is a N*1024 (N = number of sentences) NumPy array
```

If the sentences are not in the same language, you can pass a list of language codes:
```python
embeddings = laser.embed_sentences(
    ['I love pasta.',
     "J'adore les pâtes.",
     'Ich liebe Pasta.'],
    lang=['en', 'fr', 'de'])
```

If you downloaded the models into a specific directory:

```python
from laserembeddings import Laser

path_to_bpe_codes = ...
path_to_bpe_vocab = ...
path_to_encoder = ...

laser = Laser(path_to_bpe_codes, path_to_bpe_vocab, path_to_encoder)

# you can also supply file objects instead of file paths
```

If you want to pull the models from S3:

```python
from io import BytesIO, StringIO
from laserembeddings import Laser
import boto3

s3 = boto3.resource('s3')
MODELS_BUCKET = ...

f_bpe_codes = StringIO(s3.Object(MODELS_BUCKET, 'path_to_bpe_codes.fcodes').get()['Body'].read().decode('utf-8'))
f_bpe_vocab = StringIO(s3.Object(MODELS_BUCKET, 'path_to_bpe_vocabulary.fvocab').get()['Body'].read().decode('utf-8'))
f_encoder = BytesIO(s3.Object(MODELS_BUCKET, 'path_to_encoder.pt').get()['Body'].read())

laser = Laser(f_bpe_codes, f_bpe_vocab, f_encoder)
```

## What are the differences with the original implementation?

Some dependencies of the original project have been replaced with pure-python dependencies, to make this package easy to install and deploy.

Here's a summary of the differences:

| Part of the pipeline | LASER dependency (original project) | laserembeddings dependency (this package) | Reason |
|----------------------|-------------------------------------|----------------------------------------|--------|
| Normalization / tokenization | [Moses](https://github.com/moses-smt/mosesdecoder) | [Sacremoses](https://github.com/alvations/sacremoses) 0.0.35, which seems to be the closest version to the Moses version used to train the model | Moses is implemented in Perl |
| BPE encoding | [fastBPE](https://github.com/glample/fastBPE) | [subword-nmt](https://github.com/rsennrich/subword-nmt) | fastBPE cannot be installed via pip and requires compiling C++ code |
| Japanese segmentation (optional) | [MeCab](https://github.com/taku910/mecab) / [JapaneseTokenizer](https://github.com/Kensuke-Mitsuzawa/JapaneseTokenizers) | [mecab-python3](https://github.com/SamuraiT/mecab-python3) and [ipadic](https://github.com/polm/ipadic-py) dictionary | mecab-python3 comes with wheels for major platforms (no compilation needed) |

## Will I get the exact same embeddings?

**For most languages, in most of the cases, yes.**

Some slight (and not so slight 🙄) differences exist for some languages due to differences in the implementation of the Tokenizer.

**[An exhaustive comparison of the embeddings generated with LASER and laserembeddings](tests/report/comparison-with-LASER.md) is automatically generated and will be updated for each new release.**

## FAQ

**How can I train the encoder?**

You can't. LASER models are pre-trained and do not need to be fine-tuned. The embeddings are generic and perform well without fine-tuning. See https://github.com/facebookresearch/LASER/issues/3#issuecomment-404175463.

## Credits

Thanks a lot to the creators of [LASER](https://github.com/facebookresearch/LASER) for open-sourcing the code of LASER and releasing the pre-trained models. All the kudos should go to them 👏.

A big thanks to the creators of [Sacremoses](https://github.com/alvations/sacremoses) and [Subword Neural Machine Translation](https://github.com/rsennrich/subword-nmt/) for their great packages.

## Testing

The first thing you'll need is [Poetry](https://github.com/sdispater/poetry). Please refer to the [installation guidelines](https://poetry.eustace.io/docs/#installation).

Clone this repository and install the project:
```
poetry install -E zh -E ja
```

To run the tests:
```
poetry run pytest
```

### Testing the similarity between the embeddings computed with LASER and laserembeddings

First, install the project with the extra dependencies (Chinese and Japanese support):
```
poetry install -E zh -E ja
```

Then, download the test data:
```
poetry run python -m laserembeddings download-test-data
```

👉 If you want to know more about the contents and the generation of the test data, check out the [laserembeddings-test-data](https://github.com/yannvgn/laserembeddings-test-data) repository.

Then, run the test with `SIMILARITY_TEST` env. variable set to `1`.

```
SIMILARITY_TEST=1 poetry run pytest tests/test_laser.py
```

Now, have a coffee ☕️ and wait for the test to finish.

The similarity report will be generated here: [tests/report/comparison-with-LASER.md](tests/report/comparison-with-LASER.md).
