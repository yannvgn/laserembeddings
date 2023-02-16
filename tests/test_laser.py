#for testing Laser2 embeddings only the function test_similarity was changed
#it might be necessary to adapt the other tests and make it executable with pytest

import os

import pytest
import numpy as np

#this import only makes sense if the package is updated
#from laserembeddings import Laser

import sys
sys.path.insert(1, '../laserembeddings')
from laserembeddings.laser import Laser

SIMILARITY_TEST = os.getenv('SIMILARITY_TEST')
SKIP_ZH = os.getenv('SKIP_ZH')
SKIP_JA = os.getenv('SKIP_JA')

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'data')

def test_laser():
    with open(Laser.DEFAULT_ENCODER_FILE, 'rb') as f_encoder:
        laser = Laser(
            Laser.DEFAULT_BPE_CODES_FILE,
            None,
            f_encoder,
        )
        assert laser.embed_sentences(
            ['hello world!', 'i hope the tests are passing'],
            lang='en').shape == (2, 1024)
        assert laser.embed_sentences(['hello world!', "j'aime les pâtes"],
                                     lang=['en', 'fr']).shape == (2, 1024)
        assert laser.embed_sentences('hello world!',
                                     lang='en').shape == (1, 1024)

        with pytest.raises(ValueError):
            laser.embed_sentences(['hello world!', "j'aime les pâtes"],
                                  lang=['en'])


def test_zh():
    if SKIP_ZH:
        pytest.skip("SKIP_ZH is set")
    laser = Laser()
    assert laser.embed_sentences(['干杯！'], lang='zh').shape == (1, 1024)


def test_ja():
    if SKIP_JA:
        pytest.skip("SKIP_JA is set")
    laser = Laser()
    assert laser.embed_sentences(['乾杯！'], lang='ja').shape == (1, 1024)


def test_similarity(mode='spm'):
    #if not SIMILARITY_TEST:
    #    pytest.skip("SIMILARITY_TEST not set")

    report = None

    if mode =='spm':
        print("Comparing to Laser2 embeddings with SPM")
        if not os.path.isfile(os.path.join(TEST_DATA_DIR, "laserembeddings2-test-data.npz")):
            raise FileNotFoundError(
            'test data file is missing, run "python -m laserembeddings download-test-data" to fix that'
        )
        test_data = os.path.join(TEST_DATA_DIR, "laserembeddings2-test-data.npz")
        report = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'report', 'comparison-with-LASER2.md')
        laser = Laser(mode='spm')
    if mode == 'bpe':
        print("Comparing to Laser embeddings with BPE")
        if not os.path.isfile(os.path.join(TEST_DATA_DIR, "laserembeddings-test-data.npz")):
            raise FileNotFoundError(
            'test data file is missing, run "python -m laserembeddings download-test-data" to fix that'
        )
        test_data = os.path.join(TEST_DATA_DIR, "laserembeddings-test-data.npz")
        report = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'report', 'comparison-with-LASER.md')
        laser = Laser(mode='bpe')

    with open(report, 'w', encoding='utf-8') as f_report:

        f_report.write(
            '# Comparison of the embeddings computed with original LASER with the embeddings computed with this package\n'
        )
        f_report.write(
            '| |language|avg. cosine similarity|min. cosine similarity|\n')
        f_report.write(
            '|-|--------|----------------------|----------------------|\n')

        test_data = np.load(test_data)

        for lang in test_data['langs']:

            sents = test_data[f'{lang}_sentences']
            orig_embeddings = test_data[f'{lang}_embeddings']

            #lang is not used in spm mode
            embeddings = laser.embed_sentences(sents, lang)

            assert embeddings.shape == orig_embeddings.shape

            cosine_similarities = np.sum(
                orig_embeddings * embeddings,
                axis=1) / (np.linalg.norm(orig_embeddings, axis=1) *
                           np.linalg.norm(embeddings, axis=1))

            similarity_mean = np.mean(cosine_similarities)
            similarity_min = np.min(cosine_similarities)

            f_report.write(
                f'|{"✅" if similarity_min > 0.99999 else "⚠️" if similarity_mean > 0.99 else "❌"}|{lang}|{similarity_mean:.5f}|{similarity_min:.5f}|\n'
            )

if __name__ == "__main__":
    # set mode
    test_similarity(mode='spm')