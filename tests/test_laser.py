import os

import pytest
import numpy as np

from laserembeddings import Laser

SIMILARITY_TEST = os.getenv('SIMILARITY_TEST')


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


def test_similarity(test_data):
    if not SIMILARITY_TEST:
        pytest.skip("SIMILARITY_TEST not set")

    if not test_data:
        raise FileNotFoundError(
            'laserembeddings-test-data.npz is missing, run "python -m laserembeddings download-test-data" to fix that'
        )

    report = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'report', 'comparison-with-LASER.md')

    laser = Laser()

    with open(report, 'w', encoding='utf-8') as f_report:

        f_report.write(
            '# Comparison of the embeddings computed with original LASER with the embeddings computed with this package\n'
        )
        f_report.write(
            '| |language|avg. cosine similarity|min. cosine similarity|\n')
        f_report.write(
            '|-|--------|----------------------|----------------------|\n')

        for lang in test_data['langs']:

            if lang in ('cmn', 'wuu', 'yue', 'zh', 'jpn', 'ja', 'el'):
                # language not supported, ignoring
                continue

            sents = test_data[f'{lang}_sentences']
            orig_embeddings = test_data[f'{lang}_embeddings']
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
