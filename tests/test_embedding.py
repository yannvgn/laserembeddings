from laserembeddings import Laser
from laserembeddings.embedding import BPESentenceEmbedding


def test_bpe_sentence_embedding():
    assert BPESentenceEmbedding(
        Laser.DEFAULT_ENCODER_FILE).embed_bpe_sentences(['hello', 'world'
                                                         ]).shape == (2, 1024)

    with open(Laser.DEFAULT_ENCODER_FILE, 'rb') as encoder_f:
        assert BPESentenceEmbedding(encoder_f).embed_bpe_sentences(
            ['hello', 'world']).shape == (2, 1024)
