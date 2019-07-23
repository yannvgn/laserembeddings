import pytest

from laserembeddings import Laser
from laserembeddings.preprocessing import Tokenizer, BPE


def test_tokenizer():
    assert Tokenizer('en').tokenize("Let's do it!") == "let 's do it !"

    with pytest.raises(NotImplementedError):
        Tokenizer(romanize=True)

    assert Tokenizer(
        'en', descape=True).tokenize("Let's do it &amp; pass that test!"
                                     ) == "let 's do it & pass that test !"

    with pytest.raises(AssertionError):
        Tokenizer(lower_case=False)


def test_bpe():
    with open(Laser.DEFAULT_BPE_VOCAB_FILE, 'r', encoding='utf-8') as f_vocab:
        bpe = BPE(Laser.DEFAULT_BPE_CODES_FILE, f_vocab)
        assert bpe.encode_tokens(
            "the tests are passing") == 'the test@@ s are passing'
