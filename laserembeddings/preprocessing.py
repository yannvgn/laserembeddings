import tempfile
from typing import TextIO, Union

import fastBPE
from sacremoses import MosesPunctNormalizer, MosesTokenizer
from sacremoses.util import xml_unescape

__all__ = ['Tokenizer', 'BPE']


###############################################################################
#
# Tokenizer
#
###############################################################################


class Tokenizer:
    """
    Tokenizer.

    Args:
        lang (str): the language code (ISO 639-1) of the texts to tokenize
        lower_case (bool, optional): if True, the texts are lower-cased before being tokenized.
            Defaults to True.
        romanize (bool, optional): if True, the texts are romanized before being tokenized.
            Defaults to False. Should be True for "el" language.
        descape (bool, optional): if True, the XML-escaped symbols get de-escaped.
            Default to False.
    """

    def __init__(self,
                 lang: str = 'en',
                 lower_case: bool = True,
                 romanize: bool = False,
                 descape: bool = False):
        assert lower_case, 'lower case is needed by all the models'

        if lang in ('cmn', 'wuu', 'yue'):
            lang = 'zh'
        if lang == 'jpn':
            lang = 'ja'

        if lang == 'zh':
            raise NotImplementedError('jieba is not yet implemented')
        if lang == 'ja':
            raise NotImplementedError('mecab is not yet implemented')
        if romanize:
            raise NotImplementedError('romanize is not yet implemented')

        self.lower_case = lower_case
        self.romanize = romanize
        self.descape = descape

        self.normalizer = MosesPunctNormalizer(lang=lang)
        self.tokenizer = MosesTokenizer(lang=lang)

    def tokenize(self, text: str) -> str:
        """Tokenizes a text and returns the tokens as a string"""
        if self.lower_case:
            text = text.lower()

        # REM_NON_PRINT_CHAR
        # not implemented

        # NORM_PUNC
        text = self.normalizer.normalize(text)

        # DESCAPE
        if self.descape:
            text = xml_unescape(text)

        # MOSES_TOKENIZER

        # see: https://github.com/facebookresearch/LASER/issues/55#issuecomment-480881573
        text = self.tokenizer.tokenize(text,
                                       return_str=True,
                                       escape=False,
                                       aggressive_dash_splits=False)

        # jieba
        # MECAB
        # ROMAN_LC
        # not implemented

        return text


###############################################################################
#
# Apply BPE
#
###############################################################################


class BPE:
    """
    BPE encoder.

    Args:
        bpe_codes (str or TextIO): the path to LASER's BPE codes (``93langs.fcodes``),
            or a text-mode file object.
        bpe_codes (str or TextIO): the path to LASER's BPE vocabulary (``93langs.fvocab``),
            or a text-mode file object.
    """

    def __init__(self, bpe_codes: Union[str, TextIO],
                 bpe_vocab: Union[str, TextIO]):

        f_bpe_codes = None
        f_bpe_vocab = None

        try:
            if not isinstance(bpe_codes, str):
                f_bpe_codes = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8')
                f_bpe_codes.write(bpe_codes.read())
                bpe_codes = f_bpe_codes.name

            if isinstance(bpe_vocab, str):
                f_bpe_vocab = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8')
                f_bpe_vocab.write(bpe_codes.read())
                bpe_vocab = f_bpe_vocab.name

            self.bpe = fastBPE.fastBPE(bpe_codes, bpe_vocab)
        finally:
            if f_bpe_codes:
                f_bpe_codes.close()
            if f_bpe_vocab:
                f_bpe_vocab.close()

    def encode_tokens(self, sentence_tokens: str) -> str:
        """Returns the BPE-encoded sentence from a tokenized sentence"""
        return self.bpe.apply([sentence_tokens])[0]
