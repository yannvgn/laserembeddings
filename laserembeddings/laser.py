from typing import Dict, Any, Union, List, Optional
from io import TextIOBase, BufferedIOBase
import os

import numpy as np

from .preprocessing import Tokenizer, BPE
from .embedding import BPESentenceEmbedding
from .utils import sre_performance_patch

__all__ = ['Laser']


class Laser:
    """
    End-to-end LASER embedding.

    The pipeline is: ``Tokenizer.tokenize`` -> ``BPE.encode_tokens`` -> ``BPESentenceEmbedding.embed_bpe_sentences``

    Args:
        bpe_codes (str or TextIOBase, optional): the path to LASER's BPE codes (``93langs.fcodes``),
            or a text-mode file object. If omitted, ``Laser.DEFAULT_BPE_CODES_FILE`` is used.
        bpe_codes (str or TextIOBase, optional): the path to LASER's BPE vocabulary (``93langs.fvocab``),
            or a text-mode file object. If omitted, ``Laser.DEFAULT_BPE_VOCAB_FILE`` is used.
        encoder (str or BufferedIOBase, optional): the path to LASER's encoder PyToch model (``bilstm.93langs.2018-12-26.pt``),
            or a binary-mode file object. If omitted, ``Laser.DEFAULT_ENCODER_FILE`` is used.
        tokenizer_options (Dict[str, Any], optional): additional arguments to pass to the tokenizer.
            See ``.preprocessing.Tokenizer``.
        embedding_options (Dict[str, Any], optional): additional arguments to pass to the embedding layer.
            See ``.embedding.BPESentenceEmbedding``.
    
    Class attributes:
        DATA_DIR (str): the path to the directory of default LASER files.
        DEFAULT_BPE_CODES_FILE: the path to default BPE codes file.
        DEFAULT_BPE_VOCAB_FILE: the path to default BPE vocabulary file.
        DEFAULT_ENCODER_FILE: the path to default LASER encoder PyTorch model file.
    """

    DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'data')
    DEFAULT_BPE_CODES_FILE = os.path.join(DATA_DIR, '93langs.fcodes')
    DEFAULT_BPE_VOCAB_FILE = os.path.join(DATA_DIR, '93langs.fvocab')
    DEFAULT_ENCODER_FILE = os.path.join(DATA_DIR,
                                        'bilstm.93langs.2018-12-26.pt')

    def __init__(self,
                 bpe_codes: Optional[Union[str, TextIOBase]] = None,
                 bpe_vocab: Optional[Union[str, TextIOBase]] = None,
                 encoder: Optional[Union[str, BufferedIOBase]] = None,
                 tokenizer_options: Optional[Dict[str, Any]] = None,
                 embedding_options: Optional[Dict[str, Any]] = None):

        if tokenizer_options is None:
            tokenizer_options = {}
        if embedding_options is None:
            embedding_options = {}

        if bpe_codes is None:
            if not os.path.isfile(self.DEFAULT_BPE_CODES_FILE):
                raise FileNotFoundError(
                    '93langs.fcodes is missing, run "python -m laserembeddings download-models" to fix that'
                )
            bpe_codes = self.DEFAULT_BPE_CODES_FILE
        if bpe_vocab is None:
            if not os.path.isfile(self.DEFAULT_BPE_VOCAB_FILE):
                raise FileNotFoundError(
                    '93langs.fvocab is missing, run "python -m laserembeddings download-models" to fix that'
                )
            bpe_vocab = self.DEFAULT_BPE_VOCAB_FILE
        if encoder is None:
            if not os.path.isfile(self.DEFAULT_ENCODER_FILE):
                raise FileNotFoundError(
                    'bilstm.93langs.2018-12-26.pt is missing, run "python -m laserembeddings download-models" to fix that'
                )
            encoder = self.DEFAULT_ENCODER_FILE

        self.tokenizer_options = tokenizer_options
        self.tokenizers: Dict[str, Tokenizer] = {}

        self.bpe = BPE(bpe_codes, bpe_vocab)
        self.bpeSentenceEmbedding = BPESentenceEmbedding(
            encoder, **embedding_options)

    def _get_tokenizer(self, lang: str) -> Tokenizer:
        """Returns the Tokenizer instance for the specified language. The returned tokenizers are cached."""

        if lang not in self.tokenizers:
            self.tokenizers[lang] = Tokenizer(lang, **self.tokenizer_options)

        return self.tokenizers[lang]

    def embed_sentences(self, sentences: Union[List[str], str],
                        lang: Union[str, List[str]]) -> np.ndarray:
        """
        Computes the LASER embeddings of provided sentences using the tokenizer for the specified language.

        Args:
            sentences (str or List[str]): the sentences to compute the embeddings from.
            lang (str or List[str]): the language code(s) (ISO 639-1) used to tokenize the sentences
                (either as a string - same code for every sentence - or as a list of strings - one code per sentence).

        Returns:
            np.ndarray: A N * 1024 NumPy array containing the embeddings, N being the number of sentences provided.
        """
        sentences = [sentences] if isinstance(sentences, str) else sentences
        lang = [lang] * len(sentences) if isinstance(lang, str) else lang

        if len(sentences) != len(lang):
            raise ValueError(
                'lang: invalid length: the number of language codes does not match the number of sentences'
            )

        with sre_performance_patch():  # see https://bugs.python.org/issue37723
            sentence_tokens = [
                self._get_tokenizer(sentence_lang).tokenize(sentence)
                for sentence, sentence_lang in zip(sentences, lang)
            ]
            bpe_encoded = [
                self.bpe.encode_tokens(tokens) for tokens in sentence_tokens
            ]

            return self.bpeSentenceEmbedding.embed_bpe_sentences(bpe_encoded)
