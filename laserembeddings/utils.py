from typing import TextIO

__all__ = ['BPECodesAdapter', 'sre_performance_patch']


class BPECodesAdapter:
    """
    A file object kind-of wrapper converting fastBPE codes to subword_nmt BPE codes.

    Args:
        bpe_codes_f (TextIO): the text-mode file object of fastBPE codes
    """

    def __init__(self, bpe_codes_f: TextIO):
        self.bpe_codes_f = bpe_codes_f

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.bpe_codes_f.seek(offset, whence)

    def readline(self, limit: int = -1) -> str:
        return self._adapt_line(self.bpe_codes_f.readline(limit))

    def __iter__(self):
        return self

    def __next__(self):
        return self._adapt_line(next(self.bpe_codes_f))

    @staticmethod
    def _adapt_line(line: str) -> str:
        parts = line.strip('\r\n ').split(' ')
        return ' '.join(parts[:-1]) + '\n' if len(parts) == 3 else line


class sre_performance_patch:
    """
    Patch fixing https://bugs.python.org/issue37723 for Python 3.7 (<= 3.7.4)
    and Python 3.8 (<= 3.8.0 beta 3)
    """

    def __init__(self):
        self.sre_parse = None
        self.original_sre_parse_uniq = None

    def __enter__(self):
        #pylint: disable=import-outside-toplevel
        import sys

        if self.original_sre_parse_uniq is None and (
                0x03070000 <= sys.hexversion <= 0x030704f0
                or 0x03080000 <= sys.hexversion <= 0x030800b3):
            try:
                import sre_parse
                self.sre_parse = sre_parse
                #pylint: disable=protected-access
                self.original_sre_parse_uniq = sre_parse._uniq
                sre_parse._uniq = lambda x: list(dict.fromkeys(x))
            except (ImportError, AttributeError):
                self.sre_parse = None
                self.original_sre_parse_uniq = None

    def __exit__(self, type_, value, traceback):
        if self.sre_parse and self.original_sre_parse_uniq:
            #pylint: disable=protected-access
            self.sre_parse._uniq = self.original_sre_parse_uniq
            self.original_sre_parse_uniq = None
