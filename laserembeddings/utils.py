from typing import TextIO

__all__ = ['BPECodesAdapter']


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
