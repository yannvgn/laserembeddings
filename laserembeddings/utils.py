__all__ = ['sre_performance_patch']


class sre_performance_patch:
    """
    Patch fixing https://bugs.python.org/issue37723 for Python 3.7 (<= 3.7.4)
    and Python 3.8 (<= 3.8.0 beta 3)
    """

    def __init__(self):
        self.sre_parse = None
        self.original_sre_parse_uniq = None

    def __enter__(self):
        # pylint: disable=import-outside-toplevel
        import sys

        if self.original_sre_parse_uniq is None and (
                0x03070000 <= sys.hexversion <= 0x030704f0
                or 0x03080000 <= sys.hexversion <= 0x030800b3):
            try:
                import sre_parse
                self.sre_parse = sre_parse
                # pylint: disable=protected-access
                self.original_sre_parse_uniq = sre_parse._uniq
                sre_parse._uniq = lambda x: list(dict.fromkeys(x))
            except (ImportError, AttributeError):
                self.sre_parse = None
                self.original_sre_parse_uniq = None

    def __exit__(self, type_, value, traceback):
        if self.sre_parse and self.original_sre_parse_uniq:
            # pylint: disable=protected-access
            self.sre_parse._uniq = self.original_sre_parse_uniq
            self.original_sre_parse_uniq = None
