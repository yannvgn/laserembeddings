from io import StringIO

from laserembeddings.utils import BPECodesAdapter


def test_bpe_codes_adapter():
    test_f = StringIO(
        '#version:2.0\ne n 52708119\ne r 51024442\ne n</w> 47209692')

    adapted = BPECodesAdapter(test_f)

    assert adapted.readline() == '#version:2.0\n'
    assert adapted.readline() == 'e n\n'
    assert adapted.readline() == 'e r\n'

    for line in adapted:
        assert line == 'e n</w>\n'

    adapted.seek(0)

    for line in adapted:
        assert line == '#version:2.0\n'
        break
