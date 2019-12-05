from io import StringIO

from laserembeddings.utils import BPECodesAdapter, sre_performance_patch


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


def test_sre_performance_patch():
    #pylint: disable=protected-access
    try:
        import sre_parse
        uniq = sre_parse._uniq

        with sre_performance_patch():
            assert sre_parse._uniq(['5', '2', '3', '2', '5',
                                    '1']) == ['5', '2', '3', '1']

        # make sure the original sre_parse._uniq was restored
        assert sre_parse._uniq == uniq
    except (ImportError, AttributeError):
        pass
