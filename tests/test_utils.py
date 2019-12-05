from laserembeddings.utils import sre_performance_patch


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
