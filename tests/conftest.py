import os

import pytest


@pytest.fixture
def test_data():
    import numpy as np
    test_data_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  'data', 'laserembeddings-test-data.npz')

    return np.load(test_data_file) if os.path.isfile(test_data_file) else None
