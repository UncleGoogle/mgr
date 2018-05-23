import pytest

from code import airy


@pytest.fixture
def im():
    import numpy as np
    return np.array([[0, 0, 0, 0, 0, 0, 0],
                     [0, 2, 1, 3, 1, 2, 0],
                     [0, 0, 0, 0, 0, 0, 0]])

def test_read_a_simple(im):
    assert airy.read_a(im, threshold=1) == 5

def test_read_a_with_dark_disks(im):
    assert airy.read_a(im, threshold=2) == 5

