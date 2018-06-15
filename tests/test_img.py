import unittest

from .context import src


class ImgTest(unittest.TestCase):
    def setUp(self):
        self.test_img = src.img.Img('img', 'test_path', 1, 10)
        test_img.readImage()
        # TODO

    def test_find_circle(self):
        pass

if __name__ == "__main__":
    unittest.main()
