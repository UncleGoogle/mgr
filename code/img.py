"""provide classes describing images generated in experiment"""

import json
import pathlib

import cv2
import numpy as np


class ImgSet(object):
    """keeps set of images common for secific purpose"""

    def __init__(self, name, imgs, path):
        """
        :param name     name of data set
        :param imgs     array of Img items
        :param path     str with dataset location (relative or abs)
        """
        self.name = name
        self.imgs = imgs
        self.path = pathlib.Path(path)
        if not self.path.exists():
            raise RuntimeError(f"given data file dosn't exists {self.path}")

    def __repr__(self):
        return '{}: {} images'.format(self.name, len(self.imgs))

    def __len__(self):
        return len(self.imgs)

    def __iter__(self):
        for i in self.imgs:
            yield i

    def __getitem__(self, i):
        return self.imgs[i]

    @classmethod
    def load_from_json_factory(cls, json_path, data_path):
        """
        :param json_path    path to json file containing data
        :param data_path    path directory containing all data
        """
        data_path = pathlib.Path(data_path).resolve()

        with open(json_path, 'r') as f:
            dataset = json.load(f)

        img_sets = []
        for dataset_name in dataset:
            focus_reference = dataset[dataset_name]['focus_reference']
            dataset_path = data_path / dataset_name
            imgs = []
            for i in dataset[dataset_name]['imgs']:
                im_path = dataset_path / ('_MG_' + str(i['name']) + '.JPG')
                imgs.append(Img(i['kind'],
                                im_path,
                                focus_reference - i['x'],
                                i['t'],
                                i.get('f'),
                               ))

            img_sets.append(cls(dataset_name, imgs, path=dataset_path))
        return img_sets


class Img(object):
    """represents a shot and image"""
    def __init__(self, kind, path, x, t, f):
        """
        :param kind     ['img', 'dot', 'sym']
        :param path     str or Path pointing to image file
        :param x        deblured distance (counting from the focal point)
        :param t        1 over time -> shot exposure
        :param f        #f
        """
        self.kind = kind
        self.path = path
        self.time = t
        self.x = x
        self.f_number = f if f else 5.6

        self.im = None

    def __repr__(self):
        return f'<Img class instance> {self.kind}: {self.path} at {self.x}mm'

    def _findCircle(self):

        if self.im is None:
            print('self.im not loaded my ladies')
            return

        # -------------------------------------blur image---------------------------
        # im = cv2.medianBlur(im, 5)
        # cv2.imshow("output", im)
        # cv2.waitKey()

        output = im.copy()

        print(f'searching for circles...')
        circles = cv2.HoughCircles(self.im,
                                cv2.HOUGH_GRADIENT,
                                dp=2,
                                minDist=4,
                                param1=100,
                                param2=100,
                                minRadius=1
                                )

        if circles is not None:
            print(f'found {circles.shape[1]} circles')
            circles = np.round(circles).astype("int")

            # debug/testing
            # for (x, y, r) in circles:
            #     cv2.circle(output, (x, y), r, (255, 255, 255), 1)
            #     cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 255), -1)
            #     cv2.imshow("output", np.hstack([im, output]))
            #     cv2.waitKey(0)

            if circles.shape[1] != 1:
                raise NotImplementedError("more than 1 circle are found")
            return circles[0]

    def read_a(self, threshold):
        """Calculates CoC from horizontal image crossection at half for height
        :param im           numpy array image
        :param threshold    [0-255]
        :return             diameter of middle maximum
        """
        x, y, r = self._findCircle()
        middle_crossection = self.im[x]
        a_px = 0
        for pixel in middle_crossection:
            if pixel >= threshold:
                a_px += 1
        return a_px

    def readImage(self, dtype=None, color_mode=0):
        """Read image files as numpy arrays using opencv python wrapper.
        :param color_mode       1 - color, 0 - greyscale(default), -1 - unchanged
        :param dtype            data type of returned numpy array (ex. np.float32)
                                If any *float* type is used, convert to [0,1] range.
                                If not given original is used.
        :return                 numpy arrays of read image
        """
        im = cv2.imread(str(self.path), color_mode)
        if im is None:
            raise OSError('image {} load failed!'.format(self.path))
        # search for normalization factor (max no. to divide by)
        max_no = np.iinfo(im.dtype).max
        if im.dtype != dtype:
            im = np.asarray(im, dtype=dtype)
        if max_no and im.dtype in [np.float16, np.float32, np.float64]:
            im = im / max_no
        self.im = im


class SimulationImg(Img):
    def __init__(self):
        super.__init__()

    def read_a(self, threshold):
        """Calculates CoC from horizontal image crossection at half for height
        :param im           numpy array image
        :param threshold    [0-255]
        :return             diameter of middle maximum
        """
        if self.im is None:
            print('im not loaded for this Img')
            return None
        middle_crossection = self.im[int(self.im.shape[0]//2)]
        a_px = 0
        for pixel in middle_crossection:
            if pixel >= threshold:
                a_px += 1
        return a_px



if __name__ == '__main__':
    """for testing purpose"""

    import sys
    if len(sys.argv) < 2:
        sys.exit('usage: img.py data_json data_dir')

    data = ImgSet.load_from_json_factory(sys.argv[1], sys.argv[2])
    img1 = data[0][0]
    img1.readImage()

    # -------------------------------------resize image-------------------------
    scaled_h, scaled_w = img1.im.shape[0]//8, img1.im.shape[1]//8
    im = np.ndarray(shape=(scaled_w, scaled_h))
    im = cv2.resize(img1.im, im.shape)
    cv2.imshow("output", im)
    cv2.waitKey()

    # -------------------------------------blur image---------------------------
    # im = cv2.medianBlur(im, 5)
    # cv2.imshow("output", im)
    # cv2.waitKey()

    output = im.copy()

    print(f'searching for circles...')
    circles = cv2.HoughCircles(im,
                               cv2.HOUGH_GRADIENT,
                               dp=2,
                               minDist=4,
                               param1=100,
                               param2=100,
                               minRadius=1
                               )

    if circles is not None:
        print(f'found {circles.shape[1]} circles')

        # convert to integers
        circles = np.round(circles).astype("int")
        print('converted to integers')

        for (x, y, r) in circles:
            # draw a circle that was detected
            cv2.circle(output, (x, y), r, (255, 255, 255), 1)
            # mark the circle center
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 255), -1)

            # show the original on the left and ouptut on the right
            cv2.imshow("output", np.hstack([im, output]))
            cv2.waitKey(0)

