"""Provids API to images generated in simulations and those from the experiment.  """

import json
import pathlib
import abc
import pathlib

import cv2
import numpy as np

from slider import cv2_slider
from statistics import timeit



class ImgSet(object):
    """Keeps series of images with common origin"""

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

    def get_dot_for_x(self, x):
        for dat in self:
            if dat.kind == 'dot' and dat.x == x:
                return dat
        raise IndexError(f"No 'dot' image in {self} for requested x {x}")

    @classmethod
    def load_simulation_dataset(cls, data_path, dataset_name):
        """
        This method assumes that out of focus distance x (in mm) is written
        in filenames as imgname_x.(bmp / JPG)
        :param data_path    path directory containing all data
        """
        imgs = []
        for p in pathlib.Path(data_path).iterdir():
            if p.is_file() and (p.suffix == '.bmp' or p.suffix == ".JPG"):
                x = int(p.stem.split('_')[-1])  # reading x from image filename
                imgs.append(SimulationImg('dot', p, x))
        return cls(dataset_name, imgs, data_path)

    @classmethod
    def load_experimental_dataset(cls, json_path, data_path):
        """
        :param json_path    path to json file containing data
        :param data_path    path directory containing all data
        """
        data_path = pathlib.Path(data_path).resolve()

        with open(json_path, 'r') as f:
            dataset = json.load(f)

        img_sets = {}
        for dataset_name in dataset:
            focus_reference = dataset[dataset_name]['focus_reference']
            dataset_path = data_path / dataset_name
            imgs = []
            for i in dataset[dataset_name]['imgs']:
                im_path = dataset_path / ('_MG_' + str(i['name']) + '.jpg')
                imgs.append(ExperimentalImg(i['kind'],
                                            im_path,
                                            i['x'] - focus_reference,
                                            i['t'],
                                            i.get('f'),
                                            ))

            img_sets[dataset_name] = cls(dataset_name, imgs, path=dataset_path)
        return img_sets


class Img(abc.ABC):
    """Abstract image representation. Use ExperimentalImg or SimulationImg."""

    @abc.abstractmethod
    def __init__(self, kind, path, x):
        """
        :param kind     ['img', 'dot']
        :param path     str or Path pointing to image file
        :param x        deblured distance (counting from the focal point)
        """
        self.kind = kind
        self.path = path
        self.x = x
        self.im = None  # waits for self.load_image()

    def __str__(self):
        return f'{self.__class__.__name__}: {self.kind} at {self.x}mm'

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.kind} on {self.path} at {self.x}mm'

    def load_image(self, dtype=None, color_mode=0):
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

    def resize_im(self, scale_down_by):
        """Scales down self.im
        :param scale_down_by    int or float to divide both x and y by
        """
        output = None
        scaled_h = np.round(self.im.shape[0] // scale_down_by).astype("int")
        scaled_w  = np.round(self.im.shape[1] // scale_down_by).astype("int")
        print('{} resized from {}x{} to: {}x{} '.format(
            self.path,
            self.im.shape[0],
            self.im.shape[1],
            scaled_h,
            scaled_w))
        im = np.ndarray(shape=(scaled_w, scaled_h))
        self.im = cv2.resize(self.im, im.shape)


class ExperimentalImg(Img):
    """represents a shot and image"""
    def __init__(self, kind, path, x, t, f):
        """
        :param kind     ['img', 'dot']
        :param path     str or Path pointing to image file
        :param x        deblured distance (counting from the focal point)
        :param t        1 over time -> shot exposure
        :param f        #f
        """
        super().__init__(kind, path, x)
        self.time = t
        self.f_number = f if f else 5.6
        self.im = None

    def _find_circle(self):

        param2_max = 200
        dp_max = 5
        minRadius_max = 50
        minDist_max = 20
        bin_thresh_max = 255

        @cv2_slider(param2=param2_max, dp=dp_max, bin_thresh=bin_thresh_max)
        def interactive_circle_finder(im, bin_thresh=10, **kwargs):
            for key in kwargs.keys():
                if kwargs[key] < 1:
                    kwargs[key] = 1
            # only im will be visible for user
            im_mod = cv2.medianBlur(im, ksize=5)
            _, im_mod = cv2.threshold(im_mod, bin_thresh, 255, cv2.THRESH_BINARY)
            im_mod = cv2.morphologyEx(im_mod, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
            circles = cv2.HoughCircles(im_mod, cv2.HOUGH_GRADIENT, **kwargs)
            if circles is None or circles[0][0][2] == 0:
                return im, None
            circles = np.round(circles[0]).astype("int")
            for i, (x, y, r) in enumerate(circles):
                cv2.circle(im, (x, y), r, (255, 255, 255), 1)
                cv2.rectangle(im, (x - 2, y - 2), (x + 2, y + 2), (255, 255, 255), -1)
                cv2.putText(im, f"({x}, {y}), radius: {r}",
                        (3, im.shape[0]-20*i), cv2.FONT_HERSHEY_PLAIN,
                        3, (255,255,255), 1);
            return im, circles

        im, circles = interactive_circle_finder(self.im,
                                dp=2,
                                minDist=3,
                                param1=100, # dummy cannyHighEdgeThreshold
                                param2=45,
                                minRadius=1,
                                bin_thresh=8
                                )
        if circles is None:
            return None
        chosen_circle = circles[0]
        return chosen_circle


    def _find_circle_old(self):
        if self.im is None:
            raise RuntimeError('self.im not loaded my ladies')

        # slider case ==================================== end

        circles = []
        cannyHighEdgeThreshold = 100
        param2 = 100

        while (cannyHighEdgeThreshold > 20):
            print(f'Searching for circles... ')
            circles = cv2.HoughCircles(self.im,
                    cv2.HOUGH_GRADIENT,
                    dp=2,
                    minDist=3,
                    param1=cannyHighEdgeThreshold,
                    param2=param2,
                    minRadius=1)

            if circles is not None and circles[0][0][2] != 0:
                break

            cannyHighEdgeThreshold -= 5
            print(f'No circles found. Changing high threshold of CED (param1) to {cannyHighEdgeThreshold}')

        else:
            cv2.imshow("No circles found", self.im)
            cv2.waitKey()
            return None

        print(f'circles: \n {circles}')
        circles = np.round(circles[0]).astype("int")
        print(f'{len(circles)} circles found ')

        chosen_circle = None
        while chosen_circle is None:
            output = self.im.copy()
            for i, (x, y, r) in enumerate(circles):
                cv2.circle(output, (x, y), r, (255, 255, 255), 1)
                cv2.rectangle(output, (x - 4, y - 4), (x + 4, y + 4), (255, 255, 255), -1)
                cv2.putText(output, f"({x}, {y}), radius: {r}", (3, self.im.shape[0]-4), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1);
                cv2.imshow("Preview board", output)
                key = cv2.waitKey(0)
                if key == 13:  # enter
                    chosen_circle = circles[i]
                    break

        return chosen_circle

    def read_a(self):
        """Calculates CoC from horizontal image crossection at half for height
        :param im           numpy array image
        :return             diameter of middle maximum or -1 if circle not found
        """
        circle = self._find_circle()
        if circle is None:
            return -1
        else:
            x, y, r = circle
            return 2 * r

class SimulationImg(Img):
    """Stores simulation image and its methods"""
    def __init__(self, kind, x, path):
        super().__init__(kind, x, path)

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
    print('running main in the img.py')

    import sys
    if len(sys.argv) < 2:
        sys.exit('usage: img.py data_json data_dir')

    data = ImgSet.load_experimental_dataset(sys.argv[1], sys.argv[2])
    img1 = data['maxwell'][0]
    img1.load_image()
    img1.resize_im(scale_down_by=4)
