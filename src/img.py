"""provide classes describing images generated in experiment"""

import json
import pathlib

import cv2
import numpy as np

from slider import cv2_slider


class ImgSet(object):
    """keeps series of experimental images"""

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

        img_sets = {}
        for dataset_name in dataset:
            focus_reference = dataset[dataset_name]['focus_reference']
            dataset_path = data_path / dataset_name
            imgs = []
            for i in dataset[dataset_name]['imgs']:
                im_path = dataset_path / ('_MG_' + str(i['name']) + '.JPG')
                imgs.append(Img(i['kind'],
                                im_path,
                                i['x'] - focus_reference,
                                i['t'],
                                i.get('f'),
                               ))

            img_sets[dataset_name] = cls(dataset_name, imgs, path=dataset_path)
        return img_sets


class Img(object):
    """represents a shot and image"""
    def __init__(self, kind, path, x, t, f):
        """
        :param kind     ['img', 'dot']
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

    def __str__(self):
        return f'Exp. {self.kind} at {self.x}mm'

    def _find_circle(self):
        if self.im is None:
            raise RuntimeError('self.im not loaded my ladies')

        # -------------------------------------blur image-----------------------------
        previous = self.im.copy()
        im = cv2.medianBlur(self.im, 5)
        # cv2.imshow("medianBlur", np.hstack([previous, self.im]))
        # cv2.waitKey()

        # -------------------------------------circle detection-----------------------
        circles = []
        cannyHighEdgeThreshold = 100
        param2 = 100

        while (cannyHighEdgeThreshold > 20):
            print(f'Searching for circles... ', end='')
            circles = cv2.HoughCircles(self.im, cv2.HOUGH_GRADIENT, dp=2, minDist=3,
                                       param1=cannyHighEdgeThreshold, param2=param2,
                                       minRadius=1)

            if circles is not None and circles[0][0][2] != 0:
                break

            cannyHighEdgeThreshold -= 5
            print(f'No circles found. Changing high threshold of CED to {cannyHighEdgeThreshold}')

        else:
            cv2.imshow("No circles found", self.im)
            cv2.waitKey()
            return None

        print(f'debug: row im show {circles}')
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
    print('running main in the img.py')

    import sys
    if len(sys.argv) < 2:
        sys.exit('usage: img.py data_json data_dir')

    data = ImgSet.load_from_json_factory(sys.argv[1], sys.argv[2])
    img1 = data['maxwell'][0]
    img1.load_image()
    img1.resize_im(scale_down_by=4)