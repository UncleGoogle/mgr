""" slider decorator for openCV image display """
from functools import wraps
from pathlib import PurePath
import numpy as np

import cv2


def cv2_slider(win="Preview", **sliders):
    """Decorator to make fast-prototype GUI sliders for any openCV func. that 
    returns an image. Minimum slider values are always 0.
    ATTENTION: you have to define opencv image in the wrapped function as first arg
    :param win                 str with window name
    :param **sliders           attr1=max_val1, attr2=max_val2, ...
    """
    for key in sliders:
        if type(sliders[key]) is not int:
            raise TypeError(f"Integer expected, but `{key}` type is {type(sliders[key])}")

    def decorator(function):

        def _find_im(output, func):
            if type(output) is np.ndarray and output.ndim == 2:
                return output
            else:
                try:
                    iter(output)
                except TypeError:
                    raise TypeError(f'Image for slider not found. Return output image or iterable with that image in the {func.__name__}.')
                else:
                    for i in output:
                        if type(i) is np.ndarray and i.ndim == 2:
                            print('Numpy image found in callback')
                            return i
                    else:
                        raise TypeError(f'Add output image as callback in the decorated funtion.')

        @wraps(function)
        def wrapper(im, *args, **kwargs):
            if type(im) is not np.ndarray:
                raise TypeError('First argument is not numpy image')

            cv2.namedWindow(win, cv2.WINDOW_NORMAL)  # resizable window

            other_kwargs = {}
            slider_kwargs = {}

            for key, value in kwargs.items():
                if key in sliders:
                    print(f'creating trackbar for {key}; initial tracker position: {value}')
                    cv2.createTrackbar(key, win, value, sliders[key], lambda _: None)
                    slider_kwargs[key] = value
                else:
                    other_kwargs[key] = value

            prev = None
            rest = None
            while True:
                for key in slider_kwargs.keys():
                    slider_kwargs[key] = cv2.getTrackbarPos(key, win)

                if prev != slider_kwargs:
                    result = function(im.copy(), *args, **{**slider_kwargs, **other_kwargs})
                    res_im = _find_im(result, function)
                    cv2.imshow(win, res_im)

                key = cv2.waitKey(300)
                if key == 13:  # enter
                    cv2.destroyWindow(win)
                    name = PurePath('.') / 'slider_results' / (str(function.__name__) + str(slider_kwargs) + '.png')
                    print('tries to write file:', str(name))
                    cv2.imwrite(str(name), res_im)
                    return result
                prev = slider_kwargs.copy()

        return wrapper
    return decorator


if __name__ == "__main__":
    import sys

    im = cv2.imread(sys.argv[1], 0)
    if im is None:
        raise OSError('image {} load failed!'.format(self.path))

    x_max = im.shape[0]
    y_max = im.shape[1]
    radius_max = (im.shape[0] + im.shape[1]) // 2

    window_name = "Ladies & gentelmens! This is <=== Slider demo ===>"

    @cv2_slider(window_name, radius=radius_max, x=x_max, y=y_max)
    def testSlider(image, x, y, radius=20):
        """Original docsting"""
        cv2.circle(image, (x, y), radius, (255, 255, 255), -1)
        return (x,y,radius), image

    chosen_params, chosen_result = testSlider(im,
                               x=int(im.shape[0] // 5),
                               y=int(im.shape[1] // 4),
                               radius=20)

    cv2.namedWindow(f'Chosen result: {chosen_params}', cv2.WINDOW_NORMAL)
    cv2.imshow(f'Chosen result: {chosen_params}', chosen_result)
    cv2.waitKey(0)

