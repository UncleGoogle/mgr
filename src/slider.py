""" slider decorator for openCV image display """
from functools import wraps

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

        @wraps(function)
        def wrapper(im, *args, **kwargs):
            """ATTENTION: im is openCV (numpy) image to pass its clone every refresh"""

            cv2.namedWindow(win, cv2.WINDOW_NORMAL)  # resizable window

            other_kwargs = {}
            slider_kwargs = {}

            # TODO use inspect.getatrspec(function) instead to get defaults from func definition
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

                for key, value in slider_kwargs.items():
                    slider_kwargs[key] = cv2.getTrackbarPos(key, win)

                if prev != slider_kwargs:
                    res = function(im.copy(), *args, **{**slider_kwargs, **other_kwargs})
                    if type(res) == tuple and len(res) > 1:
                        res_im, *rest = res
                    else:
                        res_im = res
                    cv2.imshow(win, res_im)

                key = cv2.waitKey(300)
                if key == 13:  # enter
                    cv2.destroyWindow(win)
                    return (slider_kwargs, res_im, rest)
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

