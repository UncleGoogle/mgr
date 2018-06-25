""" slider decorator for openCV image display """
from functools import wraps

import cv2


def cv2_slider(sl_window="Preview", **sliders):
    """Decorator to make fast GUI sliders for any openCV func. that returns an image
    :param sl_window        str with window name
    :param **sliders        attr1=max_val1, attr2=max_val2, ...
    """
    for key in sliders:
        if type(sliders[key]) is not int:
            raise TypeError(f"Integer expected, but `{key}` type is {type(sliders[key])}")

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            cv2.namedWindow(sl_window, cv2.WINDOW_NORMAL)  # resizable window

            other_kwargs = {}
            slider_kwargs = {}
            for key, value in kwargs.items():
                if key in sliders:
                    print(f'creating trackbar for {key}; initial tracker position: {value}')
                    cv2.createTrackbar(key, sl_window, value, sliders[key], lambda _: None)
                    slider_kwargs[key] = value
                else:
                    other_kwargs[key] = value

            prev = None
            while True:

                for key, value in slider_kwargs.items():
                    slider_kwargs[key] = cv2.getTrackbarPos(key, sl_window)

                if prev != slider_kwargs:
                    print(f'updating view')
                    res = function(*args, **{**slider_kwargs, **other_kwargs})
                    cv2.imshow(sl_window, res)

                key = cv2.waitKey(500)
                if key == 27 or key == 13:
                    break

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

    @cv2_slider("Ladies & gentelmens! This is <=== Slider demo ===>",
                radius=radius_max,
                x=x_max,
                y=y_max
                )
    def testSlider(name, x, y, radius=20):
        """Original docsting"""
        cv2.circle(im, (x, y), radius, (255, 255, 255), 5)
        return im

    radius = 50
    # cv2.imshow(im)
    # cv2.waitKey()
    testSlider('TheName',
                x=int(im.shape[0] // 5),
                y=int(im.shape[1] // 4),
                radius=20)
