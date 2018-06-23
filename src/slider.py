""" slider decorator for openCV image display """
from functools import wraps

import cv2


def cv2_slider(slider):
    """Slider docstring"""
    def decorator(function):
        @wraps(function)  # to don't lose decorated function docsting and name
        def wrapper(*args, **kwargs):

            cv2.namedWindow('preview', cv2.WINDOW_NORMAL)  # resizable window

            other_kwargs = {}
            slider_kwargs = {}
            nothing = lambda _: None
            for key, value in kwargs.items():
                if key == slider:
                    try:
                        cv2.createTrackbar(key, 'preview', value, 100, nothing)
                        slider_kwargs[key] = value
                    except TypeError as e:
                        raise TypeError(f'wrong value for "{key}" in slider; ' + str(e))
                else:
                    other_kwargs[key] = value



            # low_tresh = 10

            # cv2.createTrackbar('_', 'preview', 0, 100, nothing)  # cv2 bug workaround
            # cv2.createTrackbar(slider, 'preview', low_tresh, 100, nothing)
            cv2.imshow('preview', im)

            res = function(*args, **{**slider_kwargs, **other_kwargs})

            while True:

                for key, value in slider_kwargs.items():
                    slider_kwargs[key] = cv2.getTrackbarPos(key, 'preview')
                    print('updating tracker position to ', slider_kwargs[key])

                res = function(*args, **{**slider_kwargs, **other_kwargs})

                cv2.imshow('preview', res)

                key = cv2.waitKey(1000)
                if key == 27 or key == 13:
                    break

        return wrapper
    return decorator


if __name__ == "__main__":
    import sys

    im = cv2.imread(sys.argv[1], 0)
    if im is None:
        raise OSError('image {} load failed!'.format(self.path))

    @cv2_slider("gumiak")
    def testSlider(name, other, gumiak=10):
        """Original docsting"""
        cv2.circle(im, (200, 200), radius, (255, 255, 255))
        return im

    radius = 50
    # cv2.imshow(im)
    # cv2.waitKey()
    testSlider('TheName', other=3, gumiak=4)

    # def auto_crop(self, margin):
    #     """Crops image leaving given margin from both side.
    #     :param margin       how many pixels leave from each side of countured rect
    #     """
    #     cv2.namedWindow('preview', cv2.WINDOW_NORMAL)  # resizable window

    #     im = cv2.GaussianBlur(self.im, (7,7), 0)

    #     low_tresh = 10
    #     high_tresh = 30
    #     nothing = lambda _: None
    #     cv2.createTrackbar('_', 'preview', 0, 100, nothing)
    #     cv2.createTrackbar('low_tresh', 'preview', low_tresh, 100, nothing)
    #     cv2.createTrackbar('high_tresh', 'preview', high_tresh, 200, nothing)

    #     l, h = low_tresh, high_tresh
    #     outdated = True
    #     while True:
    #         if outdated:
    #             outdated = False
    #             low_tresh, high_tresh = l, h
    #             print('compute')
    #             edges = cv2.Canny(im, low_tresh, high_tresh)
    #             _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #             # cv2.imshow('preview', edges)

    #             area_max = 0
    #             for c in contours:
    #                 xx, yy, ww, hh = cv2.boundingRect(c)
    #                 area = ww * hh
    #                 if area > area_max:
    #                     area_max = area
    #                     x, y, w, h = xx, yy, ww, hh
    #             if area_max == 0:
    #                 raise RuntimeError('No conturs for crop found')

    #             y_from = max(y - margin, 0)
    #             x_from = max(x - margin, 0)
    #             y_to = min(y + h + margin, self.im.shape[1])
    #             x_to = min(x + w + margin, self.im.shape[0])
    #             print(self.im.shape, y_from, x_from, 'to:', x_to, y_to)
    #             cv2.rectangle(edges,(x_from,y_from),(x_from,x_to),(255,255,255),3)
    #             cv2.imshow('preview', edges); cv2.waitKey()
    #             cropped_im = im[y_from : y_to, x_from : x_to]

    #         key = cv2.waitKey(300)
    #         if key == 27 or key == 13:
    #             break

    #         l = cv2.getTrackbarPos('low_tresh', 'preview')
    #         h = cv2.getTrackbarPos('high_tresh', 'preview')
    #         outdated = (low_tresh, high_tresh) != (l, h)
    #         print('outdated: {}'.format(outdated))


    #     print(f'Image is cropped from {self.im.shape} to {cropped_im.shape}')
    #     cv2.imshow('cropped_im', cropped_im); cv2.waitKey()
    #     self.im = cropped_im
    #     cv2.destroyAllWindows()
