""" openCV lider module """
import sys
from functools import wraps

import cv2


def cv2_slider(sliders):
    """Slider docstring"""
    def decorator(function):
        """Inner decorator docstring"""
        @wraps(function)  # to don't lose decorated function docsting and name
        def wrapper():
            """very inner wrapper doscring"""

            # cv2.namedWindow('preview', cv2.WINDOW_NORMAL)  # resizable window

            # low_tresh = 10
            # nothing = lambda _: None
            # # cv2.createTrackbar('_', 'preview', 0, 100, nothing)  # cv2 bug workaround
            # cv2.createTrackbar('low_tresh', 'preview', low_tresh, 100, nothing)
            # cv2.imshow('preview', im)

            # while True:

            #     low_tresh = cv2.getTrackbarPos('low_tresh', 'preview')

            #     # function(im, low_tresh, *args)
            #     cv2.circle(im, (200, 200), low_tresh, (0,0,255), -1)

            #     cv2.imshow('preview', im)

            #     key = cv2.waitKey(300)
            #     if key == 27 or key == 13:
            #         break

        return wrapper
    return decorator


if __name__ == "__main__":

    @cv2_slider("arg")
    def example():
        """Original function"""

    print(example.__name__)
    print(example.__doc__)


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
