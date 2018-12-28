import argparse
import os
import cv2
from slider import cv2_slider

@cv2_slider(threshold=100)
def equalizeHist(im):
    TODO
    return cv2.equalizeHist(im)

# /home/mieszko/Desktop/box/exp3_data/serie1/holo/imgs
parser = argparse.ArgumentParser()
parser.add_argument('experimentalDir', type=str, help="directory with experimental images")
parser.add_argument('--crop_size', type=int, help="rectangular selection for crop")
parser.add_argument('--x', type=int, help="x coordinate of left up corner of rectangle for crop")
parser.add_argument('--y', type=int, help="y coordinate of left up corner of rectangle for crop")
parser.add_argument('--output', type=str, help="where to store results")
args = parser.parse_args()

processed = []
for root, dirs, files in os.walk(args.experimentalDir):
    for file in files:
        im = cv2.imread(root+'/'+file)
        if im is None:
            raise RuntimeError('cv2 imread failed!')

        # zeroing blue and green channel
        im[:,:,0] = 0
        im[:,:,1] = 0
        # convert to grayscale
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # crop to uniform fragment
        # if args.crop_size:
        #     im = im[args.x : args.x + args.crop_size, args.y : args.y + args.crop_size]
        # else: # (1024 in right down corner) or: use slider (ToDo)
        #     im = im[im.shape[0]-1024 : im.shape[0], im.shape[1]-1024 : im.shape[1]]
        # stretch histogram
        im = cv2.equalizeHist(im[:,:,2])
        processed.append((file, im))
    break

if args.output:
    print(f'saving processed images in {args.output}...')
    for filename, im in processed:
        filename = args.output + '/' + filename[:-4] + '_s.jpg'
        cv2.imwrite(filename, im)
else:
    cv2.namedWindow('preview', cv2.WINDOW_NORMAL)  # resizable window
    for _, im in processed:
        cv2.imshow('preview', im)
        cv2.waitKey()
