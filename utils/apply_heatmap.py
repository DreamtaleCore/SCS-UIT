import cv2
import os
import argparse
import numpy as np


parser = argparse.ArgumentParser()


parser.add_argument('--img', type=str, help='Path to the image file.')
parser.add_argument('--hm',  type=str, help='Path to the heat map file.')
parser.add_argument('--out', type=str, help="output image folder", default=r'F:\tmp\results')
parser.add_argument('--r',   type=float, default=0.4, help='ratio of the heat map on the image.')
opts = parser.parse_args()

if not os.path.exists(opts.out):
    os.makedirs(opts.out)

img = cv2.imread(opts.img)
hm  = cv2.imread(opts.hm)

h, w = img.shape[:2]
hm   = cv2.resize(hm, (w, h))

ret  = np.uint8(img / 255 * (1 - opts.r) + hm / 255 * opts.r)


