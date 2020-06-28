import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('img.jpg',0)

# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()

# find and draw the keypoints
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)
img1 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
cv.imwrite('out.png',img1)