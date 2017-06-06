
import numpy as np

# print np.log(1/0)
print np.log(1e-63/1)



import cv2
img = cv2.imread('/local/home/cpchung/Desktop/fasterRCNN-voc2007.png',0)
print img.shape

cv2.rectangle(img,(510,128),(384,0),(0,255,0),3)

cv2.imwrite('./image.jpg',img)

# cv2.imshow('image',img)
# cv2.waitKey()