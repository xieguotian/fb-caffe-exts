import numpy as np
import cv2

def resize_image_preserve_ratio(img,scale):
    height,width = img.shape[:2]
    if height < width:
        ratio = scale / float(height)
        height = scale
        width = int(ratio * width)
    elif width < height:
        ratio = scale / float(width)
        height = int(ratio * height)
        width = scale
    else:
        height = scale
        width = scale

    img_tmp = cv2.resize(img,(width,height))
    return img_tmp