import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
from cv_util.b64_visualize import show_tile
from cv_util.image_util import resize_image_preserve_ratio

td = open("nn.SpatialMaxPooling1.pkl","rb")
td_pool = pickle.load(td)["nn.SpatialMaxPooling1"]
cf = open("caffe.Pooling_3.pkl","rb")
cf_pool = pickle.load(cf)["caffe.Pooling_3"]
print td_pool.sum(),cf_pool.sum()
file = open("diff.txt","w")
print >> file, td_pool-cf_pool
print td_pool.shape,cf_pool.shape

param = td_pool-cf_pool
param = np.transpose(param,(1,0,2,3))
param_images = []
for i in range(param.shape[0]):
    img = param[i,:,:,:]
    img = np.transpose(img,(1,2,0))
    #img = img - img.min()
    #img = img / img.max()
    img = img*255
    img = img.astype(np.uint8)
    param_images.append(img)

param_images = []
for i in range(param.shape[0]):
    img = param[i,:,:,:]
    img = np.transpose(img,(1,2,0))
    #img = img - img.min()
    #img = img / img.max()
    img = img*255
    img = img.astype(np.uint8)
    param_images.append(img)

param_images = []
for i in range(param.shape[0]):
    img = param[i,:,:,:]
    img = np.transpose(img,(1,2,0))
    #img = img - img.min()
    #img = img / img.max()
    img = img*255
    img = img.astype(np.uint8)
    param_images.append(img)

show_filters = show_tile(param_images,57,1)
plt.imsave("diff.png",show_filters)
