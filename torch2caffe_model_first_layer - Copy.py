import caffe
import numpy as np

net = caffe.Net('model_5_cpu-pre.prototxt','model_5_cpu_2.caffemodel',caffe.TEST)

print(net.params['caffe.Scale_1'][0].data)
print(net.params['caffe.Scale_1'][1].data)

print(net.params['caffe.Scale_1'][0].data.shape)
print(net.params['caffe.Scale_1'][1].data.shape)