import caffe
import numpy as np

net = caffe.Net('res200-cpu.prototxt','res200-cpu.caffemodel',caffe.TEST)

layers_names = net._layer_names
layers_types = net._layer_types

first_layer_name = ''
for ix,layer_type in enumerate(layers_types):
    if layer_type == 'Convolution':
        first_layer_name = layers_names[ix]
        break

scale_factor = np.array([1.0/255.0/0.299, 1.0/255.0/0.224, 1.0/255.0/0.225 ]).reshape([1,3,1,1])[:,::-1,:,:]
net.params[first_layer_name][0].data[:] = net.params[first_layer_name][0].data[:,::-1,:,:] * scale_factor

net.save('res200-cpu_2.caffemodel')