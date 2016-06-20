## Torch to Caffe Model Converter
(forked from [Cysu's branch](https://github.com/Cysu/fb-caffe-exts), originally from [fb-caffe-exts](https://github.com/facebook/fb-caffe-exts)) 

### Get Started
0. An easy option for fast using is launch an AWS EC2 g2.x2large instance I created. Choose N.California Sever and search the instance name of FB-Torch2Caffe (ami-03542e63). 
0. The package currently only works on Ubuntu 14.04. Please make sure Torch and Caffe (with pycaffe and python layer) are correctly installed.
0. Download the code and install the dependencies
  ```bash
  git clone https://github.com/zhanghang1989/fb-caffe-exts.git
  sudo bash install-dep.sh
  ```
  
0. Convert your first model:
  ```bash
  th convert.lua torch_model.t7b
  ```
  
0. Or custormize the conversion:
  ```bash
  th torch2caffe/torch2caffe.lua --input torch_model.t7b --preprocessing --prepnv.lua --prototxt name.prototxt --caffemodel name.caffemodel --input_dims 1 3 64 256
  ```

###. The Layers We Added Support
0. ``ELU`` 
0. ``SpatialDropout`` We scale the weights of previous layers by (1-p) to hide the difference between torch and caffe. 
0. ``SpatialMaxPooling`` It has slightly different behaviours in Torch and Caffe. Torch uses floor(n/s+1) and Caffe uses floor(n/s). Therefore, only the conversion of even featuremap size is supported. 
0. ``SpatialBatchNormalization`` Caffe BatchNorm doesn't have bias. We only support non-affine BN. Alternatively, you can convert it into a customized version of BN as in [Cysu's branch](https://github.com/Cysu/fb-caffe-exts).

### 
