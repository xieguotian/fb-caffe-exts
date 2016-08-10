export LD_PRELOAD=$HOME/caffe/.build_release/lib/libcaffe.so;

for f in *.t7b 
do 
  th convert.lua $f
done
