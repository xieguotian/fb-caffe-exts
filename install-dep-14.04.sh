
echo
echo This script will install fblualib and all its dependencies.
echo It has been tested on Ubuntu 13.10 and Ubuntu 14.04, Linux x86_64.
echo

set -e
set -x


if [[ $(arch) != 'x86_64' ]]; then
    echo "x86_64 required" >&2
    exit 1
fi

issue=$(cat /etc/issue)
extra_packages=
if [[ $issue =~ ^Ubuntu\ 14\.04 ]]; then
    extra_packages=libiberty-dev
else
    echo "Ubuntu 14.04 required" >&2
    exit 1
fi

dir=$(mktemp --tmpdir -d fblualib-build.XXXXXX)

echo Working in $dir
echo
cd $dir

echo Installing required packages
echo
sudo apt-get install -y \
    git \
    curl \
    wget \
    g++ \
    automake \
    autoconf \
    autoconf-archive \
    libtool \
    libboost-all-dev \
    libevent-dev \
    libdouble-conversion-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    liblz4-dev \
    liblzma-dev \
    libsnappy-dev \
    make \
    zlib1g-dev \
    binutils-dev \
    libjemalloc-dev \
    $extra_packages \
    flex \
    bison \
    libkrb5-dev \
    libsasl2-dev \
    libnuma-dev \
    pkg-config \
    libssl-dev \
    libedit-dev \
    libmatio-dev \
    libpython-dev \
    libpython3-dev \
    python-numpy

echo
echo Cloning repositories
echo
git clone -b v0.35.0  --depth 1 https://github.com/facebook/folly
git clone -b v0.24.0  --depth 1 https://github.com/facebook/fbthrift
git clone -b v1.0 https://github.com/facebook/thpp
git clone -b v1.0 https://github.com/facebook/fblualib

echo
echo Building folly
echo

cd $dir/folly/folly
autoreconf -ivf
./configure
make
sudo make install
sudo ldconfig 

echo
echo Building fbthrift
echo

cd $dir/fbthrift/thrift
autoreconf -ivf
./configure
make
sudo make install

echo
echo 'Installing TH++'
echo

cd $dir/thpp/thpp
./build.sh

echo
echo 'Installing FBLuaLib'
echo

cd $dir/fblualib/fblualib
./build.sh
cd $dir/fblualib/fblualib/python
luarocks make rockspec/fbpython-0.1-1.rockspec

echo
echo 'Almost done!'
echo

git clone https://github.com/torch/nn && ( cd nn && git checkout getParamsByDevice && luarocks make rocks/nn-scm-1.rockspec )

git clone https://github.com/facebook/fbtorch.git && ( cd fbtorch && luarocks make rocks/fbtorch-scm-1.rockspec )

git clone https://github.com/facebook/fbnn.git && ( cd fbnn && luarocks make rocks/fbnn-scm-1.rockspec )


echo
echo 'All done!'
echo
