#!/bin/bash
EXIT=""
# EXIT="exit" # uncomment for termination on failure to install package
HL="\e[41m"  # Error highlight
RS="\e[49m"


echo "CNTK-deps installation error log: $(date +%d-%m-%y_%T) " > install_failed.log
ERR_LOGGER="| tee -a install_failed.log"

 apt-get -y update
 apt-get -y upgrade
 apt-get -y autoremove

## Basic dependency list
ARRAY=(  git autoconf automake libtool curl make
g++ unzip zlib1g-dev libbz2-dev python-dev cmake
libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev # OpenCV prerequisites
)

for package in "${ARRAY[@]}"; do
   apt-get -y install $package -f -q #> /dev/null
done

for package in "${ARRAY[@]}"; do
  if dpkg -s $package > /dev/null; then
    echo "Package $package is installed"
  else
    echo -e "${HL}Package $package is NOT installed${RS}" $ERR_LOGGER
    ${EXIT}
  fi
done

## Install MKL-ml and MKL-dnn
 mkdir /usr/local/mklml -p && \
wget https://github.com/01org/mkl-dnn/releases/download/v0.12/mklml_lnx_2018.0.1.20171227.tgz && \
 tar -xzf mklml_lnx_2018.0.1.20171227.tgz -C /usr/local/mklml && \
wget --no-verbose -O - https://github.com/01org/mkl-dnn/archive/v0.12.tar.gz | tar -xzf - && \
cd mkl-dnn-0.12 && \
 ln -s /usr/local external && \
mkdir -p build && cd build && \
cmake .. && make && \
 make install && \
cd ../.. && rm -rf mkl-dnn-0.12
if [ $? -ne 0 ] ; then
  echo -e "${HL}FAILED in Intel-MKL; check if conflicting packages or same package already installed${RS}" $ERR_LOGGER
  ${EXIT}
fi

## Open-MPI
wget https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.3.tar.gz && \
tar -xzvf ./openmpi-1.10.3.tar.gz && \
cd openmpi-1.10.3 && \
./configure --prefix=/usr/local/mpi && \
make clean && make -j "$(nproc)" all &&  make install
if [ $? -ne 0 ] ; then
  echo -e "${HL}FAILED in open-MPI; check if conflicting packages or same package already installed${RS}" $ERR_LOGGER
  ${EXIT}
fi
## Echo in bashrc
echo "## Lines added by CNTK - installer -----" >> ~/.bashrc
echo "export PATH=/usr/local/mpi/bin:\$PATH " >> ~/.bashrc && \
echo "export LD_LIBRARY_PATH=/usr/local/mpi/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc && \
source ~/.bashrc

## Google protobuf
wget https://github.com/google/protobuf/archive/v3.1.0.tar.gz && \
tar -xzf v3.1.0.tar.gz && \
cd protobuf-3.1.0 && ./autogen.sh && \
./configure CFLAGS=-fPIC CXXFLAGS=-fPIC --disable-shared --prefix=/usr/local/protobuf-3.1.0 && \
make clean && make -j $(nproc) && make install
if [ $? -ne 0 ] ; then
  echo -e "${HL}FAILED in Google protobuf installation; check if conflicting packages or same package already installed${RS}" $ERR_LOGGER
  ${EXIT}
fi

## LibZip
wget http://nih.at/libzip/libzip-1.1.2.tar.gz && \
tar -xzvf ./libzip-1.1.2.tar.gz && \
cd libzip-1.1.2 &&
./configure && make clean && make -j $(nproc) all && make install
if [ $? -ne 0 ] ; then
  echo -e "${HL}FAILED in libzip installation; check if conflicting packages or same package already installed${RS}"  $ERR_LOGGER
  ${EXIT}
fi
##Echo in bashrc
echo "export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH"  >> ~/.bashrc && source ~/.bashrc

## Install boost libraries
wget -O - https://sourceforge.net/projects/boost/files/boost/1.60.0/boost_1_60_0.tar.gz/download | tar -xzf - && \
cd boost_1_60_0 && \
./bootstrap.sh --prefix=/usr/local/boost-1.60.0 && \
 ./b2 -d0 -j $(nproc) install
if [ $? -ne 0 ] ; then
  echo -e "${HL}FAILED in Boost library installation; check if conflicting packages or same package already installed${RS}" $ERR_LOGGER
  ${EXIT}
fi
##Echo in bashrc
echo "export BOOST_INCLUDEDIR=/usr/local/boost-1.60.0" >> ~/.bashrc && source ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/boost-1.60.0\$LD_LIBRARY_PATH" >> ~/.bashrc && source ~/.bashrc

## Install OpenCV
wget https://github.com/Itseez/opencv/archive/3.1.0.zip && unzip 3.1.0.zip && \
cd opencv-3.1.0 && mkdir release && cd release && \
 cmake -D WITH_CUDA=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local/opencv-3.1.0 .. && \
 make clean &&  make all &&  make install
if [ $? -ne 0 ] ; then
  echo -e "${HL}FAILED in OpenCV installation; check if conflicting packages or same package already installed${RS}" $ERR_LOGGER
  ${EXIT}
fi

##Set LD path to find hip-libs
echo "export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:\$LD_LIBRARY_PATH"  >> ~/.bashrc && source ~/.bashrc
echo "##--- end cntk deps---" >> ~/.bashrc

 apt -y update
 apt -y upgrade