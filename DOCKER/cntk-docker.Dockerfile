# This dockerfile is meant to be personalized, and serves as a template and demonstration.
# Modify it directly, but it is recommended to copy this dockerfile into a new build context (directory),
# modify to taste and modify docker-compose.yml.template to build and run it.

# It is recommended to control docker containers through 'docker-compose' https://docs.docker.com/compose/
# Docker compose depends on a .yml file to control container sets
# rocm-setup.sh can generate a useful docker-compose .yml file
# `docker-compose run --rm <rocm-terminal>`

# To build the dockerfile run ‘ docker build -f cntk-docker.Dockerfile -t name:tag .’ in the current directory.

# If it is desired to run the container manually through the docker command-line, the following is an example
# 'docker run -it --rm -v [host/directory]:[container/directory]:ro <user-name>/<project-name>'.

FROM ubuntu:16.04

# Initialize the image
# Modify to pre-install dev tools and ROCm packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl && \
  curl -sL http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | apt-key add - && \
  sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list' && \
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  sudo \
  libelf1 \
  libnuma-dev \
  build-essential \
  git \
  wget \
  vim-nox \
  byobu \
  cmake-curses-gui && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Grant members of 'sudo' group passwordless privileges
# Comment out to require sudo
###COPY sudo-nopasswd /etc/sudoers.d/sudo-nopasswd
RUN echo "%sudo   ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/sudo-nopasswd

# This is meant to be used as an interactive developer container
# Create user rocm-user as member of sudo group
# Append /opt/rocm/bin to the system PATH variable
RUN useradd --create-home -G sudo --shell /bin/bash rocm-user
#    sed --in-place=.rocm-backup 's|^\(PATH=.*\)"$|\1:/opt/rocm/bin"|' /etc/environment

WORKDIR /home/rocm-user
ENV PATH "${PATH}:/opt/rocm/bin"

#other ROCM libs
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install rocm-dev \
    rocm-libs \
    hip_hcc

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y upgrade


ENV HIP_PLATFORM "hcc"
RUN git clone https://github.com/ROCmSoftwarePlatform/CNTK-1.git && cd CNTK-1 && \
   ./prerequisites_install.sh && mkdir -p build && cd build && \
    mkdir -p release && cd release && git checkout hipDNN_debug


#Modify HIP
RUN sed -i "s=\<using half\>=//using half=g" /opt/rocm/hip/include/hip/hcc_detail/hip_fp16.h && \
    sed -i "s=\<using half\>=//using half=g" /opt/rocm/hip/include/hip/hcc_detail/hip_fp16_gcc.h

RUN cd /home/rocm-user/CNTK-1/Tools/devInstall/Linux/ && ./install-swig.sh && ~/home/rocm-user/CNTK-1/

#Install anaconda

#1. cd /tmp && \
#2. curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
#3. sha256sum Anaconda3-5.0.1-Linux-x86_64.sh
#4. bash Anaconda3-5.0.1-Linux-x86_64.sh  
#5. Approve terms and conditions.
#6. Provide location
#7. To activate run 'source ~/anaconda3/etc/profile.d/conda.sh'
#8. conda activate cntk-py35
#9. export LD_LIBRARY_PATH=/root/anaconda3/lib:{CNTK_PATH}/build/release/lib"
#10 export PATH=/root/anaconda3/bin

# For a Python 3.5 based version:
#conda env create --file [CNTK clone root]/Scripts/install/linux/conda-linux-cntk-py35-environment.yml
# To configure run '../../configure --asgd=no --with-swig=/usr/local/swig-3.0.10 --with-py35-path=$HOME/anaconda3/envs/cntk-py35 ' in build/release folder
# To build make -j12 && sudo make install && 

# The following are optional enhancements for the command-line experience
# Uncomment the following to install a pre-configured vim environment based on http://vim.spf13.com/
# 1.  Sets up an enhanced command line dev environment within VIM
# 2.  Aliases GDB to enable TUI mode by default
#RUN curl -sL https://j.mp/spf13-vim3 | bash && \
#    echo "alias gdb='gdb --tui'\n" >> ~/.bashrc

# Default to a login shell

CMD ["bash", "-l"]
