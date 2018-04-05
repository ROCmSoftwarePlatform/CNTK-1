#!/bin/bash

#Script directory
rootDir=$(dirname "$(readlink -f "$0")")
cd $rootDir

#External Directory
externalDir=external/HIP
mkdir ${externalDir} -p
cd ${externalDir}
cur_dir=$(pwd)
mkdir lib64 -p

#List of repos to be cloned and installed
repoList=(hipBLAS rocRAND HcSPARSE)

#Installation directories
installDir=(" " " " "hcsparse")


#git command
clone="git clone https://github.com/ROCmSoftwarePlatform"

#build steps
build_dir=build
cmake_it="cmake -DCMAKE_INSTALL_PREFIX=../.."
build_test=("" "-DCMAKE_MODULE_PATH=$rootDir/$externalDir/hip/cmake -DBUILD_TEST=OFF" "" "")
remove="rm -rf"

#function for building - TODO:
#build ()
#{
#	$clone/$1.git
#	cd $1
#	if [ "$1" != "hipDNN" ]; then
#		mkdir $build_dir -p
#		cd $build_dir
#		$cmake_it$2 $3 ..
#		make
#		make install
#		cd ../../
#	fi
#}

#function to check if local repo exists already
check()
{
	Repo=$(echo $(pwd)/$1|cut -d' ' -f1)
	if [ -d $Repo ]; then
		return 1
	return 0
	fi
}

#HIP installation
echo -e "\n--------------------- HIP LIBRARY INSTALLATION ---------------------\n"
check HIP
hipRepo=$?
if [ "$hipRepo" == "1" ]; then
	echo -e "\t\t----- HIP already exists -----\n"
else
	echo -e "\n--------------------- CLONING HIP ---------------------\n"
	git clone https://github.com/ROCm-Developer-Tools/HIP.git
	cd HIP && mkdir $build_dir -p && cd $build_dir
    $cmake_it/hip .. && make && make install
    cd $rootDir/$externalDir
fi

export HIP_PATH="$rootDir/$externalDir/hip"
HIP_PATH="$rootDir/$externalDir/hip"

#platform deducing
platform=$($rootDir/$externalDir/hip/bin/hipconfig --platform)

dependencies=("make" "cmake-curses-gui" "pkg-config")
if [ "$platform" == "hcc" ]; then
	dependencies+=("python2.7" "python-yaml" "libssl-dev" "libboost-dev" "libboost-system-dev" "libboost-filesystem-dev")
fi

for package in "${dependencies[@]}"; do
    if [[ $(dpkg-query --show --showformat='${db:Status-Abbrev}\n' ${package} 2> /dev/null | grep -q "ii"; echo $?) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      sudo apt install -y --no-install-recommends ${package}
    fi
done

#extra repos for hcc
if [ "$platform" == "hcc" ]; then
	export HIP_SUPPORT=on
	export CXX=/opt/rocm/bin/hcc
	repoList+=(MIOpenGEMM MIOpen)
	installDir+=(miopengemm miopen)
	check rocBLAS
	rocblasRepo=$?
	if [ "$rocblasRepo" == "1" ]; then
		echo -e "\t\t----- rocBLAS already exists -----\n"
	else
		echo -e "\n--------------------- CLONING rocBLAS ---------------------\n"
		$clone/rocBLAS.git
		echo -e "\n--------------------- INSTALLING rocBLAS ---------------------\n"
                cd rocBLAS && mkdir $build_dir -p && cd $build_dir
                $cmake_it/ .. && make && make install
                cd $rootDir/$externalDir
	fi
	export rocblas_DIR=$rootDir/$externalDir/rocblas/lib/cmake/rocblas
#dependencies for miopengemm

	#opencl
	#sudo apt update
	sudo apt install ocl-icd-opencl-dev

	#rocm make package
	git clone https://github.com/RadeonOpenCompute/rocm-cmake.git
	cd rocm-cmake
	mkdir $build_dir -p && cd $build_dir
	$cmake_it/ ..
	cmake --build . --target install
	export ROCM_DIR=$(pwd)/../share/rocm/cmake/
	cd $rootDir/$externalDir

#dependencies for miopen

	#clang-ocl
	git clone https://github.com/RadeonOpenCompute/clang-ocl.git
	cd clang-ocl
	mkdir $build_dir -p && cd $build_dir
	$cmake_it/ ..
	cmake --build . --target install
	cd $rootDir/$externalDir

	#ssl
	#sudo apt-get install libssl-dev
fi

repoList+=(hipDNN)
installDir+=("hipdnn")

#cloning and install
for i in "${!repoList[@]}"
do
    #check if local repo exists
    check ${repoList[$i]}
    localRepo=$?
    if [ "$localRepo" == "1" ]; then
        echo -e "\t\t----- ${repoList[$i]} already exists -----\n"
    else
        echo -e "\n--------------------- CLONING ${repoList[$i]} ---------------------\n"
        $clone/${repoList[$i]}.git
        cd ${repoList[$i]}
        if [ "${repoList[$i]}" == "rocRAND" ]; then
            git checkout rocm_1_7_1
        fi
        echo -e "\n--------------------- INSTALLING ${repoList[$i]} ---------------------\n"
        if [ "${repoList[$i]}" != "hipDNN" ] && [ "${repoList[$i]}" != "MIOpen" ]; then
            mkdir $build_dir -p && cd $build_dir
            $cmake_it/${installDir[$i]} ${build_test[$i]} .. && make && make install
	    elif [ "${repoList[$i]}" == "MIOpen" ]; then
	        export miopengemm_DIR=$rootDir/$externalDir/miopengemm/lib/cmake/miopengemm
            mkdir $build_dir -p && cd $build_dir
            CXX=/opt/rocm/hcc/bin/hcc cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm/hcc;${HIP_PATH}" -DCMAKE_CXX_FLAGS="-isystem /usr/include/x86_64-linux-gnu/" -DCMAKE_INSTALL_PREFIX=../../ .. && make && make install
        else
            make INSTALL_DIR=../hipdnn HIP_PATH=$rootDir/$externalDir/hip MIOPEN_PATH=$rootDir/$externalDir/miopen/
        fi
        cd $rootDir/$externalDir
    fi
done

#cloning cub-hip
cubRepo=$(echo $(pwd)/cub-hip |cut -d' ' -f1)
if [ -d $cubRepo ]; then
    echo -e "\t\t----- CUB-HIP already exists -----\n"
else
    git clone https://github.com/ROCmSoftwarePlatform/cub-hip.git
    cd cub-hip
    git checkout hip_port_1.7.4
    #git checkout 3effedd23f4e80ccec5d0808d8349f7d570e488e
    cd $rootDir/$externalDir
fi

#copying shared objects
DIRS=`ls -l --time-style="long-iso" . | egrep '^d' | awk '{print $8}'`
for DIR in $DIRS
do
    cd $DIR
    SUB=`ls -l --time-style="long-iso" $MYDIR | egrep '^d' | awk '{print $8}'`
    for sub in $SUB
    do
        if [ "$sub" == "lib" ]; then
            cp -a lib/. ../lib64/
        fi
    done
    cd ..
done

echo -e "\n--------------------- HIP LIB INSTALLATION COMPLETE ---------------------\n"
