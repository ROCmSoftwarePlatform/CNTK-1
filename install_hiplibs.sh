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
repoList=(hcBLAS rocRAND HcSPARSE)

#Installation directories
installDir=("hcblas" " " "hcsparse")

#git command
clone="git clone https://github.com/ROCmSoftwarePlatform"

#build steps
build_dir=build
cmake_it="cmake -DCMAKE_INSTALL_PREFIX=../.."
build_test=("" "-DBUILD_TEST=OFF" "")
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

#platform deducing
platform=$($rootDir/$externalDir/hip/bin/hipconfig --platform)

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
                cd rocBLAS && mkdir $build_dir -p && cd $build_dir
                $cmake_it/rocblas .. && make && make install
                cd $rootDir/$externalDir
	fi
fi

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
        if [ "${repoList[$i]}" != "hipDNN" ]; then
            mkdir $build_dir -p && cd $build_dir
            $cmake_it/${installDir[$i]} ${build_test[$i]} .. && make && make install
        else
            make INSTALL_DIR=../hipDNN
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
    git checkout developer-cub-hip
    git checkout 3effedd23f4e80ccec5d0808d8349f7d570e488e
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
