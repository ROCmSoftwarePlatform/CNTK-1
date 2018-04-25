#!/bin/bash

rootDir=$(dirname "$(readlink -f "$0")") #Script directory
cd $rootDir

externalDir=external/HIP #External Directory
rocmDir=/opt/rocm
mkdir ${externalDir} -p
cd ${externalDir}
cur_dir=$(pwd)

RED=$(tput setaf 1) GREEN=$(tput setaf 2) YELLOW=$(tput setaf 3) NC=$(tput sgr0) #output colours

clone="git clone https://github.com/ROCmSoftwarePlatform"

#build steps

build_dir=build
cmake_it="cmake "
build_test=("" "-DCMAKE_MODULE_PATH=$rootDir/$externalDir/hip/cmake -DBUILD_TEST=OFF" "" "")
install=0
remove="rm -rf"
spacef="\n\t\t-----"
spaceb="-----\n\t\t"

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
	Repo=$(echo $rocmDir/$1|cut -d' ' -f1)
	if [ -d $Repo ]; then
		return 1
	return 0
	fi
}

#HIP installation

echo -e "$GREEN $spacef HIP LIBRARY INSTALLATION $spaceb"
echo -e "$GREEN $spacef Please specify path for the following , skip if Invalid :$spaceb" 
read -p "$YELLOW HIP SOURCE CODE :" HIP_SCP

#if [[ -z "$HIP_SCP" ]]; then
#    echo "No value entered"
#else
if [[ "$HIP_SCP" ]]; then
    #if [ !"$(ls -A $HIP_SCP)" ]; then
    if [ $(find $HIP_SCP -maxdepth 0 -type d -empty 2>/dev/null) ]; then
        echo -e "$RED $spacef Specified directory is Empty. HIP header and shared object will be checked under /opt/rocm , if not found HIP will be pulled and installed !$spaceb"
    fi
fi

check hip
hipRepo=$?
install=0

if [ "$hipRepo" == "1" ]; then
    echo -e "$YELLOW $spacef HIP already installed , Checking for the necessary files $spaceb"
    HIPCONFIG=`find $rocmDir/hip/bin -name hipconfig -printf '%h\n' -quit`
    if [ -n "$HIPCONFIG" ]; then
        platform=$($rocmDir/hip/bin/hipconfig --platform)
        HEADER=`find $rocmDir/hip -name hip_runtime.h -printf '%h\n' -quit`
        if [ -n "$HEADER" ]; then
            echo -e "$GREEN Found HIP Header \t: $HEADER"
            if [ "$platform" == "hcc" ]; then
                FILE=`find $rocmDir/hip -name libhip_hcc.so -printf '%h\n' -quit`
                if [ -n "$FILE" ]; then
                    echo -e "$GREEN Found HIP libs   \t: $FILE"
                    install=1
                fi
            else
                install=1
            fi
        else
            echo -e "$RED $spacef Necessary files not found ! HIP will be freshly installed $spaceb"
        fi
    else
        echo -e "$RED $spacef hipconfig not found ! HIP will be freshly installed $spaceb"
    fi
fi

if [ "$install" == "0" ]; then
    if [[ -z "$HIP_SCP" ]]; then
        echo -e "$NC $spacef CLONING HIP $spaceb"
        git clone https://github.com/ROCm-Developer-Tools/HIP.git
        cd HIP
    else
        echo -e "$NC $spacef Installing the available Source Code $spaceb"
        cd $HIP_SCP
    fi
    mkdir $build_dir -p && cd $build_dir
    $cmake_it .. && make && sudo make install
    cd $rootDir/$externalDir
fi

echo -e "$YELLOW $spacef HIP installation complete $spaceb"
export HIP_PATH=$rocmDir/hip

#platform deducing

platform=$($HIP_PATH/bin/hipconfig --platform)

if [ "$platform" == "nvcc" ]; then
	sudo mkdir -p /opt/rocm/hip/lib/cmake/hip
	sudo cp $rootDir/hip-config.cmake /opt/rocm/hip/lib/cmake/hip
fi

#dependencies

dependencies=("make" "cmake-curses-gui" "pkg-config")
if [ "$platform" == "hcc" ]; then
	dependencies+=("python2.7" "python-yaml" "libssl-dev")
fi

for package in "${dependencies[@]}"; do
    if [[ $(dpkg-query --show --showformat='${db:Status-Abbrev}\n' ${package} 2> /dev/null | grep -q "ii"; echo $?) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      sudo apt install -y --no-install-recommends ${package}
    fi
done

#repos needed for AMD

if [ "$platform" == "hcc" ]; then
    repoList+=(rocBLAS MIOpenGEMM MIOpen)
    installDir+=(rocblas miopengemm miopen)
    libList+=(rocblas miopengemm MIOpen)
    headerList+=(rocblas miogemm miopen)
    scpLIST+=(rocBLAS_SCP MIOpenGEMM_SCP MIOpen_SCP)
fi

repoList+=(rocRAND HcSPARSE hipBLAS hipDNN)
installDir+=(hiprand hcsparse hipblas hipDNN)
libList+=(hiprand hipsparse hipblas hipDNN)
scpLIST+=(rocRAND_SCP HcSPARSE_SCP hipBLAS_SCP hipDNN_SCP)
headerList+=(hiprand hipsparse hipblas hipDNN)

echo -e "\n\n"

#read source code paths

declare -A pathlist
for i in "${!repoList[@]}"
do
    loop=0
    while [[ "$loop" == "0" ]]
    do
        read -p "$YELLOW ${repoList[$i]} Source Code Path :" pathlist[${scpLIST[$i]}]
        if [[ "${pathlist["${scpLIST[$i]}"]}" ]] && ! [[ -e "${pathlist["${scpLIST[$i]}"]}" ]]; then
            echo -e "$RED \n Please enter a valid directory\n"
        else
            if [[ "${pathlist["${scpLIST[$i]}"]}" ]]; then
                if [ $(find "${pathlist["${scpLIST[$i]}"]}" -maxdepth 0 -type d -empty 2>/dev/null) ]; then
                    echo -e "$RED $spacef Specified directory is Empty. HIP header and shared object will be checked under /opt/rocm , if not found HIP will be pulled and installed !$spaceb"
                fi
            fi
            loop=1
        fi
    done
done

echo -e "\n\n"

if [ "$platform" == "hcc" ]; then
	export HIP_SUPPORT=on
	export CXX=/opt/rocm/bin/hcc
#dependencies for miopengemm

    #opencl
    #sudo apt update
    sudo apt install ocl-icd-opencl-dev

    #rocm make package
    check rocm-cmake
    rocmcmakeRepo=$?
    FILE=`find $rocmDir -iname ROCMConfig.cmake -print -quit`
    if ! [ -n "$FILE" ]; then
        git clone https://github.com/RadeonOpenCompute/rocm-cmake.git
        cd rocm-cmake
        mkdir $build_dir -p && cd $build_dir
        $cmake_it/ ..
        sudo cmake --build . --target install
    fi
    cd $rootDir/$externalDir

#dependencies for miopen

    #clang-ocl
    check clang-ocl
    clangoclRepo=$?
    FILE=`find $rocmDir/bin -iname clang-ocl -print -quit`
    if ! [ -n "$FILE" ]; then
        git clone https://github.com/RadeonOpenCompute/clang-ocl.git
        cd clang-ocl
        mkdir $build_dir -p && cd $build_dir
        $cmake_it/ ..
        sudo cmake --build . --target install
    fi
    cd $rootDir/$externalDir

    #ssl
    #sudo apt-get install libssl-dev
fi

#cloning and install
for i in "${!repoList[@]}"
do
    #check if local repo exists
    echo -e "$YELLOW $spacef Installing ${repoList[$i]} & Checking ${installDir[$i]} $spaceb"
    install=0
    check ${installDir[$i]}
    localRepo=$?
    if [ "$localRepo" == "1" ]; then
        echo -e "$YELLOW $spacef ${repoList[$i]} already installed $spaceb"
        #cd $rocmDir/${libList[$i]}/lib
        FILE=`find $rocmDir/${installDir[$i]} -iname lib${libList[$i]}.so -print -quit`
        if [ -n "$FILE" ]; then
               echo -e "$GREEN Found ${repoList[$i]} libs  \t: $FILE"
               if [ "${repoList[$i]}" == "MIOpenGEMM" ]; then
                   HEADER=`find $rocmDir/${installDir[$i]} -iname miogemm.hpp -print -quit`
               else
                   HEADER=`find $rocmDir/${installDir[$i]} \( -iname ${headerList[$i]}.h -o -iname ${headerList[$i]}.hpp \) -print -quit`
               fi
               if [ -n "$HEADER" ]; then
                   echo -e "$GREEN Found ${repoList[$i]} header \t: $HEADER"
                   install=1
               else
                   echo -e "$RED Broken library - header files not found. Library will be installed fresh\n"
               fi
        else
               echo -e "$RED Broken library - shared object Not found.Library will be installed fresh\n"
        fi
    fi
    if [ "$install" == "0" ]; then
        if [[ -z "${pathlist["${scpLIST[$i]}"]}" ]]; then
            echo -e "$NC $spacef CLONING ${repoList[$i]} $spaceb"
            $clone/${repoList[$i]}.git
            cd ${repoList[$i]}
        else
            echo -e "$YELLOW $spacef Installing the available Source Code $spaceb"
            cd ${pathlist["${scpLIST[$i]}"]}
        fi
        echo -e "$NC $spacef INSTALLING ${repoList[$i]} $spaceb"
        if [ "${repoList[$i]}" != "hipDNN" ] && [ "${repoList[$i]}" != "MIOpen" ]; then
            mkdir $build_dir -p && cd $build_dir
            $cmake_it .. && make -j $(nproc) && sudo make install
        elif [ "${repoList[$i]}" == "MIOpen" ]; then
	        #export miopengemm_DIR=$rootDir/$externalDir/miopengemm/lib/cmake/miopengemm
            wget -O half.zip https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip/download && unzip half.zip -d half && cd half/include 
            HALF_DIRECTORY=$(pwd)
            cd $rootDir/$externalDir/${repoList[$i]}
            mkdir $build_dir -p && cd $build_dir
            CXX=/opt/rocm/hcc/bin/hcc cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm/hcc;${HIP_PATH}" -DHALF_INCLUDE_DIR=$HALF_DIRECTORY -DCMAKE_CXX_FLAGS="-isystem /usr/include/x86_64-linux-gnu/" .. && make -j $(nproc) && sudo make install
        else
            make -j $(nproc)
        fi
        cd $rootDir/$externalDir
    fi
done

#cub-hip

check cub-hip
cubRepo=$?
install=0
if [ "$cubRepo" == "1" ]; then
    echo -e "$YELLOW $spacef CUB-HIP already exists $spaceb"
    FILE=`find $rocmDir/cub-hip -iname cub.cuh -print -quit`
    if [ -n "$FILE" ]; then
        echo -e "$GREEN Found cub header : $FILE \n"
        install=1
    else
        echo -e "$RED Header not found. CUB will be pulled and installed fresh\n"
    fi
fi

if [ "$install" == "0" ]; then
    echo -e "$NC $spacef CLONING CUB-HIP $spaceb"
    git clone https://github.com/ROCmSoftwarePlatform/cub-hip.git
    cd cub-hip
    git checkout hip_port_1.7.4
    #git checkout 3effedd23f4e80ccec5d0808d8349f7d570e488e
    sudo cp -r ../cub-hip /opt/rocm/
    cd $rootDir/$externalDir
fi

#copying shared objects

repoList+=(hipRAND)
installDir+=(hiprand hip)
libList+=(hiprand)
headerList+=(hiprand)

for DIR in "${!installDir[@]}"
do
    cd /opt/rocm/${installDir[$DIR]}
    SUB=`ls -l --time-style="long-iso" $MYDIR | egrep '^d' | awk '{print $8}'`
    for sub in $SUB
    do
        if [ "$sub" == "lib" ]; then
            sudo cp -a lib/. /opt/rocm/lib64/
        fi
    done
done

cd $rootDir

#validating if all libs are installed proper

echo -e "$YELLOW $spacef Validating the installation process $spaceb"
for i in "${!repoList[@]}"
do
    perfect=0
    check ${installDir[$i]}
    localRepo=$?
    if [ "$localRepo" == "1" ]; then
        FILE=`find $rocmDir/${installDir[$i]} -name lib${libList[$i]}.so -print -quit`
        if [ -n "$FILE" ]; then
               HEADER=`find $rocmDir/${installDir[$i]} \( -name ${headerList[$i]}.h -o -name ${headerList[$i]}.hpp \) -print -quit`
               if [ -n "$HEADER" ]; then
                   #echo "Found ${repoList[$i]} header \t: $HEADER"
                   echo -e "\n $GREEN ${repoList[$i]} installed properly"
                   perfect=1
               else
                   echo -e "\n $RED ${repoList[$i]} Broken - header files not found."
               fi
        else
               echo -e "\n $RED ${repoList[$i]}Broken - shared object Not found."
        fi
    fi

    if [ "$perfect" == "0" ]; then
        echo -e "\n $RED ${repoList[$i]} is not installed properly. Kindly check the error log"
    fi
done

echo -e "$YELLOW $spacef Validation done $spaceb"

echo -e "$GREEN $spacef HIP LIB INSTALLATION COMPLETE $spaceb"

echo -e "\n\n $NC Do you wish to remove the cloned source repos ?"
select choice in "Yes" "No"; do
    case $choice in
        Yes ) rm -rf $rootDir/$externalDir ; break;;
        No ) exit ;;
    esac
done
