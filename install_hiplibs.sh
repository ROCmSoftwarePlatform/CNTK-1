#!/bin/bash

rootDir=$(dirname "$(readlink -f "$0")") #Script directory
cd $rootDir

externalDir=external/HIP #External Directory
rocmDir=/opt/rocm
mkdir ${externalDir} -p
cd ${externalDir}
cur_dir=$(pwd)

RED=$(tput setaf 1) GREEN=$(tput setaf 2) NC=$(tput sgr0) #output colours

clone="git clone https://github.com/ROCmSoftwarePlatform"

#build steps

build_dir=build
build_test=("" "-DCMAKE_MODULE_PATH=$rootDir/$externalDir/hip/cmake -DBUILD_TEST=OFF" "" "")
install=0
remove="rm -rf"
spacef="\n\t\t-----"
spaceb="-----\n\t\t"

#function to check if local repo exists already

# CMake build
cmake_build(){
    install_path=$1
    mkdir -p build
    cd build
    $remove
    cmake  -DCMAKE_INSTALL_PREFIX=$install_path ..
    make all
    make install
}

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
echo -e "$GREEN Please specify the local source code path. Press [ENTER] to skip :\n"
read -p "$NC HIP SOURCE CODE :" HIP_SCP

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
    echo -e "$NC $spacef HIP already installed , Checking for the necessary files $spaceb"
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

# Installing HIP libraries(ROCm) from apt
if  [ "$install" = "0" ]; then
    sudo apt-get update -y -qq && sudo apt dist-upgrade -y -qq
    wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add - &&
    echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
    sudo apt-get update -y &>/dev/null
    echo -e "$NC $spacef HIP[ROCm-libs] from apt $spaceb"
    sudo apt-get install rocm-libs -y
    if [ $? -eq 0 ];then
        install=1
    else
        echo -e "$RED $spacef apt-get of rocm-libs Failed! HIP will be freshly installed from source $spaceb "
    fi
    # Echo in bashrc for Linking
    echo -e "\n## ---Added HIP-libs ROCm installation---" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH"
fi

# Installing HIP libraries(ROCm) from Source (incase apt fails)
if [ "$install" == "0" ]; then
    if [[ -z "$HIP_SCP" ]]; then
        echo -e "$NC $spacef CLONING HIP $spaceb"
        rm -rf HIP
        git clone https://github.com/ROCm-Developer-Tools/HIP.git
        cd HIP
        if [ "$FREEZE_COMMIT" == "1" ]; then
            hipcommit="0e44ca7"
            git reset --hard $hipcommit
            echo "commit id $hipcommit"
        fi
    else
        echo -e "$NC $spacef Installing the available Source Code $spaceb"
        cd $HIP_SCP
    fi
    mkdir $build_dir -p && cd $build_dir
    cmake  .. && make && sudo make install
    cd $rootDir/$externalDir
fi

echo -e "$NC $spacef HIP installation complete $spaceb"
export HIP_PATH=$rocmDir/hip

if [ "$install" == "1" ];then
   sudo sed -i "s=\<using half\>=//using half=g" /opt/rocm/hip/include/hip/hcc_detail/hip_fp16.h
   sudo sed -i "s=\<using half\>=//using half=g" /opt/rocm/hip/include/hip/hcc_detail/hip_fp16_gcc.h
fi

#Platform deducing

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

#repos needed for AMD and Nvidia

repoList+=(rocRAND HcSPARSE hipBLAS hipDNN rocPRIM)
commitList+=(1890bb31675a6cbaa7766e947c8e35c4d1010ad6 907a505c27bac57a6d1f372154b744dd14ced943 193e50ed975a02d5efad566239107e3d7c768712 898b9d9ae7ed58a46beecc0fb0b785716da204f9 caef132d64b29a7d857eb68af5323fc302d26766)
installDir+=(hiprand hcsparse hipblas hipdnn hipcub)
libList+=(hiprand hipsparse hipblas hipdnn hipcub)
scpLIST+=(rocRAND_SCP HcSPARSE_SCP hipBLAS_SCP hipDNN_SCP rocPRIM_SCP)
headerList+=(hiprand hipsparse hipblas hipdnn hipcub)

if [ "$platform" == "hcc" ]; then
    repoList+=(MIOpenGEMM MIOpen rocBLAS )
    commitList+=(9547fb9e8499a5a9f16da83b1e6b749de82dd9fb 08114baa029a519ea12b52c5274c0bd8f4ad0d26 cb738e830f4239a14b6b73f9a58fdf943d914030)
    installDir+=(miopengemm miopen rocblas)
    libList+=(miopengemm MIOpen rocblas)
    headerList+=(miogemm miopen rocblas)
    scpLIST+=(MIOpenGEMM_SCP MIOpen_SCP rocBLAS_SCP)
elif [ "$platform" == "nvcc" ]; then
     repoList+=(cub)
     commitList+=(c3cceac115c072fb63df1836ff46d8c60d9eb304)
     installDir+=(hipcub)
     libList+=(hipcub) #dummy
     headerList+=(cub)
     scpLIST+=(CUB_SCP)
fi

echo -e "\n\n"

#read source code paths

declare -A pathlist
for i in "${!repoList[@]}"
do
    loop=0
    while [[ "$loop" == "0" ]]
    do
        read -p "$NC ${repoList[$i]} Source Code Path :" pathlist[${scpLIST[$i]}]
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

# MIopen installation on AMD

if [ "$platform" == "hcc" ]; then
	export HIP_SUPPORT=on
	export CXX=/opt/rocm/bin/hcc

    # dependencies for miopengemm
    #opencl
    sudo apt install ocl-icd-opencl-dev -y
    #rocm make package
    check rocm-cmake
    rocmcmakeRepo=$?
    FILE=`find $rocmDir -iname ROCMConfig.cmake -print -quit`
    if ! [ -n "$FILE" ]; then
        git clone https://github.com/RadeonOpenCompute/rocm-cmake.git
        cd rocm-cmake
        mkdir $build_dir -p && cd $build_dir
        cmake/ ..
        sudo cmake --build . --target install
    fi
    cd $rootDir/$externalDir

    # dependencies for miopen
    #clang-ocl
    check clang-ocl
    clangoclRepo=$?
    FILE=`find $rocmDir/bin -iname clang-ocl -print -quit`
    if ! [ -n "$FILE" ]; then
        git clone https://github.com/RadeonOpenCompute/clang-ocl.git
        cd clang-ocl
        mkdir $build_dir -p && cd $build_dir
        cmake/ ..
        sudo cmake --build . --target install
    fi
    cd $rootDir/$externalDir
    #ssl
    #sudo apt-get install libssl-dev -y

    # Install MIOpen from apt
    echo -e "$NC $spacef Installing MIopen from apt $spaceb"
    sudo apt-get install miopen-hip -y -q
    miopenApt=$?
    if [ $miopenApt -ne 0 ]; then
        echo -e "$RED $spacef apt-get of MIOpen Failed! MIopen will be freshly installed from source $spaceb "
    fi
fi

#cloning and install
for i in "${!repoList[@]}"
do
    #check if local repo exists
    echo -e "$NC $spacef Installing ${repoList[$i]} & Checking ${installDir[$i]} $spaceb"
    install=0
    check ${installDir[$i]}
    localRepo=$?
    if [ "$localRepo" == "1" ]; then
        echo -e "$NC $spacef ${repoList[$i]} already installed $spaceb"
        #cd $rocmDir/${libList[$i]}/lib
        if [ "${repoList[$i]}" == "MIOpenGEMM" ]; then
            HEADER=`find $rocmDir/${installDir[$i]} -iname miogemm.hpp -print -quit`
        else
            HEADER=`find $rocmDir/${installDir[$i]} \( -iname ${headerList[$i]}.h -o -iname ${headerList[$i]}.hpp -o -iname ${headerList[$i]}.cuh \) -print -quit`
        fi
        if [ -n "$HEADER" ]; then
            echo -e "$GREEN Found ${repoList[$i]} header \t: $HEADER"
            if [ "${repoList[$i]}" != rocPRIM ] && [ "${repoList[$i]}" != cub ]; then
                FILE=`find $rocmDir/${installDir[$i]} -iname lib${libList[$i]}.so -print -quit`
            fi
            if [ -n "$FILE" ]; then
                echo -e "$GREEN Found ${repoList[$i]} libs  \t: $FILE"
                install=1
                if [ "${repoList[$i]}" == "MIOpen" ]; then
                    miopen_hip_status=`grep "MIOPEN_BACKEND_HIP" $rocmDir/${installDir[$i]}/include/miopen/config.h `
                    if [ "$(echo $miopen_hip_status | cut -d ' ' -f 3-)" == "0" ]; then
                            echo "MIopen with opencl backend has been installed. It will be overwritten by MIopen with hip background"
                            install=0
                    fi
                fi
            else
                echo -e "$RED Broken library - shared object Not found.Library will be installed fresh\n"
            fi
        else
            echo -e "$RED Broken library - header files not found. Library will be installed fresh\n"
        fi
    fi
    if [ "$install" == "0" ]; then
        if [[ -z "${pathlist["${scpLIST[$i]}"]}" ]]; then
            echo -e "$NC $spacef CLONING ${repoList[$i]} $spaceb"
            rm -rf ${repoList[$i]}
            if [ ${repoList[$i]} == "cub" ]; then
                git clone https://github.com/NVlabs/cub.git
            else
                $clone/${repoList[$i]}.git
            fi
            cd ${repoList[$i]}
            if [ "$FREEZE_COMMIT" == "1" ]; then
                git reset --hard ${commitList[$i]}
                echo ${commitList[$i]}
            fi
        else
            echo -e "$NC $spacef Installing the available Source Code $spaceb"
            cd ${pathlist["${scpLIST[$i]}"]}
        fi
        echo -e "$NC $spacef INSTALLING ${repoList[$i]} $spaceb"
        if [ "${repoList[$i]}" != "MIOpen" ] && [ "${repoList[$i]}" != "cub" ]; then
            mkdir $build_dir -p && cd $build_dir
            cmake -DBUILD_TEST=OFF .. && make -j $(nproc) && sudo make install
        elif [ "${repoList[$i]}" == "MIOpen" ]; then
            export LD_LIBRARY_PATH=/usr/local/boost-1.60.0/lib:$LD_LIBRARY_PATH
            export PATH=/usr/local/boost-1.60.0/:$PATH
	        #export miopengemm_DIR=$rootDir/$externalDir/miopengemm/lib/cmake/miopengemm
            wget -O half.zip https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip/download && unzip half.zip -d half && cd half/include
            HALF_DIRECTORY=$(pwd)
            cd $rootDir/$externalDir/${repoList[$i]}
            mkdir $build_dir -p && cd $build_dir
            CXX=/opt/rocm/hcc/bin/hcc cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm/hcc;${HIP_PATH}" -DHALF_INCLUDE_DIR=$HALF_DIRECTORY -DCMAKE_CXX_FLAGS="-isystem /usr/include/x86_64-linux-gnu/" .. && make -j $(nproc) && sudo make install
        elif [ "${repoList[$i]}" == "cub" ]; then
            sudo mkdir -p $rocmDir/${installDir[$i]}/
            sudo cp -r $rootDir/${externalDir}/${repoList[$i]}/cub $rocmDir/${installDir[$i]}/
        else
            make -j $(nproc)
        fi
        cd $rootDir/$externalDir
    fi
done

# #copying shared objects

# repoList+=(hipRAND hipCUB)
# installDir+=(rocrand)
# libList+=(rocrand)
# headerList+=(rocrand)

# if [ "$platform" == "hcc" ];then
#     installDir+=(rocprim)
#     headerList+=(rocprim)
# elif [ "$platform" == "nvcc" ];then
#     installDir+=(cub)
#     headerList+=(cub.cuh)
# fi

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

echo -e "$NC $spacef Validating the installation process $spaceb"
for i in "${!repoList[@]}"
do
    perfect=0
    check ${installDir[$i]}
    localRepo=$?
    if [ "$localRepo" == "1" ]; then
        HEADER=`find $rocmDir/${installDir[$i]} \( -name ${headerList[$i]}.h -o -name ${headerList[$i]}.hpp -o -name ${headerList[$i]}.cuh \) -print -quit`
        if [ -n "$HEADER" ]; then
            #echo -e "Found ${repoList[$i]} header \t: $HEADER"
            if [ "${repoList[$i]}" != "cub" ] && [ "${repoList[$i]}" != "rocPRIM" ]; then
                FILE=`find $rocmDir/${installDir[$i]} -name lib${libList[$i]}.so -print -quit`
            else
                FILE=0
            fi
            if [ -n "$FILE" ]; then
                echo -e "\n $GREEN ${repoList[$i]} installed properly"
                perfect=1
            else
                echo -e "\n $RED ${repoList[$i]} Broken - shared object Not found."
            fi
        else
            echo -e "\n $RED ${repoList[$i]} Broken - header files not found."
        fi
    fi

    if [ "$perfect" == "0" ]; then
        echo -e "\n $RED ${repoList[$i]} is not installed properly. Kindly check the error log"
    fi
done

echo -e "$NC $spacef Validation done $spaceb"

echo -e "$GREEN $spacef HIP LIB INSTALLATION COMPLETE $spaceb"

while [[ 1 ]]
do
    read -p "$NC Do you wish to remove the cloned source repos ? [ Yes / No ] " choice
    case $choice in
        [Yy][eE][sS] | [y] | [Y] ) rm -rf $rootDir/$externalDir ; echo -e "$GREEN $spacef Repos removed $spaceb $NC" ; break;;
        [Nn][Oo] | [n] | [N] ) echo -e "$GREEN $spacef Repos not removed $spaceb $NC" ; break;;
        * ) echo -e "$RED $spacef Invalid Input - Enter either Yes / No $spaceb $NC" ;;
    esac
done
