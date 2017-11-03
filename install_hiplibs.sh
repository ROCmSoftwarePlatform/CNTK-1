#Script directory
rootDir=$(dirname "$(readlink -f "$0")")
cd $rootDir

#External Directory
externalDir=External/HIP
mkdir ${externalDir} -p
cd ${externalDir}
cur_dir=$(pwd)
mkdir lib64 -p

#List of repos to be cloned and installed
repoList=(HIP hipBLAS rocRAND HcSPARSE)

#Installation directories
installDir=("hip" "hipblas" "" "hcsparse")

#git command
clone="git clone https://github.com/ROCmSoftwarePlatform/"

#build steps
build_dir=build
cmake_it="cmake -DCMAKE_INSTALL_PREFIX=../../"
build_test=("" "" "-DBUILD_TEST=OFF" "")
remove="rm -rf"

echo -e "\n--------------------- HIP LIBRARY INSTALLATION ---------------------\n"
#cloning and install
for i in "${!repoList[@]}"
do
    #check if local repo exists
    localRepo=$(echo $(pwd)/${repoList[$i]}|cut -d' ' -f1)
    if [ -d $localRepo ]; then
        echo -e "\t\t----- ${repoList[$i]} already exists -----\n"
    else
        echo -e "\n--------------------- CLONING ${repoList[$i]} ---------------------\n"
        if [ "${repoList[$i]}" == "HIP" ]; then
            git clone https://github.com/ROCm-Developer-Tools/HIP.git
        else
            $clone${repoList[$i]}.git
        fi
        cd ${repoList[$i]}
        if [ "${repoList[$i]}" != "hipDNN" ]; then
            mkdir $build_dir -p
            cd $build_dir
            $cmake_it${installDir[$i]} ${build_test[$i]} ..
            make
            make install
        else
            make INSTALL_DIR=../hipDNN
        fi
        cd ../../
    fi
done

#cloning cub-hip
cubRepo=$(echo $(pwd)/cub-hip |cut -d' ' -f1)
if [ -d $cubRepo ]; then
    echo -e "\t\t----- CUB-HIP already exists -----\n"
else
    git clone https://github.com/ROCmSoftwarePlatform/cub-hip.git
    git checkout developer-cub-hip
    gti checkout 3effedd23f4e80ccec5d0808d8349f7d570e488e
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
