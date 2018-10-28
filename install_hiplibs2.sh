#change to external directory
root_dir="external/HIP"
mkdir -p $root_dir
cd $root_dir

rocm_platform_repo="https://github.com/ROCmSoftwarePlatform"
repo_list=( HIP hipBLAS hipDNN hiprand HcSparse )

clone_rep(){
    repo_name=$1
    echo "$rocm_platform_repo/$repo_name"
}

echo $PWD

for repo in "${repo_list[@]}"
do
    echo $repo
done