#!/usr/bin/env bash

sysOS=`uname -s`
NUM_THREADS=1
if [ $sysOS == "Darwin" ];then
    #echo "I'm MacOS"
    NUM_THREADS=$(sysctl -n hw.ncpu)
elif [ $sysOS == "Linux" ];then
    #echo "I'm Linux"
    NUM_THREADS=$(grep ^processor /proc/cpuinfo | wc -l)
else
    echo "Other OS: $sysOS"
fi

echo "Setting the Number of Threads=$NUM_THREADS Using an OpenMP Environment Variable"
set OMP_NUM_THREADS=$NUM_THREADS

##### run test on MacOS or Linux
pushd build
#export LD_LIBRARY_PATH="../onnxruntime-shared/linux:../onnxruntime-shared/macos"
./benchmark --models ../models --image ../../test_imgs/long1.jpg \
--numThread $NUM_THREADS --loopCount 5 -G 0
popd