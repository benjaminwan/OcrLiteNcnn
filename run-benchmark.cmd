chcp 65001
:: Set Param
@ECHO OFF
@SETLOCAL
echo "Setting the Number of Threads=%NUMBER_OF_PROCESSORS% Using an OpenMP Environment Variable"
set OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%

:: run Win x64
pushd build
benchmark.exe --models ../models --image ../../test_imgs/long1.jpg ^
                --numThread $NUM_THREADS --numThread $NUM_THREADS --loopCount 10 -G 0
popd
PAUSE
@ENDLOCAL
