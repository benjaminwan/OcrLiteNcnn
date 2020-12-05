#include "AngleNet.h"
#include "OcrUtils.h"
#include <numeric>

AngleNet::~AngleNet() {
    net.clear();
}

void AngleNet::setNumOfThreads(int numOfThread) {
    numThread = numOfThread;
}

bool AngleNet::initModel(std::string &pathStr) {
    int ret_param = net.load_param((pathStr + "/angle_op.param").c_str());
    int ret_bin = net.load_model((pathStr + "/angle_op.bin").c_str());
    if (ret_param != 0 || ret_bin != 0) {
        printf("AngleNet load param(%d), model(%d)\n", ret_param, ret_bin);
        return false;
    } else {
        return true;
    }
}

Angle scoreToAngle(const float *srcData, int w) {
    int angleIndex = 0;
    float maxValue = -1000.0f;
    for (int i = 0; i < w; i++) {
        if (i == 0)maxValue = srcData[i];
        else if (srcData[i] > maxValue) {
            angleIndex = i;
            maxValue = srcData[i];
        }
    }
    return {angleIndex, maxValue};
}

Angle AngleNet::getAngle(cv::Mat &src) {
    ncnn::Mat input = ncnn::Mat::from_pixels(
            src.data, ncnn::Mat::PIXEL_RGB,
            src.cols, src.rows);
    input.substract_mean_normalize(meanValues, normValues);
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_num_threads(numThread);
    extractor.input("input", input);
    ncnn::Mat out;
    extractor.extract("out", out);
    return scoreToAngle((float *) out.data, out.w);
}

std::vector<Angle> AngleNet::getAngles(std::vector<cv::Mat> &partImgs, const char *path,
                                       const char *imgName, bool doAngle, bool mostAngle) {
    int size = partImgs.size();
    std::vector<Angle> angles(size);
    if (doAngle) {
#ifdef __OPENMP__
#pragma omp parallel for num_threads(numThread)
#endif
        for (int i = 0; i < size; ++i) {
            double startAngle = getCurrentTime();
            auto angleImg = adjustTargetImg(partImgs[i], dstWidth, dstHeight);
            Angle angle = getAngle(angleImg);
            double endAngle = getCurrentTime();
            angle.time = endAngle - startAngle;

            angles[i] = angle;

            //OutPut AngleImg
            if (isOutputAngleImg) {
                std::string angleImgFile = getDebugImgFilePath(path, imgName, i, "-angle-");
                saveImg(angleImg, angleImgFile.c_str());
            }
        }
    } else {
        for (int i = 0; i < size; ++i) {
            angles[i] = Angle{-1, 0.f};
        }
    }
    //Most Possible AngleIndex
    if (doAngle && mostAngle) {
        auto angleIndexes = getAngleIndexes(angles);
        double sum = std::accumulate(angleIndexes.begin(), angleIndexes.end(), 0.0);
        double halfPercent = angles.size() / 2.0f;
        int mostAngleIndex;
        if (sum < halfPercent) {//all angle set to 0
            mostAngleIndex = 0;
        } else {//all angle set to 1
            mostAngleIndex = 1;
        }
        printf("Set All Angle to mostAngleIndex(%d)\n", mostAngleIndex);
        for (int i = 0; i < angles.size(); ++i) {
            Angle angle = angles[i];
            angle.index = mostAngleIndex;
            angles.at(i) = angle;
        }
    }

    return angles;
}