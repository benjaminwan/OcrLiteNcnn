#include "DbNet.h"
#include "OcrUtils.h"

void DbNet::setGPUIndex(int gpu_index) {
    gpuIndex = gpu_index;
    if (gpuIndex >= 0) {
        vkdev = ncnn::get_gpu_device(gpuIndex);
        if (vkdev == NULL) {
            printf("vkdev NULL! There is no GPU(%d)! Try to use the default GPU\n", gpuIndex);
            vkdev = ncnn::get_gpu_device();
        }
        if (vkdev != NULL) {
            net.opt.use_vulkan_compute = true;
            g_blob_vkallocator = new ncnn::VkBlobAllocator(vkdev);
            g_staging_vkallocator = new ncnn::VkStagingAllocator(vkdev);
            net.opt.blob_vkallocator = g_blob_vkallocator;
            net.opt.workspace_vkallocator = g_blob_vkallocator;
            net.opt.staging_vkallocator = g_staging_vkallocator;
        } else {
            fprintf(stderr, "vkdev NULL! GPU init failed! force to use CPU!\n");
        }
    } else {
        net.opt.use_vulkan_compute = false;
    }
}

DbNet::~DbNet() {
    if (vkdev != NULL) {
        delete vkdev;
    }
    if (g_blob_vkallocator != NULL) {
        g_blob_vkallocator->clear();
        delete g_blob_vkallocator;
    }
    if (g_staging_vkallocator != NULL) {
        g_staging_vkallocator->clear();
        delete g_staging_vkallocator;
    }
    net.clear();
}

void DbNet::setNumOfThreads(int numOfThread) {
    numThread = numOfThread;
}

bool DbNet::initModel(std::string &pathStr) {
    int dbParam = net.load_param((pathStr + "/dbnet_op.param").c_str());
    int dbModel = net.load_model((pathStr + "/dbnet_op.bin").c_str());
    if (dbParam != 0 || dbModel != 0) {
        printf("DBNet load param(%d), model(%d)\n", dbParam, dbModel);
        return false;
    } else {
        return true;
    }
}

void DbNet::initVulkanCompute(){
    if (net.opt.use_vulkan_compute){
        if (g_blob_vkallocator != NULL) {
            g_blob_vkallocator->clear();
        }
        if (g_staging_vkallocator != NULL) {
            g_staging_vkallocator->clear();
        }
    }
}

std::vector<TextBox>
DbNet::getTextBoxes(cv::Mat &src, ScaleParam &s,
                    float boxScoreThresh, float boxThresh, float minArea, float unClipRatio) {
    cv::Mat srcResize;
    resize(src, srcResize, cv::Size(s.dstWidth, s.dstHeight));
    initVulkanCompute();
    ncnn::Mat input = ncnn::Mat::from_pixels(srcResize.data, ncnn::Mat::PIXEL_RGB,
                                             srcResize.cols, srcResize.rows);

    input.substract_mean_normalize(meanValues, normValues);
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_num_threads(numThread);
    extractor.input("input0", input);
    ncnn::Mat out;
    extractor.extract("out1", out);

    cv::Mat fMapMat(srcResize.rows, srcResize.cols, CV_32FC1);
    memcpy(fMapMat.data, (float *) out.data, srcResize.rows * srcResize.cols * sizeof(float));

    std::vector<TextBox> rsBoxes;
    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;
    rsBoxes.clear();
    std::vector<std::vector<cv::Point>> contours;
    findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); ++i) {
        std::vector<cv::Point> minBox;
        float minEdgeSize, allEdgeSize;
        getMiniBoxes(contours[i], minBox, minEdgeSize, allEdgeSize);

        if (minEdgeSize < minArea)
            continue;
        float score = boxScoreFast(fMapMat, contours[i]);

        if (score < boxScoreThresh)
            continue;
        //---use clipper start---
        std::vector<cv::Point> newbox;
        unClip(minBox, allEdgeSize, newbox, unClipRatio);

        getMiniBoxes(newbox, minBox, minEdgeSize, allEdgeSize);

        if (minEdgeSize < minArea + 2)
            continue;
        //---use clipper end---

        for (int j = 0; j < minBox.size(); ++j) {
            minBox[j].x = (minBox[j].x / s.scaleWidth);
            minBox[j].x = (std::min)((std::max)(minBox[j].x, 0), s.srcWidth);

            minBox[j].y = (minBox[j].y / s.scaleHeight);
            minBox[j].y = (std::min)((std::max)(minBox[j].y, 0), s.srcHeight);
        }

        rsBoxes.emplace_back(TextBox{minBox, score});
    }
    reverse(rsBoxes.begin(), rsBoxes.end());
    return rsBoxes;
}
