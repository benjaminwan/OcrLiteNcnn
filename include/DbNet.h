#ifndef __OCR_DBNET_H__
#define __OCR_DBNET_H__

#include "OcrStruct.h"
#include "ncnn/net.h"
#include <opencv/cv.hpp>

class DbNet {
public:
    ~DbNet();

    void setNumOfThreads(int numOfThread);

    void setGPUIndex(int gpuIndex);

    bool initModel(std::string &pathStr);

    std::vector<TextBox> getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh,
                                 float boxThresh, float minArea, float unClipRatio);

private:
    int numThread;
    ncnn::Net net;
    const float meanValues[3] = {0.485 * 255, 0.456 * 255, 0.406 * 255};
    const float normValues[3] = {1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0, 1.0 / 0.225 / 255.0};

    void initVulkanCompute();
    int gpuIndex = -1;
    ncnn::VulkanDevice* vkdev = NULL;
    ncnn::VkAllocator* g_blob_vkallocator = NULL;
    ncnn::VkAllocator* g_staging_vkallocator = NULL;
};


#endif //__OCR_DBNET_H__
