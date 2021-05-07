#pragma once

#include <vector>
#include "dl_advance_detector_anchor_point.hpp"

class CatFaceDetector : public dl::advance::DetectorAnchorPoint<int16_t, int16_t>
{
public:
    /**
     * @brief Construct a new Human Face Detector object
     * 
     * @param input_shape       The shape of input image
     * @param resize_scale      The resize scale
     * @param score_threshold   The predicted boxes with higher score than the threshold will be remained
     * @param nms_threshold     The predicted boxes with lower IoU than the threshold will be remained
     * @param top_k             The k highest score boxes will be remained 
     */
    CatFaceDetector(std::vector<int> input_shape, float resize_scale, const float score_threshold, const float nms_threshold, const int top_k);

    /**
     * @brief Destroy the Cat Face Detector object
     * 
     */
    ~CatFaceDetector();

    /**
     * @brief Forward model and parse output feature map
     * 
     */
    void call();
};
