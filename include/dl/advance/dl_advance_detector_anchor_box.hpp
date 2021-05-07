#pragma once

#include "dl_advance_detector.hpp"

namespace dl
{
    namespace advance
    {
        typedef struct
        {
            int stride_y;
            int stride_x;
            int offset_y;
            int offset_x;
            int min_input_size;
            std::vector<std::vector<int>> anchor_shape;
        } detector_anchor_box_stage_t;

        template <typename model_input_t, typename model_output_t>
        class DetectorAnchorBox : public Detector<model_input_t, model_output_t>
        {
        public:
            std::vector<detector_anchor_box_stage_t> stages;

            /**
             * @brief Construct a new Detector Anchor Box object
             * 
             * @param input_shape 
             * @param resize_scale 
             * @param score_threshold 
             * @param nms_threshold 
             * @param with_keypoint 
             * @param top_k 
             * @param stages 
             */
            DetectorAnchorBox(std::vector<int> input_shape, const float resize_scale, const float score_threshold, const float nms_threshold, const bool with_keypoint, const int top_k, const std::vector<detector_anchor_box_stage_t> stages);

            /**
             * @brief Destroy the Detector Anchor Box object
             * 
             */
            ~DetectorAnchorBox();

            /**
             * @brief 
             * 
             * @param score 
             * @param box 
             * @param stage_index 
             */
            void parse_stage(Feature<model_output_t> &score, Feature<model_output_t> &box, const int stage_index);

            /**
             * @brief 
             * 
             * @param score 
             * @param box 
             * @param keypoint 
             * @param stage_index 
             */
            void parse_stage(Feature<model_output_t> &score, Feature<model_output_t> &box, Feature<model_output_t> &keypoint, const int stage_index);
        };
    }
}