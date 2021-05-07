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
        } detector_anchor_point_stage_t;

        template <typename model_input_t, typename model_output_t>
        class DetectorAnchorPoint : public Detector<model_input_t, model_output_t>
        {
        public:
            std::vector<detector_anchor_point_stage_t> stages;

            /**
             * @brief Construct a new Detector Anchor Point object
             * 
             * @param input_shape 
             * @param resize_scale 
             * @param score_threshold 
             * @param nms_threshold 
             * @param with_keypoint 
             * @param top_k 
             * @param stages 
             */
            DetectorAnchorPoint(std::vector<int> input_shape,
                                const float resize_scale,
                                const float score_threshold,
                                const float nms_threshold,
                                const bool with_keypoint,
                                const int top_k,
                                const std::vector<detector_anchor_point_stage_t> stages);

            /**
             * @brief Destroy the Detector Anchor Point object
             * 
             */
            ~DetectorAnchorPoint();

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
