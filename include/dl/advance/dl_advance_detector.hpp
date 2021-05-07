#pragma once

#include <vector>
#include <list>
#include <algorithm>
#include <math.h>

#include "dl_variable.hpp"
#include "dl_image.hpp"
#include "dl_define.hpp"
#include "dl_tool.hpp"

namespace dl
{
    namespace advance
    {
        typedef struct
        {
            int category;              /*<! category index */
            float score;               /*<! score in logit */
            std::vector<int> box;      /*<! [left_up_x, left_up_y, right_down_x, right_down_y] */
            std::vector<int> keypoint; /*<! [x1, y1, x2, y2, ...] */
        } detection_prediction_t;

        /**
         * @brief 
         * 
         * @tparam model_input_t 
         * @tparam model_output_t 
         */
        template <typename model_input_t, typename model_output_t>
        class Detector
        {
        public:
            const std::vector<int> input_shape;         /*<! The shape of input */
            const float score_threshold;                /*<! Candidate box with lower score than score_threshold will be filtered */
            const float nms_threshold;                  /*<! Candidate box with higher IoU than nms_threshold will be filtered */
            const bool with_keypoint;                   /*<! true: detection with keypoint; false: detection without keypoint */
            const int top_k;                            /*<! Keep top_k number of candidate boxes */
            float resize_scale_y;                       /*<! Resize scale in vertical */
            float resize_scale_x;                       /*<! Resize scale in horizon */
            std::list<detection_prediction_t> box_list; /*<! Detected box list */
            Feature<model_input_t> resized_input;       /*<! Resized input */

            /**
             * @brief Construct a new Detector object
             * 
             * @param input_shape       the shape of input
             * @param resize_scale      resize scale
             * @param score_threshold   Candidate box with lower score than score_threshold will be filtered
             * @param nms_threshold     Candidate box with higher IoU than nms_threshold will be filtered
             * @param with_keypoint     true: detection with keypoint; false: detection without keypoint
             * @param top_k             Keep top_k number of candidate boxes
             */
            Detector(std::vector<int> input_shape,
                     const float resize_scale,
                     const float score_threshold,
                     const float nms_threshold,
                     const bool with_keypoint,
                     const int top_k) : input_shape(input_shape),
                                        score_threshold(score_threshold),
                                        nms_threshold(nms_threshold),
                                        with_keypoint(with_keypoint),
                                        top_k(top_k)
            {
                int resized_y = int(input_shape[0] * resize_scale + 0.5);
                int resized_x = int(input_shape[1] * resize_scale + 0.5);

                this->resize_scale_y = (float)input_shape[0] / resized_y;
                this->resize_scale_x = (float)input_shape[1] / resized_x;

                this->resized_input.set_shape({resized_y, resized_x, input_shape[2]});
            }

            /**
             * @brief Destroy the Detector object
             * 
             */
            ~Detector() {}

            /**
             * @brief Parse the feature map
             * 
             * @param score 
             * @param box 
             * @param stage_index 
             */
            virtual void parse_stage(Feature<model_output_t> &score, Feature<model_output_t> &box, const int stage_index) = 0;

            /**
             * @brief Parse the feature map
             * 
             * @param score 
             * @param box 
             * @param keypoint 
             * @param stage_index 
             */
            virtual void parse_stage(Feature<model_output_t> &score, Feature<model_output_t> &box, Feature<model_output_t> &keypoint, const int stage_index) = 0;

            /**
             * @brief Net forward and parse
             * 
             */
            virtual void call() = 0;

            /**
             * @brief Inference
             * 
             * @tparam T 
             * @param input 
             * @return std::list<detection_prediction_t>& 
             */
            template <typename T>
            std::list<detection_prediction_t> &infer(T *input)
            {
#if CONFIG_PRINT_DETECTOR_LATENCY
                tool::Latency latency;
                latency.start();
#endif
                // resize
                this->resized_input.calloc_element();
                image::resize_image_to_rgb888(this->resized_input.element,
                                              this->resized_input.padding[0],
                                              this->resized_input.padding[0] + this->resized_input.shape[0],
                                              this->resized_input.padding[2],
                                              this->resized_input.padding[2] + this->resized_input.shape[1],
                                              this->resized_input.shape[2],
                                              input,
                                              this->input_shape[0],
                                              this->input_shape[1],
                                              this->resized_input.shape[1],
                                              image::IMAGE_RESIZE_NEAREST);
#if CONFIG_PRINT_DETECTOR_LATENCY
                latency.end();
                latency.print("Resize");
                latency.start();
#endif

                // call
                this->box_list.clear();
                this->call();

                // NMS
#if CONFIG_PRINT_DETECTOR_LATENCY
                latency.end();
                latency.print("Call");
                latency.start();
#endif
                int kept_number = 0;
                for (std::list<detection_prediction_t>::iterator kept = this->box_list.begin(); kept != this->box_list.end(); kept++)
                {
                    kept_number++;

                    if (kept_number >= this->top_k)
                    {
                        this->box_list.erase(++kept, this->box_list.end());
                        break;
                    }

                    int kept_area = (kept->box[2] - kept->box[0] + 1) * (kept->box[3] - kept->box[1] + 1);

                    std::list<detection_prediction_t>::iterator other = kept;
                    other++;
                    for (; other != this->box_list.end();)
                    {
                        int inter_lt_x = DL_MAX(kept->box[0], other->box[0]);
                        int inter_lt_y = DL_MAX(kept->box[1], other->box[1]);
                        int inter_rb_x = DL_MIN(kept->box[2], other->box[2]);
                        int inter_rb_y = DL_MIN(kept->box[3], other->box[3]);

                        int inter_height = inter_rb_y - inter_lt_y + 1;
                        int inter_width = inter_rb_x - inter_lt_x + 1;

                        if (inter_height > 0 && inter_width > 0)
                        {
                            int other_area = (other->box[2] - other->box[0] + 1) * (other->box[3] - other->box[1] + 1);
                            int inter_area = inter_height * inter_width;
                            float iou = (float)inter_area / (kept_area + other_area - inter_area);
                            if (iou > this->nms_threshold)
                            {
                                other = this->box_list.erase(other);
                                continue;
                            }
                        }
                        other++;
                    }
                }
#if CONFIG_PRINT_DETECTOR_LATENCY
                latency.end();
                latency.print("NMS");
#endif
                return this->box_list;
            }
        };
    }
}
