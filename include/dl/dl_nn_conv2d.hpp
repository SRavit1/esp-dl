#pragma once

#include "dl_constant.hpp"
#include "dl_variable.hpp"
#include "dl_tool.hpp"

namespace dl
{
    namespace nn
    {
        /**
         * @brief Do conv2d operation. 
         * NOTE: Please use filter [3, 3, >=3, 8x] and filter [1, 1, >=3, >0] only by now
         * 
         * @tparam input_t 
         * @tparam output_t 
         * @param output    
         * @param input 
         * @param padding 
         * @param filter 
         * @param stride_y 
         * @param stride_x 
         * @param bias 
         * @param relu 
         */
        template <typename input_t, typename output_t>
        void conv2d(Feature<output_t> &output, Feature<input_t> &input, std::vector<int> &padding, const Filter<output_t> &filter, const int stride_y, const int stride_x, const Bias<output_t> &bias, const ReLU<output_t> &relu);

        template <typename input_t, typename output_t>
        void conv2d(Feature<output_t> &output, Feature<input_t> &input, std::vector<int> &padding, const Filter<output_t> &filter, const int stride_y, const int stride_x, const Bias<output_t> &bias);

        template <typename input_t, typename output_t>
        void conv2d(Feature<output_t> &output, Feature<input_t> &input, std::vector<int> &padding, const Filter<output_t> &filter, const int stride_y, const int stride_x, const ReLU<output_t> &relu);

        template <typename input_t, typename output_t>
        void conv2d(Feature<output_t> &output, Feature<input_t> &input, std::vector<int> &padding, const Filter<output_t> &filter, const int stride_y, const int stride_x);

        template <typename input_t, typename output_t>
        Feature<output_t> conv2d(Feature<input_t> &input, const Filter<output_t> &filter, const int stride_y, const int stride_x, const padding_type_t pad_type, const int output_exponent)
        {
            std::vector<int> output_shape = tool2d::get_output_shape(input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type, false);
            Feature<output_t> output;
            output.set_exponent(output_exponent).set_shape(output_shape).calloc_element();
            Feature<input_t> *padded_input = NULL;

            if (pad_type == PADDING_VALID)
            {
                padded_input = &input;
            }
            else
            {
                std::vector<int> padding = tool2d::get_pad_size(output_shape, input.shape, filter.shape, stride_y, stride_x, pad_type);
                padded_input = new Feature<input_t>(input, padding);
            }

            conv2d(output, *padded_input, padded_input->padding, filter, stride_y, stride_x);

            if (pad_type != PADDING_VALID)
            {
                delete padded_input;
            }

            return output;
        }

        template <typename input_t, typename output_t>
        Feature<output_t> conv2d(Feature<input_t> &input, const Filter<output_t> &filter, const int stride_y, const int stride_x, const padding_type_t pad_type, const Bias<output_t> &bias, const int output_exponent)
        {
            std::vector<int> output_shape = tool2d::get_output_shape(input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type, false);
            Feature<output_t> output;
            output.set_exponent(output_exponent).set_shape(output_shape).calloc_element();
            Feature<input_t> *padded_input = NULL;

            if (pad_type == PADDING_VALID)
            {
                padded_input = &input;
            }
            else
            {
                std::vector<int> padding = tool2d::get_pad_size(output_shape, input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type);
                padded_input = new Feature<input_t>(input, padding);
            }

            conv2d(output, *padded_input, padded_input->padding, filter, stride_y, stride_x, bias);

            if (pad_type != PADDING_VALID)
            {
                delete padded_input;
            }

            return output;
        }

        template <typename input_t, typename output_t>
        Feature<output_t> conv2d(Feature<input_t> &input, const Filter<output_t> &filter, const int stride_y, const int stride_x, const padding_type_t pad_type, const ReLU<output_t> &relu, const int output_exponent)
        {
            std::vector<int> output_shape = tool2d::get_output_shape(input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type, false);
            Feature<output_t> output;
            output.set_exponent(output_exponent).set_shape(output_shape).calloc_element();
            Feature<input_t> *padded_input = NULL;

            if (pad_type == PADDING_VALID)
            {
                padded_input = &input;
            }
            else
            {
                std::vector<int> padding = tool2d::get_pad_size(output_shape, input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type);
                padded_input = new Feature<input_t>(input, padding);
            }

            conv2d(output, *padded_input, padded_input->padding, filter, stride_y, stride_x, relu);

            if (pad_type != PADDING_VALID)
            {
                delete padded_input;
            }

            return output;
        }

        template <typename input_t, typename output_t>
        Feature<output_t> conv2d(Feature<input_t> &input, const Filter<output_t> &filter, const int stride_y, const int stride_x, const padding_type_t pad_type, const Bias<output_t> &bias, const ReLU<output_t> &relu, const int output_exponent)
        {
            std::vector<int> output_shape = tool2d::get_output_shape(input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type, false);
            Feature<output_t> output;
            output.set_exponent(output_exponent).set_shape(output_shape).calloc_element();
            Feature<input_t> *padded_input = NULL;

            if (pad_type == PADDING_VALID)
            {
                padded_input = &input;
            }
            else
            {
                std::vector<int> padding = tool2d::get_pad_size(output_shape, input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type);
                padded_input = new Feature<input_t>(input, padding);
            }

            conv2d(output, *padded_input, padded_input->padding, filter, stride_y, stride_x, bias, relu);

            if (pad_type != PADDING_VALID)
            {
                delete padded_input;
            }

            return output;
        }
    } // namespace nn
} // namespace dl