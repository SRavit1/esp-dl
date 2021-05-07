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
         * NOTE: When padding_type == SAME, make sure padding is already added in input
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
         * @param activation 
         */
        template <typename input_t, typename output_t>
        void conv2d(Feature<output_t> &output, Feature<input_t> &input, std::vector<int> &padding, const Filter<output_t> &filter, const int stride_y, const int stride_x, const Bias<output_t> *bias = NULL, const Activation<output_t> *activation = NULL);

        /**
         * @brief 
         * 
         * @tparam input_t 
         * @tparam output_t 
         * @param output_exponent 
         * @param input 
         * @param filter 
         * @param stride_y 
         * @param stride_x 
         * @param pad_type 
         * @param bias 
         * @param activation 
         * @return Feature<output_t> 
         */
        template <typename input_t, typename output_t>
        Feature<output_t> conv2d(const int output_exponent, Feature<input_t> &input, const Filter<output_t> &filter, const int stride_y, const int stride_x, const padding_type_t pad_type, const Bias<output_t> *bias, const Activation<output_t> *activation)
        {
            std::vector<int> output_shape = tool2d::get_output_shape(input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type, false);
            Feature<output_t> output;
            output.set_exponent(output_exponent).set_shape(output_shape).calloc_element();

            if (pad_type == PADDING_SAME || pad_type == PADDING_SAME_MXNET)
            {
                std::vector<int> padding = tool2d::get_pad_size(output_shape, input.shape, filter.shape_with_dilation, stride_y, stride_x, pad_type);
                input.set_padding(padding);
            }

            conv2d(output, input, input.padding, filter, stride_y, stride_x, bias, activation);

            return output;
        }
    } // namespace nn
} // namespace dl