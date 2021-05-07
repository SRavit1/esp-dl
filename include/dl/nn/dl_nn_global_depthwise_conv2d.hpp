#pragma once

#include "dl_constant.hpp"
#include "dl_variable.hpp"
#include "dl_tool.hpp"

namespace dl
{
    namespace nn
    {
        template <typename input_t, typename output_t>
        void global_depthwise_conv2d(Feature<output_t> &output, Feature<input_t> &input, const Filter<input_t> &filter);

        template <typename input_t, typename output_t>
        void global_depthwise_conv2d(Feature<output_t> &output, Feature<input_t> &input, const Filter<input_t> &filter, const Bias<input_t> &bias);

        template <typename input_t, typename output_t>
        Feature<output_t> global_depthwise_conv2d(Feature<input_t> &input, const Filter<output_t> &filter, const Bias<output_t> &bias, const int output_exponent)
        {
            std::vector<int> output_shape = dl::tool2d::get_output_shape(input.shape, filter.shape_with_dilation, 1, 1, PADDING_VALID, true);
            Feature<output_t> output;
            output.set_exponent(output_exponent).set_shape(output_shape).calloc_element();

            global_depthwise_conv2d(output, input, filter, bias);

            return output;
        }

        template <typename input_t, typename output_t>
        Feature<output_t> global_depthwise_conv2d(Feature<input_t> &input, const Filter<output_t> &filter, const int output_exponent)
        {
            std::vector<int> output_shape = dl::tool2d::get_output_shape(input.shape, filter.shape_with_dilation, 1, 1, PADDING_VALID, true);
            Feature<output_t> output;
            output.set_exponent(output_exponent).set_shape(output_shape).calloc_element();

            global_depthwise_conv2d(output, input, filter);

            return output;
        }
    } // namespace nn
} // namespace dl