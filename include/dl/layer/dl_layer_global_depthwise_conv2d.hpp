#pragma once

#include <vector>
#include "dl_constant.hpp"
#include "dl_variable.hpp"
#include "dl_tool.hpp"
#include "dl_nn_global_depthwise_conv2d.hpp"

namespace dl
{
    namespace layer
    {
        /**
         * @brief Activation(GlobalDepthwiseConv2D(filter, input) + bias)
         * 
         * @tparam input_t      type of input
         * @tparam output_t     type of output
         */
        template <typename input_t, typename output_t>
        class GlobalDepthwiseConv2D : public Layer
        {
        private:
            const Filter<output_t> *filter;
            const Bias<output_t> *bias;
            const Activation<output_t> *activation;

        public:
            Feature<output_t> output;

            /**
             * @brief Construct a new Global Depthwise Conv2D object.
             * 
             * @param output_exponent 
             * @param filter 
             * @param bias 
             * @param activation 
             * @param name 
             */
            GlobalDepthwiseConv2D(const int output_exponent,
                                  const Filter<output_t> *filter,
                                  const Bias<output_t> *bias = NULL,
                                  const Activation<output_t> *activation = NULL,
                                  const char *name = NULL) : Layer(name),
                                                             filter(filter),
                                                             bias(bias),
                                                             activation(activation)
            {
                this->output.set_exponent(output_exponent);
            }

            /**
             * @brief Destroy the Global Depthwise Conv2D object.
             * 
             */
            ~GlobalDepthwiseConv2D() {}

            /**
             * @brief update output shape and padding.
             * 
             * @param input 
             */
            void build(Feature<input_t> &input)
            {
                this->build(this->output, input);
            }

            /**
             * @brief update output shape and padding.
             * 
             * @param output 
             * @param input 
             */
            void build(Feature<output_t> &output, Feature<input_t> &input)
            {
                assert(input.shape[0] > 0);
                assert(input.shape[1] > 0);

                std::vector<int> output_shape = tool2d::get_output_shape(input.shape, this->filter->shape_with_dilation, 1, 1, PADDING_VALID, true);
                output.set_shape(output_shape);
            }

            /**
             * @brief calloc output's element if not calloc yet. Call global_depthwise_conv2d.
             * 
             * @param input 
             * @param autoload_enable 1: inference with autoload. 0: inference without autoload.
             * @return Feature<output_t>& 
             */
            Feature<output_t> &call(Feature<input_t> &input, uint8_t autoload_enable = 0)
            {
#if CONFIG_DEBUG_MODE
                printf("%s:\n", this->name);
                dl::tool::Latency latency;
                latency.start();
#endif
                this->output.calloc_element();
#if CONFIG_DEBUG_MODE
                latency.end();
                latency.print("\tcalloc");
#endif
                this->call(this->output, input, autoload_enable);
                return this->output;
            }

            /**
             * @brief Call global_depthwise_conv2d.
             * 
             * @param output 
             * @param input 
             * @param autoload_enable 1: inference with autoload. 0: inference without autoload.
             */
            void call(Feature<output_t> &output, Feature<input_t> &input, uint8_t autoload_enable = 0)
            {
                if (autoload_enable)
                {
                    this->autoload(output, input);
                }
                if (this->bias)
                {
                    nn::global_depthwise_conv2d(output, input, *(this->filter), *(this->bias));
                }
                else
                {
                    nn::global_depthwise_conv2d(output, input, *(this->filter));
                }
            }

            /**
             * @brief Preload the filter to Cache. 
             * 
             */
            void preload()
            {
                size_t size = sizeof(output_t);
                int shape_size = this->filter->shape.size();
                for (int i = 0; i < shape_size; ++i)
                {
                    size *= filter->shape[i];
                }
                dl::cachetool::preload_func((uint32_t)(this->filter->element), size);
            }

            /**
             * @brief Autoload the output feature and input feature to Cache.
             * 
             * @param output 
             * @param input
             */
            static inline void autoload(Feature<output_t> &output, Feature<input_t> &input)
            {
                uint32_t out_size = output.get_size() * sizeof(output_t);
                uint32_t in_size = input.get_size() * sizeof(output_t);
                dl::cachetool::autoload_func((uint32_t)(output.element), out_size, (uint32_t)(input.element), in_size);
            }
        };
    } // namespace layer
} // namespace dl
