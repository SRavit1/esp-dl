#pragma once

#include "dl_constant.hpp"
#include "dl_variable.hpp"
#include "dl_tool.hpp"
#include "dl_nn_conv2d.hpp"
#include "dl_layer_base.hpp"

namespace dl
{
    namespace layer
    {
        /**
         * @brief Activation(Conv2D(input, filter) + bias)
         * 
         * @tparam input_t 
         * @tparam output_t 
         */
        template <typename input_t, typename output_t>
        class Conv2D : public Layer
        {
        private:
            const Filter<output_t> *filter;         /*<! filter >*/
            const int stride_y;                     /*<! stride in height >*/
            const int stride_x;                     /*<! stride in width >*/
            const padding_type_t padding_type;      /*<! padding type >*/
            const Bias<output_t> *bias;             /*<! bias >*/
            const Activation<output_t> *activation; /*<! activation >*/
            std::vector<int> padding;               /*<! padding >*/

        public:
            Feature<output_t> output; /*<! output >*/

            /**
             * @brief Construct a new Conv2D object
             * 
             * @param output_exponent 
             * @param filter 
             * @param bias 
             * @param activation 
             * @param padding_type 
             * @param stride_y 
             * @param stride_x 
             * @param name 
             */
            Conv2D(const int output_exponent,
                   const Filter<output_t> *filter,
                   const Bias<output_t> *bias = NULL,
                   const Activation<output_t> *activation = NULL,
                   const padding_type_t padding_type = PADDING_VALID,
                   const int stride_y = 1,
                   const int stride_x = 1,
                   const char *name = NULL) : Layer(name),
                                              filter(filter),
                                              stride_y(stride_y),
                                              stride_x(stride_x),
                                              padding_type(padding_type),
                                              bias(bias),
                                              activation(activation)
            {
                this->output.set_exponent(output_exponent);
            }

            /**
             * @brief Destroy the Conv2D object
             * 
             */
            ~Conv2D() {}

            /**
             * @brief Update output padding and input padding
             * 
             * @param input 
             */
            void build(Feature<input_t> &input)
            {
                this->build(this->output, input);
            }

            /**
             * @brief Update output padding and input padding
             * 
             * @param output 
             * @param input 
             */
            void build(Feature<output_t> &output, Feature<input_t> &input)
            {
                assert(input.shape[0] > 0);
                assert(input.shape[1] > 0);

                std::vector<int> output_shape = tool2d::get_output_shape(input.shape, this->filter->shape_with_dilation, this->stride_y, this->stride_x, this->padding_type, false);
                output.set_shape(output_shape);

                this->padding = tool2d::get_pad_size(output_shape, input.shape, this->filter->shape_with_dilation, this->stride_y, this->stride_x, this->padding_type);
                input.set_padding(this->padding);
            }

            /**
             * @brief Calloc output's element if not calloc yet. Call depthwise_conv2d
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
             * @brief Call depthwise_conv2d
             * 
             * @param output 
             * @param input
             * @param autoload_enable 1: inference with autoload. 0: inference without autoload. 
             */
            void call(Feature<output_t> &output, Feature<input_t> &input, uint8_t autoload_enable = 0)
            {
#if CONFIG_DEBUG_MODE
                dl::tool::Latency latency;
                latency.start();
#endif
                if (autoload_enable)
                {
                    this->autoload(output, input);
                }

                nn::conv2d(output, input, this->padding, *(this->filter), this->stride_y, this->stride_x, this->bias, this->activation);
#if CONFIG_DEBUG_MODE
                latency.end();
                latency.print(this->name);
#endif
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
