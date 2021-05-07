# Tutorial

This tutorial will teach you with an example to build your own model with ESP-FACE step by step. The example is a model for MNIST classification.



#### Step 1: Save Model Coefficient

Save the float point coefficients of model into npy files layer by layer, e.g.

```python
import numpy
numpy.save(file=f'{root}/{layer_name}_filter.npy', arr=filter) # filter must be numpy.ndarray
numpy.save(file=f'{root}/{layer_name}_bias.npy', arr=bias)     # bias must be numpy.ndarray
```

> NOTE: make sure the coefficients are in numpy.ndarray type.

**e.g.** `./model/npy/` contains the npy files of the example.



#### Step 2: Write Model Configuration

Write a config.json file for the model configuration layer by layer, the format is 

```json
{
    "layer_name": {             // must be the same as the corresponding npy file
        "operation": "conv2d",  // "conv2d" or "depthwise_conv2d"
        "filter_exponent": -10, // exponent of filter. This can be settled automatically be remove this item.
        "output_type": "s16",   // data type of output, "s16" or "s8"
        "bias": "True",			// "True": with bias, "False": without bias
        "output_exponent": -10, // exponent of output. Take a not that the exponent of bias must be equal to output's
        "activation": {
            "type": "LeakyReLU" // "ReLU", "LeakyReLU", "PReLU" or with no activation by removing "activation"
            "exponent": -16		// exponent of activation if it has
        }
    }, 
    ... ...
}
```

**e.g.** `./model/npy/config.json`

```json
{
    "l1": {
        "operation": "conv2d",
        "output_type": "s16",
        "bias": "True",
        "output_exponent": -2,
        "activation": {
            "type": "ReLU"
        }
    },
    "l2_depth": {
        "operation": "depthwise_conv2d",
        "output_type": "s16",
        "activation": {
            "type": "ReLU"
        }
    },
    "l2_compress": {
        "operation": "conv2d",
        "output_type": "s16",
        "bias": "True",
        "output_exponent": -3
    },
    "l3_a_depth": {
        "operation": "depthwise_conv2d",
        "output_type": "s16",
        "activation": {
            "type": "ReLU"
        }
    },
    "l3_a_compress": {
        "operation": "conv2d",
        "output_type": "s16",
        "bias": "True",
        "output_exponent": -12
    },
    "l3_b_depth": {
        "operation": "depthwise_conv2d",
        "output_type": "s16",
        "activation": {
            "type": "ReLU"
        }
    },
    "l3_b_compress": {
        "operation": "conv2d",
        "output_type": "s16",
        "bias": "True",
        "output_exponent": -12
    },
    "l3_c_depth": {
        "operation": "depthwise_conv2d",
        "output_type": "s16",
        "activation": {
            "type": "ReLU"
        }
    },
    "l3_c_compress": {
        "operation": "conv2d",
        "output_type": "s16",
        "bias": "True",
        "output_exponent": -12
    },
    "l3_d_depth": {
        "operation": "depthwise_conv2d",
        "output_type": "s16",
        "activation": {
            "type": "ReLU"
        }
    },
    "l3_d_compress": {
        "operation": "conv2d",
        "output_type": "s16",
        "bias": "True",
        "output_exponent": -11
    },
    "l3_e_depth": {
        "operation": "depthwise_conv2d",
        "output_type": "s16",
        "activation": {
            "type": "ReLU"
        }
    },
    "l3_e_compress": {
        "operation": "conv2d",
        "output_type": "s16",
        "bias": "True",
        "output_exponent": -12
    },
    "l4_depth": {
        "operation": "depthwise_conv2d",
        "output_type": "s16",
        "activation": {
            "type": "LeakyReLU"
        }
    },
    "l4_compress": {
        "operation": "conv2d",
        "output_type": "s16",
        "bias": "True",
        "output_exponent": -11
    },
    "l5_depth": {
        "operation": "depthwise_conv2d",
        "output_type": "s16",
        "activation": {
            "type": "LeakyReLU"
        }
    },
    "l5_compress": {
        "operation": "conv2d",
        "output_type": "s16",
        "bias": "True",
        "output_exponent": -9
    }
}
```



#### Step 3: Convert Model Coefficient

Make sure that coefficient npy files and configuration file are all ready and in the same folder. Then, [convert.py](../convert.py) can help to convert coefficient into C/C++ code.

**e.g.** 

```python
# -t: target chip: esp32, esp32s2, esp32s3, esp32c3
# -i: where the coefficient and configuration files saved
# -n: generated coefficient filename
# -o: where the generated coefficient file saved
# under the tutorial root
python ../convert.py -t esp32s3 -i ./model/npy -n mnist_coefficient -o ./model
```

After that, `mnist_coefficient.cpp` and `mnist_coefficient.hpp` can be found in `./model/`

**Or**

Add command into [CMakeLists.txt](./main/CMakeLists.txt#L4) file so that the coefficient files could be generated accordingly when compile.



#### Step 4: Build Model

Build a model by deriving `Model` class in `"dl_layer_model.hpp"`. Two pure abstract functions have to be implemented:

`void build(Feature<input_t> &input)`: for passing on the change of shape and padding layer by layer

`void call(Feature<input_t> &input)`: for running the model layer by layer

**e.g.** `./model/mnist_model.hpp`

```c++
#pragma once
#include "dl_layer_model.hpp"
#include "dl_layer_conv2d.hpp"
#include "dl_layer_depthwise_conv2d.hpp"
#include "dl_layer_concat2d.hpp"
#include "mnist_coefficient.hpp"
#include <stdint.h>

using namespace dl;
using namespace layer;
using namespace mnist_coefficient;

class MNIST : public Model<int16_t, int16_t>
{
private:
    Conv2D<int16_t, int16_t> l1;
    DepthwiseConv2D<int16_t, int16_t> l2_depth;
    Conv2D<int16_t, int16_t> l2_compress;
    DepthwiseConv2D<int16_t, int16_t> l3_a_depth;
    Conv2D<int16_t, int16_t> l3_a_compress;
    DepthwiseConv2D<int16_t, int16_t> l3_b_depth;
    Conv2D<int16_t, int16_t> l3_b_compress;
    DepthwiseConv2D<int16_t, int16_t> l3_c_depth;
    Conv2D<int16_t, int16_t> l3_c_compress;
    DepthwiseConv2D<int16_t, int16_t> l3_d_depth;
    Conv2D<int16_t, int16_t> l3_d_compress;
    DepthwiseConv2D<int16_t, int16_t> l3_e_depth;
    Conv2D<int16_t, int16_t> l3_e_compress;
    Concat2D<int16_t> l3_concat;
    DepthwiseConv2D<int16_t, int16_t> l4_depth;
    Conv2D<int16_t, int16_t> l4_compress;
    DepthwiseConv2D<int16_t, int16_t> l5_depth;

public:
    Conv2D<int16_t, int16_t> l5_compress;
    MNIST() : l1(Conv2D<int16_t, int16_t>(-2, get_l1_filter(), get_l1_bias(), get_l1_activation(), PADDING_VALID, 2, 2, "l1")),
              l2_depth(DepthwiseConv2D<int16_t, int16_t>(-1, get_l2_depth_filter(), NULL, get_l2_depth_activation(), PADDING_SAME, 2, 2, "l2_depth")),
              l2_compress(Conv2D<int16_t, int16_t>(-3, get_l2_compress_filter(), get_l2_compress_bias(), NULL, PADDING_SAME, 1, 1, "l2_compress")),
              l3_a_depth(DepthwiseConv2D<int16_t, int16_t>(-1, get_l3_a_depth_filter(), NULL, get_l3_a_depth_activation(), PADDING_VALID, 1, 1, "l3_a_depth")),
              l3_a_compress(Conv2D<int16_t, int16_t>(-12, get_l3_a_compress_filter(), get_l3_a_compress_bias(), NULL, PADDING_VALID, 1, 1, "l3_a_compress")),
              l3_b_depth(DepthwiseConv2D<int16_t, int16_t>(-2, get_l3_b_depth_filter(), NULL, get_l3_b_depth_activation(), PADDING_VALID, 1, 1, "l3_b_depth")),
              l3_b_compress(Conv2D<int16_t, int16_t>(-12, get_l3_b_compress_filter(), get_l3_b_compress_bias(), NULL, PADDING_VALID, 1, 1, "l3_b_compress")),
              l3_c_depth(DepthwiseConv2D<int16_t, int16_t>(-12, get_l3_c_depth_filter(), NULL, get_l3_c_depth_activation(), PADDING_SAME, 1, 1, "l3_c_depth")),
              l3_c_compress(Conv2D<int16_t, int16_t>(-12, get_l3_c_compress_filter(), get_l3_c_compress_bias(), NULL, PADDING_SAME, 1, 1, "l3_c_compress")),
              l3_d_depth(DepthwiseConv2D<int16_t, int16_t>(-12, get_l3_d_depth_filter(), NULL, get_l3_d_depth_activation(), PADDING_SAME, 1, 1, "l3_d_depth")),
              l3_d_compress(Conv2D<int16_t, int16_t>(-11, get_l3_d_compress_filter(), get_l3_d_compress_bias(), NULL, PADDING_SAME, 1, 1, "l3_d_compress")),
              l3_e_depth(DepthwiseConv2D<int16_t, int16_t>(-11, get_l3_e_depth_filter(), NULL, get_l3_e_depth_activation(), PADDING_SAME, 1, 1, "l3_e_depth")),
              l3_e_compress(Conv2D<int16_t, int16_t>(-12, get_l3_e_compress_filter(), get_l3_e_compress_bias(), NULL, PADDING_SAME, 1, 1, "l3_e_compress")),
              l3_concat("l3_concat"),
              l4_depth(DepthwiseConv2D<int16_t, int16_t>(-12, get_l4_depth_filter(), NULL, get_l4_depth_activation(), PADDING_VALID, 1, 1, "l4_depth")),
              l4_compress(Conv2D<int16_t, int16_t>(-11, get_l4_compress_filter(), get_l4_compress_bias(), NULL, PADDING_VALID, 1, 1, "l4_compress")),
              l5_depth(DepthwiseConv2D<int16_t, int16_t>(-10, get_l5_depth_filter(), NULL, get_l5_depth_activation(), PADDING_VALID, 1, 1, "l5_depth")),
              l5_compress(Conv2D<int16_t, int16_t>(-9, get_l5_compress_filter(), get_l5_compress_bias(), NULL, PADDING_VALID, 1, 1, "l5_compress")) {}

    void build(Feature<int16_t> &input)
    {
        this->l1.build(input);
        this->l2_depth.build(this->l1.output);
        this->l2_compress.build(this->l2_depth.output);
        this->l3_a_depth.build(this->l2_compress.output);
        this->l3_a_compress.build(this->l3_a_depth.output);
        this->l3_b_depth.build(this->l2_compress.output);
        this->l3_b_compress.build(this->l3_b_depth.output);
        this->l3_c_depth.build(this->l3_b_compress.output);
        this->l3_c_compress.build(this->l3_c_depth.output);
        this->l3_d_depth.build(this->l3_b_compress.output);
        this->l3_d_compress.build(this->l3_d_depth.output);
        this->l3_e_depth.build(this->l3_d_compress.output);
        this->l3_e_compress.build(this->l3_e_depth.output);
        this->l3_concat.build({&this->l3_a_compress.output, &this->l3_c_compress.output, &this->l3_e_compress.output});
        this->l4_depth.build(this->l3_concat.output);
        this->l4_compress.build(this->l4_depth.output);
        this->l5_depth.build(this->l4_compress.output);
        this->l5_compress.build(this->l5_depth.output);

        this->l3_concat.backward();
    }

    void call(Feature<int16_t> &input)
    {
        this->l1.call(input);
        input.free_element();

        this->l2_depth.call(this->l1.output);
        this->l1.output.free_element();

        this->l2_compress.call(this->l2_depth.output);
        this->l2_depth.output.free_element();

        this->l3_a_depth.call(this->l2_compress.output);
        // this->l2_compress.output.free_element();

        this->l3_concat.calloc_element(); // calloc a memory for layers concat in future.

        this->l3_a_compress.call(this->l3_a_depth.output);
        this->l3_a_depth.output.free_element();

        this->l3_b_depth.call(this->l2_compress.output);
        this->l2_compress.output.free_element();

        this->l3_b_compress.call(this->l3_b_depth.output);
        this->l3_b_depth.output.free_element();

        this->l3_c_depth.call(this->l3_b_compress.output);
        // this->l3_b_compress.output.free_element();

        this->l3_c_compress.call(this->l3_c_depth.output);
        this->l3_c_depth.output.free_element();

        this->l3_d_depth.call(this->l3_b_compress.output);
        this->l3_b_compress.output.free_element();

        this->l3_d_compress.call(this->l3_d_depth.output);
        this->l3_d_depth.output.free_element();

        this->l3_e_depth.call(this->l3_d_compress.output);
        this->l3_d_compress.output.free_element();

        this->l3_e_compress.call(this->l3_e_depth.output);
        this->l3_e_depth.output.free_element();

        this->l4_depth.call(this->l3_concat.output);
        this->l3_concat.output.free_element();

        this->l4_compress.call(this->l4_depth.output);
        this->l4_depth.output.free_element();

        this->l5_depth.call(this->l4_compress.output);
        this->l4_compress.output.free_element();

        this->l5_compress.call(this->l5_depth.output);
        this->l5_depth.output.free_element();
    }
};
```



#### Step 5: Run Model

Create a model object and run its `forward()` function

**e.g.** `./main/app_main.cpp`

```c++
#include <stdio.h>
#include <stdlib.h>

#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "mnist_model.hpp"

__attribute__((aligned(16))) int16_t example_element[] = {...};

extern "C" void app_main(void)
{
    // input
    Feature<int16_t> input;
    input.set_element((int16_t *)example_element).set_exponent(0).set_shape({28, 28, 3}).set_auto_free(false);

    // model forward
    MNIST model;
    model.forward(input);

    // parse
    int16_t *score = model.l5_compress.output.get_element_ptr();
    int16_t max_score = score[0];
    int max_index = 0;
    printf("%d, ", max_score);
    for (size_t i = 1; i < 10; i++)
    {
        printf("%d, ", score[i]);
        if (score[i] > max_score)
        {
            max_score = score[i];
            max_index = i;
        }
    }
    printf("\nPrediction Result: %d\n", max_index);
}
```

flash and monitor

```bash
$ idf.py -p /dev/ttyUSB0 flash monitor # NOTICE: please select a right device

// esp32
// -7166, -9783, -12293, -11405, -12351, -1363, -11715, -116, -11436, 7851,
// Prediction Result: 9

// esp32s2
// -7166, -9783, -12293, -11405, -12351, -1363, -11715, -116, -11436, 7851
// Prediction Result: 9

// esp32s3
// -7166, -9783, -12293, -11405, -12351, -1363, -11715, -116, -11436, 7851
// Prediction Result: 9

// esp32c3
// -7166, -9783, -12293, -11405, -12351, -1363, -11715, -116, -11436, 7851
// Prediction Result: 9
```

