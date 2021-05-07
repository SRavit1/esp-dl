# ESP-FACE

This is a component which provides API for **Neural Network** and some deep-learning **Applications**, such as Cat Face Detection, Human Face Detection and Human Face Recognition. It can be used as a component of some project as it doesn't support any interface of peripherals. By default, it works along with ESP-WHO, which is a project-level repository. 

> ESP-FACE is facing a thoroughly change for the upcoming ESP32-S3. This change has been proven to be performance improvement. However, it's still an alpha version. We are working hard on perfecting it. The document tells what has been done and what will be done. We hope you'll have a good experience.
>
> Please use ESP-IDF release/v4.3 branch for ESP32-S3-beta2. Check [ESP-IDF](https://github.com/espressif/esp-idf) for ESP-IDF requirement of other chips.



## Neural Network

ESP-FACE only supports quantization calculation. Element is quantized in following rule.

$$
element_{float} * 2^{exponent} = element_{quantized}
$$



| API                                                          | [ESP32](./lib/esp32)  |       ESP32-S2        |       ESP32-C3        | [ESP32-S3-beta2/beta3](./lib/esp32s3) | [ESP32-S3](./lib/esp32s3) |
| ------------------------------------------------------------ | :-------------------: | :-------------------: | :-------------------: | :-----------------------------------: | :-----------------------: |
| [Conv2D](./include/dl/layer/dl_layer_conv2d.hpp)             | **16-bit**, **8-bit** | **16-bit**, **8-bit** | **16-bit**, **8-bit** |              **16-bit**               |     **16-bit**, 8-bit     |
| [DepthwiseConv2D](./include/dl/layer/dl_layer_depthwise_conv2d.hpp) | **16-bit**, **8-bit** | **16-bit**, **8-bit** | **16-bit**, **8-bit** |              **16-bit**               |     **16-bit**, 8-bit     |
| [Concat2D](./include/dl/layer/dl_layer_concat2d.hpp)         | **16-bit**, **8-bit** | **16-bit**, **8-bit** | **16-bit**, **8-bit** |              **16-bit**               |     **16-bit**, 8-bit     |
| [ReLU](./include/dl/dl_constant.hpp)                         | **16-bit**, **8-bit** | **16-bit**, **8-bit** | **16-bit**, **8-bit** |              **16-bit**               |     **16-bit**, 8-bit     |
| [LeakyReLU](./include/dl/dl_constant.hpp)                    | **16-bit**, **8-bit** | **16-bit**, **8-bit** | **16-bit**, **8-bit** |              **16-bit**               |     **16-bit**, 8-bit     |
| [PReLU](./include/dl/dl_constant.hpp)                        | **16-bit**, **8-bit** | **16-bit**, **8-bit** | **16-bit**, **8-bit** |              **16-bit**               |     **16-bit**, 8-bit     |


> In which, *16-bit* means 16-bit-quantization. *8-bit* means 8-bit-quantization. In bold means supported, on the contrary, means not supported yet but will be supported.

Some specific operations, e.g. Conv2D_3x3, Conv2D_1x1 and DepthwiseConv2D_3x3, are optimized and recommended strongly. Please click the chip name for more details.



## Application

| Application                          | API Navigation                                               | Example Navigation                                           |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Cat Face Detection                   | [./include/model/cat_face_detector.hpp](./include/model/cat_face_detector.hpp) | will be here: [ESP-WHO/examples/cat_face_detection](https://github.com/espressif/esp-who/tree/master/examples/cat_face_detection) |
| Human Face Detection and Recognition | [./include/model/human_face_detector.hpp](./include/model/human_face_detector.hpp) | will be here: [ESP-WHO/examples/human_face_recognition](https://github.com/espressif/esp-who/tree/master/examples/human_face_recognition) |



## Build Your Own Model

[Here](./tutorial) is a tutorial to teach you how to build your own model with ESP-FACE step by step.