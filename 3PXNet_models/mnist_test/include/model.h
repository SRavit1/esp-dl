#define ESP_LIB_IMPL //_3PXNET_LIB_IMPL

#include "dl_lib_matrix3d.h"
#include "esp_log.h"

#if defined(ESP_LIB_IMPL)
#include "cnn.h"
#endif

#if defined(_3PXNET_LIB_IMPL)
#define TERNARIZE_MEDIUM //BINARIZE / TERNARIZE_LOW / ... macros indicate which 3PXNet compiled model to load
#if defined(BINARIZE)
#include "binarize/source.h"
#elif defined(TERNARIZE_LOW)
#include "ternarize_low/source.h"
#elif defined(TERNARIZE_MEDIUM)
#include "ternarize_medium/source.h"
#elif defined(TERNARIZE_HIGH)
#include "ternarize_high/source.h"
#endif
#endif

static const char *TAG = "app_process";

#if defined(_3PXNET_LIB_IMPL)
dl_matrix3d_t *_3pxnet_mnist_model (dl_matrix3d_t *image) {
    dl_matrix3d_t *result = 0;

    uint8_t curr_im[784];
    for (int i = 0; i < 784; i++) {
        curr_im[i] = (uint8_t) (image->item[i]*255);
    }

#if defined(BINARIZE)
    packBinThrsArr(curr_im, l1act_bin, F1I, 1);
    int64_t time_start = esp_timer_get_time();
    FcXnorWrap(l1act_bin, l1wght, F1I, F1O, l2act_bin, bn1thr, bn1sign);
    int64_t time_fc1 = esp_timer_get_time();
    //output defined in source.h
    int res = FcXnorNoBinWrap(l2act_bin, l2wght, F2I, F2O, output, bn2mean, bn2var, bn2gamma, bn2beta);
    int64_t time_fc2 = esp_timer_get_time();
#elif defined(TERNARIZE_LOW)
    packBinThrsArr(curr_im, l1act_bin, F1I, 1);
    int64_t time_start = esp_timer_get_time();
    Fc3pxnWrap(l1act_bin, l1wght, l1ind, F1NPI, F1O, l2act_bin, bn1thr, bn1sign);
    int64_t time_fc1 = esp_timer_get_time();
    int res = FcXnorNoBinWrap(l2act_bin, l2wght, F2I, F2O, output, bn2mean, bn2var, bn2gamma, bn2beta);
    int64_t time_fc2 = esp_timer_get_time();
#elif defined(TERNARIZE_MEDIUM)
    packBinThrsArr(curr_im, l1act_bin, F1I, 1);
    int64_t time_start = esp_timer_get_time();
    Fc3pxnWrap(l1act_bin, l1wght, l1ind, F1NPI, F1O, l2act_bin, bn1thr, bn1sign);
    int64_t time_fc1 = esp_timer_get_time();
    int res = Fc3pxnNoBinWrap(l2act_bin, l2wght, l2ind, F2NPI, F2O, output, bn2mean, bn2var, bn2gamma, bn2beta);
    int64_t time_fc2 = esp_timer_get_time();
#elif defined(TERNARIZE_HIGH)
    packBinThrsArr(curr_im, l1act_bin, F1I, 1);
    int64_t time_start = esp_timer_get_time();
    Fc3pxnWrap(l1act_bin, l1wght, l1ind, F1NPI, F1O, l2act_bin, bn1thr, bn1sign);
    int64_t time_fc1 = esp_timer_get_time();
    int res = Fc3pxnNoBinWrap(l2act_bin, l2wght, l2ind, F2NPI, F2O, output, bn2mean, bn2var, bn2gamma, bn2beta);
    int64_t time_fc2 = esp_timer_get_time();
#endif

    ESP_LOGI(TAG, "forward pass finished in %lld mu_s.", (time_fc2 - time_start));

    ESP_LOGI(TAG, "fc1 time: %lld mu_s.", (time_fc1 - time_start));
    ESP_LOGI(TAG, "fc2 time: %lld mu_s.", (time_fc2 - time_fc1));

    result = (dl_matrix3d_t *)dl_lib_calloc(1, sizeof(dl_matrix3d_t), 0);
    result->n = 1;
    result->h = 1;
    result->w = 1;
    result->c = 10;
    result->stride = 10;
    result->item = output;

    return result;
}
#endif

#if defined(ESP_LIB_IMPL)
dl_matrix3d_t *esp_mnist_model (dl_matrix3d_t *image) {
    dl_matrix3d_t *result = 0;
    
    dl_matrix3d_t *o1_1 = dl_matrix3dff_conv_common(image, &conv2d_kernel, &conv2d_bias, 1, 1, PADDING_SAME);
    dl_matrix3d_t *o1_2 = dl_matrix3d_pooling(o1_1, 2, 2, 2, 2, PADDING_VALID, DL_POOLING_MAX);
    dl_matrix3d_relu(o1_2);
    dl_matrix3d_t *o2_1 = dl_matrix3dff_conv_common(o1_2, &conv2d_1_kernel, &conv2d_1_bias, 1, 1, PADDING_SAME);
    dl_matrix3d_t *o2_2 = dl_matrix3d_pooling(o2_1, 2, 2, 2, 2, PADDING_VALID, DL_POOLING_MAX);
    dl_matrix3d_relu(o2_2);
    dl_matrix3d_t *o3_1 = dl_matrix3dff_conv_common(o2_2, &conv2d_2_kernel, &conv2d_2_bias, 1, 1, PADDING_SAME);
    dl_matrix3d_t *o3_2 = dl_matrix3d_pooling(o3_1, 2, 2, 2, 2, PADDING_VALID, DL_POOLING_MAX);
    dl_matrix3d_relu(o3_2);
    dl_matrix3d_t *o4_1 = dl_matrix3d_alloc(1, 1, 1, dense_kernel.h);
    dl_matrix3dff_fc_with_bias(o4_1, o3_2, &dense_kernel, &dense_bias);
    dl_matrix3d_relu(o4_1);
    dl_matrix3d_t *o4_2 = dl_matrix3d_alloc(1, 1, 1, dense_1_kernel.h);
    dl_matrix3dff_fc_with_bias(o4_2, o4_1, &dense_1_kernel, &dense_1_bias);

    result = o4_2;

    return result;
}
#endif

dl_matrix3d_t *mnist_model (dl_matrix3d_t *image) {
    dl_matrix3d_t *result = 0;
#if defined(ESP_LIB_IMPL)
    result = esp_mnist_model(image);
#elif defined(_3PXNET_LIB_IMPL)
    result = _3pxnet_mnist_model(image);
#else
    //ERROR: At least one of the above macros should be defined
#endif
    return result;
}