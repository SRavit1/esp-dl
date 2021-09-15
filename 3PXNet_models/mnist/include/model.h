#define ESP_LIB_IMPL

#include "dl_lib_matrix3d.h"
#include "esp_log.h"

#if defined(ESP_LIB_IMPL)
#define QUANTIZED //QUANTIZED / FULL_PREC
#if defined(FULL_PREC)
#include "cnn.h"
#elif defined(QUANTIZED)
#define QUANTIZED_8 //QUANTIZED_8 / QUANTIZED_16
#if defined(QUANTIZED_8)
#include "cnn_qu_8.h"
#elif defined(QUANTIZED_16)
#include "cnn_qu.h"
#endif
#endif
#endif

#if defined(_3PXNET_LIB_IMPL)
#define TERNARIZE_HIGH //BINARIZE / TERNARIZE_LOW / ... macros indicate which 3PXNet compiled model to load
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

#if defined(ESP_LIB_IMPL)
#if defined(FULL_PREC)
dl_matrix3d_t *esp_mnist_model (dl_matrix3d_t *image) {
    dl_matrix3d_t *result = 0;

    image->w *= image->h;
    image->h = 1;

    dl_matrix3d_t *fc1_out=dl_matrix3d_alloc(1, 1, 1, 512);
    dl_matrix3d_t *fc2_out=dl_matrix3d_alloc(1, 1, 1, 10);

    int64_t time_start = esp_timer_get_time();
    dl_matrix3dff_fc(fc1_out, image, &fc1_filter);
    int64_t time_fc1 = esp_timer_get_time();
    dl_matrix3dff_fc(fc2_out, fc1_out, &fc2_filter);
    int64_t time_fc2 = esp_timer_get_time();

    ESP_LOGI(TAG, "forward pass finished in %lld mu_s.", (time_fc2 - time_start));

    ESP_LOGI(TAG, "fc1 time: %lld mu_s.", (time_fc1 - time_start));
    ESP_LOGI(TAG, "fc2 time: %lld mu_s.", (time_fc2 - time_fc1));

    result = fc2_out;

    return result;
}
#elif defined(QUANTIZED)
dl_matrix3d_t *esp_mnist_model (dl_matrix3d_t *image) {
    dl_matrix3d_t *result = 0;

    image->w *= image->h;
    image->h = 1;

    dl_matrix3dq_t *image_qu = dl_matrix3dq_alloc(1, 28, 28, 1, 0); //dl_matrixq_from_matrix3d_qmf(image, 1);
    for (int i = 0; i < 784; i++)
        image_qu->item[i] = (uint8_t) (image->item[i]*255);

    dl_matrix3dq_t *fc1_out = dl_matrix3dq_alloc(1, 1, 1, 512, 0);
    dl_matrix3dq_t *fc2_out = dl_matrix3dq_alloc(1, 1, 1, 10, 0);

    int64_t time_start = esp_timer_get_time();
    dl_matrix3dqq_fc(fc1_out, image_qu, &fc1_filter, DL_C_IMPL, "FC1");
    int64_t time_fc1 = esp_timer_get_time();
    dl_matrix3dqq_fc(fc2_out, fc1_out, &fc2_filter, DL_C_IMPL, "FC2");
    int64_t time_fc2 = esp_timer_get_time();

    ESP_LOGI(TAG, "forward pass finished in %lld mu_s.", (time_fc2 - time_start));

    ESP_LOGI(TAG, "fc1 time: %lld mu_s.", (time_fc1 - time_start));
    ESP_LOGI(TAG, "fc2 time: %lld mu_s.", (time_fc2 - time_fc1));

    result = dl_matrix3d_from_matrixq(fc2_out);
    return result;
}
#endif
#endif

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