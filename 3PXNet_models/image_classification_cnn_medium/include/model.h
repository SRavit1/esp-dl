#define _3PXNET_LIB_IMPL

#include "dl_lib_matrix3d.h"
#include "esp_log.h"

#if defined(ESP_LIB_IMPL)
//#include "cnn.h"
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
dl_matrix3d_t *esp_mnist_model (dl_matrix3d_t *image) {
    //TODO: Not correct model
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
#endif

#if defined(_3PXNET_LIB_IMPL)
dl_matrix3d_t *_3pxnet_mnist_model (dl_matrix3d_t *image) {
    dl_matrix3d_t *result = 0;

    uint8_t curr_im[3072];
    for (int i = 0; i < 3072; i++) {
        curr_im[i] = (uint8_t) (image->item[i]*255);
    }

#if defined(BINARIZE)
    CnBnBwn(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr, bn1sign);
    CnXnorWrap(l2act_bin, l2wght, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, bn2thr, bn2sign);
    CnXnorWrap(l3act_bin, l3wght, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, bn3thr, bn3sign);
    CnXnorWrap(l4act_bin, l4wght, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, bn4thr, bn4sign);
    CnXnorWrap(l5act_bin, l5wght, C5Z, C5XY, C5XY, C5Z, C5KXY, C5KXY, C5KZ, l6act_bin, C5PD, C5PL, bn5thr, bn5sign);
    CnXnorWrap(l6act_bin, l6wght, C6Z, C6XY, C6XY, C6Z, C6KXY, C6KXY, C6KZ, l7act_bin, C6PD, C6PL, bn6thr, bn6sign);
    /*int res = */CnXnorNoBinWrap(l7act_bin, l7wght, C7Z, C7XY, C7XY, C7Z, C7KXY, C7KXY, C7KZ, output, C7PD, C7PL, bn7mean, bn7var, bn7gamma, bn7beta);
#elif defined(TERNARIZE_LOW)
    CnBnBwn(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr, bn1sign);
    Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, bn2thr, bn2sign);
    Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, bn3thr, bn3sign);
    Cn3pxnWrap(l4act_bin, l4wght, l4ind, C4NPI, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, bn4thr, bn4sign);
    Cn3pxnWrap(l5act_bin, l5wght, l5ind, C5NPI, C5Z, C5XY, C5XY, C5Z, C5KXY, C5KXY, C5KZ, l6act_bin, C5PD, C5PL, bn5thr, bn5sign);
    Cn3pxnWrap(l6act_bin, l6wght, l6ind, C6NPI, C6Z, C6XY, C6XY, C6Z, C6KXY, C6KXY, C6KZ, l7act_bin, C6PD, C6PL, bn6thr, bn6sign);
    int res = Cn3pxnNoBinWrap(l7act_bin, l7wght, l7ind, C7NPI, C7Z, C7XY, C7XY, C7Z, C7KXY, C7KXY, C7KZ, output, C7PD, C7PL, bn7mean, bn7var, bn7gamma, bn7beta);
#elif defined(TERNARIZE_MEDIUM)
    CnBnBwn(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr, bn1sign);
    Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, bn2thr, bn2sign);
    Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, bn3thr, bn3sign);
    Cn3pxnWrap(l4act_bin, l4wght, l4ind, C4NPI, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, bn4thr, bn4sign);
    Cn3pxnWrap(l5act_bin, l5wght, l5ind, C5NPI, C5Z, C5XY, C5XY, C5Z, C5KXY, C5KXY, C5KZ, l6act_bin, C5PD, C5PL, bn5thr, bn5sign);
    Cn3pxnWrap(l6act_bin, l6wght, l6ind, C6NPI, C6Z, C6XY, C6XY, C6Z, C6KXY, C6KXY, C6KZ, l7act_bin, C6PD, C6PL, bn6thr, bn6sign);
    int res = Cn3pxnNoBinWrap(l7act_bin, l7wght, l7ind, C7NPI, C7Z, C7XY, C7XY, C7Z, C7KXY, C7KXY, C7KZ, output, C7PD, C7PL, bn7mean, bn7var, bn7gamma, bn7beta);
#elif defined(TERNARIZE_HIGH)
    int64_t time_start = esp_timer_get_time();
    CnBnBwn(curr_im, l1wght, C1Z, C1XY, C1XY, C1Z, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2act_bin, bn1thr, bn1sign);
    int64_t time_conv1 = esp_timer_get_time();
    Cn3pxnWrap(l2act_bin, l2wght, l2ind, C2NPI, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3act_bin, C2PD, C2PL, bn2thr, bn2sign);
    int64_t time_conv2 = esp_timer_get_time();
    Cn3pxnWrap(l3act_bin, l3wght, l3ind, C3NPI, C3Z, C3XY, C3XY, C3Z, C3KXY, C3KXY, C3KZ, l4act_bin, C3PD, C3PL, bn3thr, bn3sign);
    int64_t time_conv3 = esp_timer_get_time();
    Cn3pxnWrap(l4act_bin, l4wght, l4ind, C4NPI, C4Z, C4XY, C4XY, C4Z, C4KXY, C4KXY, C4KZ, l5act_bin, C4PD, C4PL, bn4thr, bn4sign);
    int64_t time_conv4 = esp_timer_get_time();
    Cn3pxnWrap(l5act_bin, l5wght, l5ind, C5NPI, C5Z, C5XY, C5XY, C5Z, C5KXY, C5KXY, C5KZ, l6act_bin, C5PD, C5PL, bn5thr, bn5sign);
    int64_t time_conv5 = esp_timer_get_time();
    Cn3pxnWrap(l6act_bin, l6wght, l6ind, C6NPI, C6Z, C6XY, C6XY, C6Z, C6KXY, C6KXY, C6KZ, l7act_bin, C6PD, C6PL, bn6thr, bn6sign);
    int64_t time_conv6 = esp_timer_get_time();
    int res = Cn3pxnNoBinWrap(l7act_bin, l7wght, l7ind, C7NPI, C7Z, C7XY, C7XY, C7Z, C7KXY, C7KXY, C7KZ, output, C7PD, C7PL, bn7mean, bn7var, bn7gamma, bn7beta);
    int64_t time_fc1 = esp_timer_get_time();
#endif
    //NOTE: Error occurs with printf statements
    printf("\nforward pass finished in %lld mu_s.", (time_fc1 - time_start));
    printf("\nconv1 time: %lld mu_s.", (time_conv1 - time_start));
    printf("\nconv2 time: %lld mu_s.", (time_conv2 - time_conv1));
    printf("\nconv3 time: %lld mu_s.", (time_conv3 - time_conv2));
    printf("\nconv4 time: %lld mu_s.", (time_conv4 - time_conv3));
    printf("\nconv5 time: %lld mu_s.", (time_conv5 - time_conv4));
    printf("\nconv6 time: %lld mu_s.", (time_conv6 - time_conv5));
    printf("\nfc1 time: %lld mu_s.", (time_fc1 - time_conv6));

    /*
    ESP_LOGI(TAG, "forward pass finished in %lld mu_s.", (time_fc1 - time_start));
    ESP_LOGI(TAG, "conv1 time: %lld mu_s.", (time_conv1 - time_start));
    ESP_LOGI(TAG, "conv2 time: %lld mu_s.", (time_conv2 - time_conv1));
    ESP_LOGI(TAG, "conv3 time: %lld mu_s.", (time_conv3 - time_conv2));
    ESP_LOGI(TAG, "conv4 time: %lld mu_s.", (time_conv4 - time_conv3));
    ESP_LOGI(TAG, "conv5 time: %lld mu_s.", (time_conv5 - time_conv4));
    ESP_LOGI(TAG, "conv6 time: %lld mu_s.", (time_conv6 - time_conv5));
    ESP_LOGI(TAG, "fc1 time: %lld mu_s.", (time_fc1 - time_conv6));
    */

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