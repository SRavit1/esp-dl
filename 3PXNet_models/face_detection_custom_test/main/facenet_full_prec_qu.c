#include "facenet_full_prec_qu.h"
#include "dl_lib_matrix3d.h"
#include "mtmn.h"
#include "fd_forward.h"
#include "esp_log.h"

#include "printUtils.h"

static const char *TAG = "app_process";

//Docstrings and headers taken from mtmn.h
/**
 * @brief Forward the pnet process, coarse detection. Calculate in quantization.
 *
 * @param in        Image matrix, rgb888 format, size is 320x240
 * @return          Scores for every pixel, and box offset with respect.
 */

mtmn_net_t *pnet_lite_q(dl_matrix3du_t *in, dl_conv_mode mode) {
    ESP_LOGI(TAG, "Custom pnet_lite_q called!");

    int64_t time_start = esp_timer_get_time();
    
	dl_matrix3d_t *out_conv_1 = dl_matrix3duf_conv_common(in, &pnet_conv2d_kernel1, &pnet_conv2d_bias1, 1, 1, PADDING_VALID);
    dl_matrix3d_relu(out_conv_1);

    int64_t time_conv_1 = esp_timer_get_time();

    dl_matrix3d_t *out_pool_1 = dl_matrix3d_pooling(out_conv_1, 2, 2, 2, 2, PADDING_SAME, DL_POOLING_MAX);

    int64_t time_pool_1 = esp_timer_get_time();;

    dl_matrix3d_t *out_conv_2 = dl_matrix3dff_conv_common(out_pool_1, &pnet_conv2d_kernel2, &pnet_conv2d_bias2, 1, 1, PADDING_VALID);
    dl_matrix3d_relu(out_conv_2);

    int64_t time_conv_2 = esp_timer_get_time();

    dl_matrix3d_t *out_conv_3 = dl_matrix3dff_conv_common(out_conv_2, &pnet_conv2d_kernel3, &pnet_conv2d_bias3, 1, 1, PADDING_VALID);
    dl_matrix3d_relu(out_conv_3);

    int64_t time_conv_3 = esp_timer_get_time();

    dl_matrix3d_t *category = dl_matrix3dff_conv_common(out_conv_3, &pnet_conv2d_kernel4, &pnet_conv2d_bias4, 1, 1, PADDING_VALID);
    dl_matrix3d_softmax(category); //TODO: How to indicate that this should be done over axis 3?

    int64_t time_category = esp_timer_get_time();

    dl_matrix3d_t *offset = dl_matrix3dff_conv_common(out_conv_3, &pnet_conv2d_kernel5, &pnet_conv2d_bias5, 1, 1, PADDING_VALID);

    int64_t time_finish = esp_timer_get_time();
    
    ESP_LOGI(TAG, "pnet forward pass finished in %lld mu_s.", (time_finish - time_start));
    ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv_1 - time_start));
    ESP_LOGI(TAG, "pool_1 time: %lld mu_s.", (time_pool_1 - time_conv_1));
    ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv_2 - time_pool_1));
    ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv_3 - time_conv_2));
    ESP_LOGI(TAG, "category time: %lld mu_s.", (time_category - time_conv_3));
    ESP_LOGI(TAG, "offset time: %lld mu_s.", (time_finish - time_category));

    //TODO: Call free as soon as tensors are no longer live
    dl_matrix3d_free(out_conv_1);
    //TODO: The following statement causes problems
    //dl_matrix3d_free(out_pool_1);
    dl_matrix3d_free(out_conv_2);
    dl_matrix3d_free(out_conv_3);

    mtmn_net_t *result = (mtmn_net_t*) dl_lib_calloc(1, sizeof *result, 0);
    result->category = category;
    result->offset = offset;
    result->landmark = 0;

	return result;
}

/**
 * @brief Forward the rnet process, fine determine the boxes from pnet. Calculate in quantization.
 *
 * @param in        Image matrix, rgb888 format
 * @param threshold Score threshold to detect human face
 * @return          Scores for every box, and box offset with respect.
 */
mtmn_net_t *rnet_lite_q_with_score_verify(dl_matrix3du_t *in, float threshold, dl_conv_mode mode) {
    ESP_LOGI(TAG, "Custom rnet_lite_q_with_score_verify called!");

    int64_t time_start = esp_timer_get_time();

    dl_matrix3d_t *out_conv_1 = dl_matrix3duf_conv_common(in, &rnet_conv2d_kernel1, &rnet_conv2d_bias1, 1, 1, PADDING_VALID);
    dl_matrix3d_relu(out_conv_1);

    int64_t time_conv_1 = esp_timer_get_time();

    dl_matrix3d_t *out_pool_1 = dl_matrix3d_pooling(out_conv_1, 2, 2, 2, 2, PADDING_SAME, DL_POOLING_MAX);

    int64_t time_pool_1 = esp_timer_get_time();

    dl_matrix3d_t *out_conv_2 = dl_matrix3dff_conv_common(out_pool_1, &rnet_conv2d_kernel2, &rnet_conv2d_bias2, 1, 1, PADDING_VALID);
    dl_matrix3d_relu(out_conv_2);

    int64_t time_conv_2 = esp_timer_get_time();

    dl_matrix3d_t *out_pool_2 = dl_matrix3d_pooling(out_conv_2, 2, 2, 2, 2, PADDING_VALID, DL_POOLING_MAX);
    
    int64_t time_pool_2 = esp_timer_get_time();

    dl_matrix3d_t *out_conv_3 = dl_matrix3dff_conv_common(out_pool_2, &rnet_conv2d_kernel3, &rnet_conv2d_bias3, 1, 1, PADDING_VALID);
    dl_matrix3d_relu(out_conv_3);

    int64_t time_conv_3 = esp_timer_get_time();

    //flatten out_conv_3 for matrix multiplication
    out_conv_3->c = out_conv_3->h*out_conv_3->w*out_conv_3->c;
    out_conv_3->h = 1;
    out_conv_3->w = 1;
    
    dl_matrix3d_t *out_dense_1 = dl_matrix3d_alloc(1, 1, 1, rnet_dense_kernel1.h);
    dl_matrix3dff_fc_with_bias(out_dense_1, out_conv_3, &rnet_dense_kernel1, &rnet_dense_bias1);
    dl_matrix3d_relu(out_dense_1);

    int64_t time_dense_1 = esp_timer_get_time();

    dl_matrix3d_t *category = dl_matrix3d_alloc(1, 1, 1, rnet_dense_kernel2.h);
    dl_matrix3dff_fc_with_bias(category, out_dense_1, &rnet_dense_kernel2, &rnet_dense_bias2);
    dl_matrix3d_softmax(category);
    
    int64_t time_category = esp_timer_get_time();

    dl_matrix3d_t *offset = dl_matrix3d_alloc(1, 1, 1, rnet_dense_kernel3.h);
    dl_matrix3dff_fc_with_bias(offset, out_dense_1, &rnet_dense_kernel3, &rnet_dense_bias3);

    int64_t time_finish = esp_timer_get_time();
    ESP_LOGI(TAG, "rnet forward pass finished in %lld mu_s.", (time_finish - time_start));
    ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv_1 - time_start));
    ESP_LOGI(TAG, "pool_1 time: %lld mu_s.", (time_pool_1 - time_conv_1));
    ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv_2 - time_pool_1));
    ESP_LOGI(TAG, "pool_2 time: %lld mu_s.", (time_pool_2 - time_conv_2));
    ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv_3 - time_pool_2));
    ESP_LOGI(TAG, "dense_1 time: %lld mu_s.", (time_dense_1 - time_conv_3));
    ESP_LOGI(TAG, "category time: %lld mu_s.", (time_category - time_dense_1));
    ESP_LOGI(TAG, "offset time: %lld mu_s.", (time_finish - time_category));

    //TODO: Call free as soon as tensors are no longer live
    dl_matrix3d_free(out_conv_1);
    //dl_matrix3d_free(out_pool_1);
    dl_matrix3d_free(out_conv_2);
    //dl_matrix3d_free(out_pool_2);
    dl_matrix3d_free(out_conv_3);
    dl_matrix3d_free(out_dense_1);

    if (category->item[0] < threshold) {
        dl_matrix3d_free(category);
        dl_matrix3d_free(offset);
        return 0;
    }

    mtmn_net_t *result = (mtmn_net_t*) dl_lib_calloc(1, sizeof *result, 0);
    result->category = category;
    result->offset = offset;
    result->landmark = 0;

    return result;
}

/**
 * @brief Forward the onet process, fine determine the boxes from rnet. Calculate in quantization.
 *
 * @param in        Image matrix, rgb888 format
 * @param threshold Score threshold to detect human face
 * @return          Scores for every box, box offset, and landmark with respect.
 */
mtmn_net_t *onet_lite_q_with_score_verify(dl_matrix3du_t *in, float threshold, dl_conv_mode mode) {
    ESP_LOGI(TAG, "Custom onet_lite_q_with_score_verify called!");

    int64_t time_start = esp_timer_get_time();

    dl_matrix3d_t *out_conv_1 = dl_matrix3duf_conv_common(in, &onet_conv2d_kernel1, &onet_conv2d_bias1, 1, 1, PADDING_VALID);
    dl_matrix3d_relu(out_conv_1);
    
    int64_t time_conv_1 = esp_timer_get_time();

    dl_matrix3d_t *out_pool_1 = dl_matrix3d_pooling(out_conv_1, 3, 3, 2, 2, PADDING_SAME, DL_POOLING_MAX);

    int64_t time_pool_1 = esp_timer_get_time();

    dl_matrix3d_t *out_conv_2 = dl_matrix3dff_conv_common(out_pool_1, &onet_conv2d_kernel2, &onet_conv2d_bias2, 1, 1, PADDING_VALID);
    dl_matrix3d_relu(out_conv_2);

    int64_t time_conv_2 = esp_timer_get_time();

    dl_matrix3d_t *out_pool_2 = dl_matrix3d_pooling(out_conv_2, 3, 3, 2, 2, PADDING_VALID, DL_POOLING_MAX);

    int64_t time_pool_2 = esp_timer_get_time();

    dl_matrix3d_t *out_conv_3 = dl_matrix3dff_conv_common(out_pool_2, &onet_conv2d_kernel3, &onet_conv2d_bias3, 1, 1, PADDING_VALID);
    dl_matrix3d_relu(out_conv_3);

    int64_t time_conv_3 = esp_timer_get_time();

    dl_matrix3d_t *out_pool_3 = dl_matrix3d_pooling(out_conv_3, 2, 2, 2, 2, PADDING_SAME, DL_POOLING_MAX);

    int64_t time_pool_3 = esp_timer_get_time();

    dl_matrix3d_t *out_conv_4 = dl_matrix3dff_conv_common(out_pool_3, &onet_conv2d_kernel4, &onet_conv2d_bias4, 1, 1, PADDING_VALID);
    dl_matrix3d_relu(out_conv_4);

    int64_t time_conv_4 = esp_timer_get_time();

    //flatten out_conv_4 for matrix multiplication
    out_conv_4->c = out_conv_4->h*out_conv_4->w*out_conv_4->c;
    out_conv_4->h = 1;
    out_conv_4->w = 1;
    
    dl_matrix3d_t *out_dense_1 = dl_matrix3d_alloc(1, 1, 1, onet_dense_kernel1.h);
    dl_matrix3dff_fc_with_bias(out_dense_1, out_conv_4, &onet_dense_kernel1, &onet_dense_bias1);
    dl_matrix3d_relu(out_dense_1);

    int64_t time_dense_1 = esp_timer_get_time();

    dl_matrix3d_t *category = dl_matrix3d_alloc(1, 1, 1, onet_dense_kernel2.h);
    dl_matrix3dff_fc_with_bias(category, out_dense_1, &onet_dense_kernel2, &onet_dense_bias2);
    dl_matrix3d_softmax(category);
    
    int64_t time_category = esp_timer_get_time();

    dl_matrix3d_t *offset = dl_matrix3d_alloc(1, 1, 1, onet_dense_kernel3.h);
    dl_matrix3dff_fc_with_bias(offset, out_dense_1, &onet_dense_kernel3, &onet_dense_bias3);

    int64_t time_offset = esp_timer_get_time();

    dl_matrix3d_t *landmark = dl_matrix3d_alloc(1, 1, 1, onet_dense_kernel4.h);
    dl_matrix3dff_fc_with_bias(landmark, out_dense_1, &onet_dense_kernel4, &onet_dense_bias4);

    int64_t time_finish = esp_timer_get_time();
    ESP_LOGI(TAG, "onet forward pass finished in %lld mu_s.", (time_finish - time_start));
    ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv_1 - time_start));
    ESP_LOGI(TAG, "pool_1 time: %lld mu_s.", (time_pool_1 - time_conv_1));
    ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv_2 - time_pool_1));
    ESP_LOGI(TAG, "pool_2 time: %lld mu_s.", (time_pool_2 - time_conv_2));
    ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv_3 - time_pool_2));
    ESP_LOGI(TAG, "pool_3 time: %lld mu_s.", (time_pool_3 - time_conv_3));
    ESP_LOGI(TAG, "conv_4 time: %lld mu_s.", (time_conv_4 - time_pool_3));
    ESP_LOGI(TAG, "dense_1 time: %lld mu_s.", (time_dense_1 - time_conv_4));
    ESP_LOGI(TAG, "category time: %lld mu_s.", (time_category - time_dense_1));
    ESP_LOGI(TAG, "offset time: %lld mu_s.", (time_offset - time_category));
    ESP_LOGI(TAG, "landmark time: %lld mu_s.", (time_finish - time_offset));

    //TODO: Call free as soon as tensors are no longer live
    dl_matrix3d_free(out_conv_1);
    //dl_matrix3d_free(out_pool_1);
    dl_matrix3d_free(out_conv_2);
    //dl_matrix3d_free(out_pool_2);
    dl_matrix3d_free(out_conv_3);
    dl_matrix3d_free(out_dense_1);

    if (category->item[0] < threshold) {
        dl_matrix3d_free(category);
        dl_matrix3d_free(offset);
        dl_matrix3d_free(landmark);
        return 0;
    }

    mtmn_net_t *result = (mtmn_net_t*) dl_lib_calloc(1, sizeof *result, 0);
    result->category = category;
    result->offset = offset;
    result->landmark = landmark;

    return result;
}