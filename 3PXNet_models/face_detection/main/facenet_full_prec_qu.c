#include "facenet_full_prec_qu.h"
#include "dl_lib_matrix3dq.h"
#include "mtmn.h"
#include "fd_forward.h"
#include "esp_log.h"

//#include "printUtils.h"

static const char *TAG = "app_process";

const int EXP_TODO = 0;

//Docstrings and headers taken from mtmn.h
/**
 * @brief Forward the pnet process, coarse detection. Calculate in quantization.
 *
 * @param in        Image matrix, rgb888 format, size is 320x240
 * @return          Scores for every pixel, and box offset with respect.
 */

mtmn_net_t *pnet_lite_q_custom(dl_matrix3du_t *in, dl_conv_mode mode) {
    ESP_LOGI(TAG, "Custom pnet_lite_q called!");

    int64_t time_start = esp_timer_get_time();

	dl_matrix3dq_t *out_conv_1 = dl_matrix3duq_conv_common(in, &pnet_conv2d_kernel1_q, &pnet_conv2d_bias1_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_1);

    int64_t time_conv_1 = esp_timer_get_time();

    dl_matrix3dq_t *out_pool_1 = dl_matrix3dq_pooling(out_conv_1, 2, 2, 2, 2, PADDING_SAME, DL_POOLING_MAX);

    int64_t time_pool_1 = esp_timer_get_time();;

    dl_matrix3dq_t *out_conv_2 = dl_matrix3dqq_conv_common(out_pool_1, &pnet_conv2d_kernel2_q, &pnet_conv2d_bias2_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_2);

    int64_t time_conv_2 = esp_timer_get_time();

    dl_matrix3dq_t *out_conv_3 = dl_matrix3dqq_conv_common(out_conv_2, &pnet_conv2d_kernel3_q, &pnet_conv2d_bias3_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_3);

    int64_t time_conv_3 = esp_timer_get_time();

    dl_matrix3dq_t *category = dl_matrix3dqq_conv_common(out_conv_3, &pnet_conv2d_kernel4_q, &pnet_conv2d_bias4_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3d_t *category_f = dl_matrix3d_from_matrixq(category);
    dl_matrix3d_softmax(category_f); //TODO: How to indicate that this should be done over axis 3?

    int64_t time_category = esp_timer_get_time();

    dl_matrix3dq_t *offset = dl_matrix3dqq_conv_common(out_conv_3, &pnet_conv2d_kernel5_q, &pnet_conv2d_bias5_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3d_t *offset_f = dl_matrix3d_from_matrixq(offset);

    int64_t time_finish = esp_timer_get_time();
    
    ESP_LOGI(TAG, "pnet forward pass finished in %lld mu_s.", (time_finish - time_start));
    ESP_LOGI(TAG, "conv_1 time: %lld mu_s.", (time_conv_1 - time_start));
    ESP_LOGI(TAG, "pool_1 time: %lld mu_s.", (time_pool_1 - time_conv_1));
    ESP_LOGI(TAG, "conv_2 time: %lld mu_s.", (time_conv_2 - time_pool_1));
    ESP_LOGI(TAG, "conv_3 time: %lld mu_s.", (time_conv_3 - time_conv_2));
    ESP_LOGI(TAG, "category time: %lld mu_s.", (time_category - time_conv_3));
    ESP_LOGI(TAG, "offset time: %lld mu_s.", (time_finish - time_category));

    //TODO: Call free as soon as tensors are no longer live
    dl_matrix3dq_free(out_conv_1);
    //TODO: The following statement causes problems
    //dl_matrix3dq_free(out_pool_1);
    dl_matrix3dq_free(out_conv_2);
    dl_matrix3dq_free(out_conv_3);
    dl_matrix3dq_free(offset);
    dl_matrix3dq_free(category);

    mtmn_net_t *result = (mtmn_net_t*) dl_lib_calloc(1, sizeof *result, 0);
    result->category = category_f;
    result->offset = offset_f;
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
mtmn_net_t *rnet_lite_q_with_score_verify_custom(dl_matrix3du_t *in, float threshold, dl_conv_mode mode) {
    ESP_LOGI(TAG, "Custom rnet_lite_q_with_score_verify called!");

    int64_t time_start = esp_timer_get_time();

    dl_matrix3dq_t *out_conv_1 = dl_matrix3duq_conv_common(in, &rnet_conv2d_kernel1_q, &rnet_conv2d_bias1_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_1);

    int64_t time_conv_1 = esp_timer_get_time();

    dl_matrix3dq_t *out_pool_1 = dl_matrix3dq_pooling(out_conv_1, 2, 2, 2, 2, PADDING_SAME, DL_POOLING_MAX);

    int64_t time_pool_1 = esp_timer_get_time();

    dl_matrix3dq_t *out_conv_2 = dl_matrix3dqq_conv_common(out_pool_1, &rnet_conv2d_kernel2_q, &rnet_conv2d_bias2_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_2);

    int64_t time_conv_2 = esp_timer_get_time();

    dl_matrix3dq_t *out_pool_2 = dl_matrix3dq_pooling(out_conv_2, 2, 2, 2, 2, PADDING_VALID, DL_POOLING_MAX);
    
    int64_t time_pool_2 = esp_timer_get_time();

    dl_matrix3dq_t *out_conv_3 = dl_matrix3dqq_conv_common(out_pool_2, &rnet_conv2d_kernel3_q, &rnet_conv2d_bias3_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_3);

    int64_t time_conv_3 = esp_timer_get_time();

    //flatten out_conv_3 for matrix multiplication
    out_conv_3->c = out_conv_3->h*out_conv_3->w*out_conv_3->c;
    out_conv_3->h = 1;
    out_conv_3->w = 1;
    
    dl_matrix3dq_t *out_dense_1 = dl_matrix3dq_alloc(1, 1, 1, rnet_dense_kernel1_q.h, EXP_TODO);
    dl_matrix3dqq_fc_with_bias(out_dense_1, out_conv_3, &rnet_dense_kernel1_q, &rnet_dense_bias1_q, mode, "out_dense_1");
    dl_matrix3dq_relu(out_dense_1);

    int64_t time_dense_1 = esp_timer_get_time();

    dl_matrix3dq_t *category = dl_matrix3dq_alloc(1, 1, 1, rnet_dense_kernel2_q.h, EXP_TODO);
    dl_matrix3dqq_fc_with_bias(category, out_dense_1, &rnet_dense_kernel2_q, &rnet_dense_bias2_q, mode, "category");
    dl_matrix3d_t *category_f = dl_matrix3d_from_matrixq(category);
    dl_matrix3d_softmax(category_f);
    
    int64_t time_category = esp_timer_get_time();

    dl_matrix3dq_t *offset = dl_matrix3dq_alloc(1, 1, 1, rnet_dense_kernel3_q.h, EXP_TODO);
    dl_matrix3dqq_fc_with_bias(offset, out_dense_1, &rnet_dense_kernel3_q, &rnet_dense_bias3_q, mode, "offset");
    dl_matrix3d_t *offset_f = dl_matrix3d_from_matrixq(offset);

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
    dl_matrix3dq_free(out_conv_1);
    //dl_matrix3dq_free(out_pool_1);
    dl_matrix3dq_free(out_conv_2);
    //dl_matrix3dq_free(out_pool_2);
    dl_matrix3dq_free(out_conv_3);
    dl_matrix3dq_free(out_dense_1);
    dl_matrix3dq_free(offset);
    dl_matrix3dq_free(category);

    if (category->item[0] < threshold) {
        dl_matrix3d_free(category_f);
        dl_matrix3d_free(offset_f);
        return 0;
    }

    mtmn_net_t *result = (mtmn_net_t*) dl_lib_calloc(1, sizeof *result, 0);
    result->category = category_f;
    result->offset = offset_f;
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
mtmn_net_t *onet_lite_q_with_score_verify_custom(dl_matrix3du_t *in, float threshold, dl_conv_mode mode) {
    ESP_LOGI(TAG, "Custom onet_lite_q_with_score_verify called!");

    int64_t time_start = esp_timer_get_time();

    dl_matrix3dq_t *out_conv_1 = dl_matrix3duq_conv_common(in, &onet_conv2d_kernel1_q, &onet_conv2d_bias1_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_1);
    
    int64_t time_conv_1 = esp_timer_get_time();

    dl_matrix3dq_t *out_pool_1 = dl_matrix3dq_pooling(out_conv_1, 3, 3, 2, 2, PADDING_SAME, DL_POOLING_MAX);

    int64_t time_pool_1 = esp_timer_get_time();

    dl_matrix3dq_t *out_conv_2 = dl_matrix3dqq_conv_common(out_pool_1, &onet_conv2d_kernel2_q, &onet_conv2d_bias2_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_2);

    int64_t time_conv_2 = esp_timer_get_time();

    dl_matrix3dq_t *out_pool_2 = dl_matrix3dq_pooling(out_conv_2, 3, 3, 2, 2, PADDING_VALID, DL_POOLING_MAX);

    int64_t time_pool_2 = esp_timer_get_time();

    dl_matrix3dq_t *out_conv_3 = dl_matrix3dqq_conv_common(out_pool_2, &onet_conv2d_kernel3_q, &onet_conv2d_bias3_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_3);

    int64_t time_conv_3 = esp_timer_get_time();

    dl_matrix3dq_t *out_pool_3 = dl_matrix3dq_pooling(out_conv_3, 2, 2, 2, 2, PADDING_SAME, DL_POOLING_MAX);

    int64_t time_pool_3 = esp_timer_get_time();

    dl_matrix3dq_t *out_conv_4 = dl_matrix3dqq_conv_common(out_pool_3, &onet_conv2d_kernel4_q, &onet_conv2d_bias4_q, 1, 1, PADDING_VALID, EXP_TODO, mode);
    dl_matrix3dq_relu(out_conv_4);

    int64_t time_conv_4 = esp_timer_get_time();

    //flatten out_conv_4 for matrix multiplication
    out_conv_4->c = out_conv_4->h*out_conv_4->w*out_conv_4->c;
    out_conv_4->h = 1;
    out_conv_4->w = 1;
    
    dl_matrix3dq_t *out_dense_1 = dl_matrix3dq_alloc(1, 1, 1, onet_dense_kernel1_q.h, EXP_TODO);
    dl_matrix3dqq_fc_with_bias(out_dense_1, out_conv_4, &onet_dense_kernel1_q, &onet_dense_bias1_q, mode, "out_dense_1");
    dl_matrix3dq_relu(out_dense_1);

    int64_t time_dense_1 = esp_timer_get_time();

    dl_matrix3dq_t *category = dl_matrix3dq_alloc(1, 1, 1, onet_dense_kernel2_q.h, EXP_TODO);
    dl_matrix3dqq_fc_with_bias(category, out_dense_1, &onet_dense_kernel2_q, &onet_dense_bias2_q, mode, "category");
    dl_matrix3d_t *category_f = dl_matrix3d_from_matrixq(category);
    dl_matrix3d_softmax(category_f);
    
    int64_t time_category = esp_timer_get_time();

    dl_matrix3dq_t *offset = dl_matrix3dq_alloc(1, 1, 1, onet_dense_kernel3_q.h, EXP_TODO);
    dl_matrix3dqq_fc_with_bias(offset, out_dense_1, &onet_dense_kernel3_q, &onet_dense_bias3_q, mode, "offset");
    dl_matrix3d_t *offset_f = dl_matrix3d_from_matrixq(offset);

    int64_t time_offset = esp_timer_get_time();

    dl_matrix3dq_t *landmark = dl_matrix3dq_alloc(1, 1, 1, onet_dense_kernel4_q.h, EXP_TODO);
    dl_matrix3dqq_fc_with_bias(landmark, out_dense_1, &onet_dense_kernel4_q, &onet_dense_bias4_q, mode, "landmark");
    dl_matrix3d_t *landmark_f = dl_matrix3d_from_matrixq(landmark);

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
    dl_matrix3dq_free(out_conv_1);
    //dl_matrix3dq_free(out_pool_1);
    dl_matrix3dq_free(out_conv_2);
    //dl_matrix3dq_free(out_pool_2);
    dl_matrix3dq_free(out_conv_3);
    dl_matrix3dq_free(out_dense_1);
    dl_matrix3dq_free(offset);
    dl_matrix3dq_free(category);
    dl_matrix3dq_free(landmark);

    if (category->item[0] < threshold) {
        dl_matrix3d_free(category_f);
        dl_matrix3d_free(offset_f);
        dl_matrix3d_free(landmark_f);
        return 0;
    }

    mtmn_net_t *result = (mtmn_net_t*) dl_lib_calloc(1, sizeof *result, 0);
    result->category = category_f;
    result->offset = offset_f;
    result->landmark = landmark_f;

    return result;
}