#include "facenet_full_prec_qu.h"
#include "dl_lib_matrix3dq.h"
#include "mtmn.h"
#include "fd_forward.h"
#include "esp_log.h"
#include "facenet.h"

//#include "printUtils.h"

//Docstrings and headers taken from mtmn.h
/**
 * @brief Forward the pnet process, coarse detection. Calculate in quantization.
 *
 * @param in        Image matrix, rgb888 format, size is 320x240
 * @return          Scores for every pixel, and box offset with respect.
 */

mtmn_net_t *pnet_lite_q_custom(dl_matrix3du_t *in, dl_conv_mode mode) {
    ESP_LOGI(TAG, "Custom pnet_lite_q called!");

    dl_matrix3d_t *category=dl_matrix3d_alloc(1, 1, 1, 2), *offset=dl_matrix3d_alloc(1, 1, 1, 4);

    int res = pnet_lite_q_esp(in, category, offset, mode);
    if (res) return 0;

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
mtmn_net_t *rnet_lite_q_with_score_verify_custom(dl_matrix3du_t *in, float threshold, dl_conv_mode mode) {
    ESP_LOGI(TAG, "Custom rnet_lite_q_with_score_verify called!");

    dl_matrix3d_t *category=dl_matrix3d_alloc(1, 1, 1, 2), *offset=dl_matrix3d_alloc(1, 1, 1, 4);

    int res = rnet_lite_q_with_score_verify_esp(in, category, offset, mode, threshold);
    if (res) return 0;

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
mtmn_net_t *onet_lite_q_with_score_verify_custom(dl_matrix3du_t *in, float threshold, dl_conv_mode mode) {
    ESP_LOGI(TAG, "Custom onet_lite_q_with_score_verify called!");

    dl_matrix3d_t *category=dl_matrix3d_alloc(1, 1, 1, 2), *offset=dl_matrix3d_alloc(1, 1, 1, 4), *landmark=dl_matrix3d_alloc(1, 1, 1, 10);

    int res = onet_lite_q_with_score_verify_esp(in, category, offset, landmark, mode, threshold);
    if (res) return 0;

    mtmn_net_t *result = (mtmn_net_t*) dl_lib_calloc(1, sizeof *result, 0);
    result->category = category;
    result->offset = offset;
    result->landmark = landmark;

    return result;
}