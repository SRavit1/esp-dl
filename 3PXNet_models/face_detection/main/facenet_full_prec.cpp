#include "dl_lib_matrix3d.h"
#include "mtmn.h"
#include "facenet.h"

//Docstrings and headers taken from mtmn.h
/**
 * @brief Forward the pnet process, coarse detection. Calculate in float.
 *
 * @param in        Image matrix, rgb888 format, size is 320x240
 * @return          Scores for every pixel, and box offset with respect.
 */
mtmn_net_t *pnet_lite_f(dl_matrix3du_t *in) {
    ESP_LOGI(TAG, "pnet_lite_f called.");
    
    dl_matrix3d_t *category=dl_matrix3d_alloc(1, 1, 1, 2), *offset=dl_matrix3d_alloc(1, 1, 1, 4);
    
    pnet_lite_f_esp(in, category, offset);

    mtmn_net_t *result = (mtmn_net_t*) dl_lib_calloc(1, sizeof *result, 0);
    result->category = category;
    result->offset = offset;
    result->landmark = 0;

	return result;
}

/**
 * @brief Forward the rnet process, fine determine the boxes from pnet. Calculate in float.
 *
 * @param in        Image matrix, rgb888 format
 * @param threshold Score threshold to detect human face
 * @return          Scores for every box, and box offset with respect.
 */
mtmn_net_t *rnet_lite_f_with_score_verify(dl_matrix3du_t *in, float threshold) {
    dl_matrix3d_t *category=dl_matrix3d_alloc(1, 1, 1, 2), *offset=dl_matrix3d_alloc(1, 1, 1, 4);
    
    rnet_lite_f_with_score_verify_esp(in, category, offset);

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
 * @brief Forward the onet process, fine determine the boxes from rnet. Calculate in float.
 *
 * @param in        Image matrix, rgb888 format
 * @param threshold Score threshold to detect human face
 * @return          Scores for every box, box offset, and landmark with respect.
 */
mtmn_net_t *onet_lite_f_with_score_verify(dl_matrix3du_t *in, float threshold) {
    dl_matrix3d_t *category=dl_matrix3d_alloc(1, 1, 1, 2), *offset=dl_matrix3d_alloc(1, 1, 1, 4), *landmark=dl_matrix3d_alloc(1, 1, 1, 10);
    onet_lite_f_with_score_verify_esp(in, category, offset, landmark);

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